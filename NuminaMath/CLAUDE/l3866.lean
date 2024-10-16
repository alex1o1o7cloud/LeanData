import Mathlib

namespace NUMINAMATH_CALUDE_prob_b_greater_a_l3866_386695

-- Define the sets for a and b
def A : Finset ℕ := {1, 2, 3, 4, 5}
def B : Finset ℕ := {1, 2, 3}

-- Define the event space
def Ω : Finset (ℕ × ℕ) := A.product B

-- Define the favorable event (b > a)
def E : Finset (ℕ × ℕ) := Ω.filter (fun p => p.2 > p.1)

-- Theorem statement
theorem prob_b_greater_a :
  (E.card : ℚ) / Ω.card = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_prob_b_greater_a_l3866_386695


namespace NUMINAMATH_CALUDE_min_value_of_f_l3866_386622

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x

theorem min_value_of_f :
  ∃ (x : ℝ), x ∈ Set.Icc 0 3 ∧
  f x = -20 ∧
  ∀ (y : ℝ), y ∈ Set.Icc 0 3 → f y ≥ f x :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3866_386622


namespace NUMINAMATH_CALUDE_smallest_solution_of_equation_l3866_386642

theorem smallest_solution_of_equation (x : ℚ) :
  (7 * (10 * x^2 + 10 * x + 11) = x * (10 * x - 45)) →
  x ≥ -7/5 :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_of_equation_l3866_386642


namespace NUMINAMATH_CALUDE_stamps_per_ounce_l3866_386616

/-- Given a letter with 8 pieces of paper each weighing 1/5 ounce,
    an envelope weighing 2/5 ounce, and requiring 2 stamps total,
    prove that 1 stamp is needed per ounce. -/
theorem stamps_per_ounce (paper_weight : ℚ) (envelope_weight : ℚ) (total_stamps : ℕ) :
  paper_weight = 1/5
  → envelope_weight = 2/5
  → total_stamps = 2
  → (total_stamps : ℚ) / (8 * paper_weight + envelope_weight) = 1 := by
  sorry

end NUMINAMATH_CALUDE_stamps_per_ounce_l3866_386616


namespace NUMINAMATH_CALUDE_combined_tax_rate_l3866_386621

/-- Combined tax rate calculation -/
theorem combined_tax_rate 
  (john_tax_rate : ℝ) 
  (ingrid_tax_rate : ℝ) 
  (john_income : ℝ) 
  (ingrid_income : ℝ) 
  (h1 : john_tax_rate = 0.3)
  (h2 : ingrid_tax_rate = 0.4)
  (h3 : john_income = 58000)
  (h4 : ingrid_income = 72000) :
  (john_tax_rate * john_income + ingrid_tax_rate * ingrid_income) / (john_income + ingrid_income) = 
  (0.3 * 58000 + 0.4 * 72000) / (58000 + 72000) := by
  sorry

end NUMINAMATH_CALUDE_combined_tax_rate_l3866_386621


namespace NUMINAMATH_CALUDE_evaluate_expression_l3866_386662

theorem evaluate_expression : 
  (128 : ℝ)^(1/3) * (729 : ℝ)^(1/2) = 108 * 2^(1/3) :=
by
  -- Definitions based on given conditions
  have h1 : (128 : ℝ) = 2^7 := by sorry
  have h2 : (729 : ℝ) = 3^6 := by sorry
  
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3866_386662


namespace NUMINAMATH_CALUDE_abs_value_sum_l3866_386644

theorem abs_value_sum (a b c : ℝ) : 
  abs a = 1 → abs b = 2 → abs c = 3 → a > b → b > c → a + b - c = 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_value_sum_l3866_386644


namespace NUMINAMATH_CALUDE_exponential_function_fixed_point_l3866_386674

theorem exponential_function_fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  a^(0 : ℝ) = 1 := by sorry

end NUMINAMATH_CALUDE_exponential_function_fixed_point_l3866_386674


namespace NUMINAMATH_CALUDE_ratio_a_to_b_l3866_386628

-- Define an arithmetic sequence
def is_arithmetic_sequence (s : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, s (n + 1) - s n = d

-- Define our specific sequence
def our_sequence (s : ℕ → ℝ) (y : ℝ) : Prop :=
  s 0 = s 1 - y ∧ s 1 = y ∧ s 2 = s 1 + y ∧ s 3 = 3 * y

theorem ratio_a_to_b (s : ℕ → ℝ) (y : ℝ) :
  is_arithmetic_sequence s → our_sequence s y → s 0 / s 2 = 0 := by
  sorry


end NUMINAMATH_CALUDE_ratio_a_to_b_l3866_386628


namespace NUMINAMATH_CALUDE_tangent_slope_at_point_2_10_l3866_386648

-- Define the function f(x) = x^2 + 3x
def f (x : ℝ) : ℝ := x^2 + 3*x

-- Define the derivative of f(x)
def f_derivative (x : ℝ) : ℝ := 2*x + 3

-- Theorem statement
theorem tangent_slope_at_point_2_10 :
  f_derivative 2 = 7 :=
sorry

end NUMINAMATH_CALUDE_tangent_slope_at_point_2_10_l3866_386648


namespace NUMINAMATH_CALUDE_optimal_strategy_l3866_386611

/-- Represents the profit function for zongzi sales -/
def profit_function (x : ℝ) (a : ℝ) : ℝ := (a - 5) * x + 6000

/-- Represents the constraints on the number of boxes of type A zongzi -/
def valid_quantity (x : ℝ) : Prop := 100 ≤ x ∧ x ≤ 150

/-- Theorem stating the optimal purchasing strategy to maximize profit -/
theorem optimal_strategy (a : ℝ) (h1 : 0 < a) (h2 : a < 10) :
  (0 < a ∧ a < 5 → 
    ∀ x, valid_quantity x → profit_function 100 a ≥ profit_function x a) ∧
  (5 ≤ a ∧ a < 10 → 
    ∀ x, valid_quantity x → profit_function 150 a ≥ profit_function x a) :=
sorry

end NUMINAMATH_CALUDE_optimal_strategy_l3866_386611


namespace NUMINAMATH_CALUDE_jacksons_grade_l3866_386647

/-- Calculates a student's grade based on study time and grade increase rate -/
def calculate_grade (video_game_hours : ℝ) (study_time_ratio : ℝ) (grade_increase_rate : ℝ) : ℝ :=
  video_game_hours * study_time_ratio * grade_increase_rate

/-- Proves that Jackson's grade is 45 points given the problem conditions -/
theorem jacksons_grade :
  let video_game_hours : ℝ := 9
  let study_time_ratio : ℝ := 1/3
  let grade_increase_rate : ℝ := 15
  calculate_grade video_game_hours study_time_ratio grade_increase_rate = 45 := by
  sorry


end NUMINAMATH_CALUDE_jacksons_grade_l3866_386647


namespace NUMINAMATH_CALUDE_fifth_island_not_maya_l3866_386677

-- Define the types of residents
inductive Resident
| Knight
| Liar

-- Define the possible island names
inductive IslandName
| Maya
| NotMaya

-- Define the statements made by A and B
def statement_A (resident_A resident_B : Resident) (island : IslandName) : Prop :=
  (resident_A = Resident.Liar ∧ resident_B = Resident.Liar) ∧ island = IslandName.Maya

def statement_B (resident_A resident_B : Resident) (island : IslandName) : Prop :=
  (resident_A = Resident.Knight ∨ resident_B = Resident.Knight) ∧ island = IslandName.NotMaya

-- Define the truthfulness of statements based on the resident type
def is_truthful (r : Resident) (s : Prop) : Prop :=
  (r = Resident.Knight ∧ s) ∨ (r = Resident.Liar ∧ ¬s)

-- Theorem statement
theorem fifth_island_not_maya :
  ∀ (resident_A resident_B : Resident) (island : IslandName),
    is_truthful resident_A (statement_A resident_A resident_B island) →
    is_truthful resident_B (statement_B resident_A resident_B island) →
    island = IslandName.NotMaya :=
sorry

end NUMINAMATH_CALUDE_fifth_island_not_maya_l3866_386677


namespace NUMINAMATH_CALUDE_sqrt_4_squared_times_5_to_6th_l3866_386692

theorem sqrt_4_squared_times_5_to_6th : Real.sqrt (4^2 * 5^6) = 500 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_4_squared_times_5_to_6th_l3866_386692


namespace NUMINAMATH_CALUDE_solution_set_f_range_of_m_l3866_386681

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x| + |x + 1|

-- Theorem for the solution set of f(x) > 3
theorem solution_set_f : 
  {x : ℝ | f x > 3} = {x : ℝ | x > 1 ∨ x < -2} := by sorry

-- Theorem for the range of m
theorem range_of_m : 
  (∀ x : ℝ, m^2 + 3*m + 2*f x ≥ 0) → (m ≥ -1 ∨ m ≤ -2) := by sorry

end NUMINAMATH_CALUDE_solution_set_f_range_of_m_l3866_386681


namespace NUMINAMATH_CALUDE_max_guests_l3866_386617

/-- Represents a menu choice as a quadruple of integers -/
structure MenuChoice (n : ℕ) where
  starter : Fin n
  main : Fin n
  dessert : Fin n
  wine : Fin n

/-- The set of all valid menu choices -/
def validMenus (n : ℕ) : Finset (MenuChoice n) :=
  sorry

theorem max_guests (n : ℕ) (h : n > 0) :
  (Finset.card (validMenus n) : ℕ) = n^4 - n^3 ∧
  ∀ (S : Finset (MenuChoice n)), Finset.card S > n^4 - n^3 →
    ∃ (T : Finset (MenuChoice n)), Finset.card T = n ∧ T ⊆ S ∧
      (∃ (i : Fin 4), ∀ (x y : MenuChoice n), x ∈ T → y ∈ T → x ≠ y →
        (i.val = 0 → x.starter = y.starter) ∧
        (i.val = 1 → x.main = y.main) ∧
        (i.val = 2 → x.dessert = y.dessert) ∧
        (i.val = 3 → x.wine = y.wine)) :=
  sorry

end NUMINAMATH_CALUDE_max_guests_l3866_386617


namespace NUMINAMATH_CALUDE_marias_test_scores_l3866_386694

/-- Maria's test scores problem -/
theorem marias_test_scores 
  (score1 score2 score3 : ℝ) 
  (h1 : (score1 + score2 + score3 + 100) / 4 = 85) :
  score1 + score2 + score3 = 240 := by
  sorry

end NUMINAMATH_CALUDE_marias_test_scores_l3866_386694


namespace NUMINAMATH_CALUDE_multiple_exists_l3866_386678

theorem multiple_exists (n : ℕ) (a : Fin (n + 1) → ℕ) 
  (h : ∀ i, 0 < a i ∧ a i ≤ 2*n) : 
  ∃ i j, i ≠ j ∧ (a i ∣ a j ∨ a j ∣ a i) := by
  sorry

end NUMINAMATH_CALUDE_multiple_exists_l3866_386678


namespace NUMINAMATH_CALUDE_nancy_pearl_beads_difference_l3866_386664

/-- Prove that Nancy has 60 more pearl beads than metal beads -/
theorem nancy_pearl_beads_difference (beads_per_bracelet : ℕ) 
  (total_bracelets : ℕ) (nancy_metal_beads : ℕ) (rose_crystal_beads : ℕ) :
  beads_per_bracelet = 8 →
  total_bracelets = 20 →
  nancy_metal_beads = 40 →
  rose_crystal_beads = 20 →
  ∃ (nancy_pearl_beads : ℕ),
    nancy_pearl_beads = beads_per_bracelet * total_bracelets - 
      (nancy_metal_beads + rose_crystal_beads + 2 * rose_crystal_beads) ∧
    nancy_pearl_beads - nancy_metal_beads = 60 :=
by sorry

end NUMINAMATH_CALUDE_nancy_pearl_beads_difference_l3866_386664


namespace NUMINAMATH_CALUDE_range_of_a_l3866_386653

theorem range_of_a (a : ℝ) : 
  (∃ (x₁ x₂ x₃ : ℤ), x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    (∀ (x : ℤ), (x + 6 < 2 + 3*x ∧ (a + x) / 4 > x) ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃))) →
  (15 < a ∧ a ≤ 18) := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l3866_386653


namespace NUMINAMATH_CALUDE_punch_mixture_difference_l3866_386638

/-- Proves that in a mixture with a 3:5 ratio of two components, 
    where the total volume is 72 cups, the difference between 
    the volumes of the two components is 18 cups. -/
theorem punch_mixture_difference (total_volume : ℕ) 
    (ratio_a : ℕ) (ratio_b : ℕ) (difference : ℕ) : 
    total_volume = 72 → 
    ratio_a = 3 → 
    ratio_b = 5 → 
    difference = ratio_b * (total_volume / (ratio_a + ratio_b)) - 
                 ratio_a * (total_volume / (ratio_a + ratio_b)) → 
    difference = 18 := by
  sorry

end NUMINAMATH_CALUDE_punch_mixture_difference_l3866_386638


namespace NUMINAMATH_CALUDE_base5_412_to_base7_l3866_386656

/-- Converts a base 5 number to decimal --/
def base5ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- Converts a decimal number to base 7 --/
def decimalToBase7 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec go (m : Nat) (acc : List Nat) :=
      if m = 0 then acc
      else go (m / 7) ((m % 7) :: acc)
    go n []

theorem base5_412_to_base7 :
  decimalToBase7 (base5ToDecimal [2, 1, 4]) = [2, 1, 2] :=
sorry

end NUMINAMATH_CALUDE_base5_412_to_base7_l3866_386656


namespace NUMINAMATH_CALUDE_quadratic_roots_unique_l3866_386636

theorem quadratic_roots_unique (b c : ℝ) : 
  ({1, 2} : Set ℝ) = {x | x^2 + b*x + c = 0} → b = -3 ∧ c = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_unique_l3866_386636


namespace NUMINAMATH_CALUDE_cube_root_equation_l3866_386699

theorem cube_root_equation (x : ℝ) : (x + 1)^3 = -27 → x = -4 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_l3866_386699


namespace NUMINAMATH_CALUDE_marble_collection_l3866_386661

theorem marble_collection (total : ℕ) (friend_total : ℕ) : 
  (40 : ℚ) / 100 * total + (20 : ℚ) / 100 * total + (40 : ℚ) / 100 * total = total →
  (40 : ℚ) / 100 * friend_total = 2 →
  friend_total = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_marble_collection_l3866_386661


namespace NUMINAMATH_CALUDE_swimming_club_members_l3866_386671

theorem swimming_club_members :
  ∃ (j s v : ℕ),
    j > 0 ∧ s > 0 ∧ v > 0 ∧
    3 * s = 2 * j ∧
    5 * v = 2 * s ∧
    j + s + v = 58 :=
by sorry

end NUMINAMATH_CALUDE_swimming_club_members_l3866_386671


namespace NUMINAMATH_CALUDE_class_mean_calculation_l3866_386645

theorem class_mean_calculation (total_students : ℕ) (group1_students : ℕ) (group2_students : ℕ)
  (group1_mean : ℚ) (group2_mean : ℚ) :
  total_students = group1_students + group2_students →
  group1_students = 32 →
  group2_students = 8 →
  group1_mean = 68 / 100 →
  group2_mean = 82 / 100 →
  (group1_students * group1_mean + group2_students * group2_mean) / total_students = 708 / 1000 :=
by sorry

end NUMINAMATH_CALUDE_class_mean_calculation_l3866_386645


namespace NUMINAMATH_CALUDE_number_of_observations_l3866_386606

theorem number_of_observations (original_mean new_mean : ℝ) 
  (original_value new_value : ℝ) (n : ℕ) : 
  original_mean = 36 → 
  new_mean = 36.5 → 
  original_value = 23 → 
  new_value = 44 → 
  n * original_mean + (new_value - original_value) = n * new_mean → 
  n = 42 := by
sorry

end NUMINAMATH_CALUDE_number_of_observations_l3866_386606


namespace NUMINAMATH_CALUDE_volleyball_team_starters_l3866_386640

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem volleyball_team_starters (total_players triplets twins starters : ℕ) 
  (h1 : total_players = 16)
  (h2 : triplets = 3)
  (h3 : twins = 2)
  (h4 : starters = 6) :
  (choose (total_players - triplets - twins) starters) + 
  (triplets * choose (total_players - triplets - twins) (starters - 1)) +
  (twins * choose (total_players - triplets - twins) (starters - 1)) = 2772 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_team_starters_l3866_386640


namespace NUMINAMATH_CALUDE_tuesday_sales_l3866_386619

/-- Proves the number of bottles sold on Tuesday given inventory and sales information --/
theorem tuesday_sales (initial_inventory : ℕ) (monday_sales : ℕ) (daily_sales : ℕ) 
  (saturday_delivery : ℕ) (final_inventory : ℕ) : 
  initial_inventory = 4500 →
  monday_sales = 2445 →
  daily_sales = 50 →
  saturday_delivery = 650 →
  final_inventory = 1555 →
  initial_inventory + saturday_delivery - monday_sales - (daily_sales * 5) - final_inventory = 900 := by
  sorry

end NUMINAMATH_CALUDE_tuesday_sales_l3866_386619


namespace NUMINAMATH_CALUDE_system_solution_l3866_386641

theorem system_solution : 
  ∃ (x y z u : ℚ), 
    (x = 229 ∧ y = 149 ∧ z = 131 ∧ u = 121) ∧
    (x + y = 3/2 * (z + u)) ∧
    (x + z = -4/3 * (y + u)) ∧
    (x + u = 5/4 * (y + z)) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3866_386641


namespace NUMINAMATH_CALUDE_segment_length_l3866_386672

/-- A rectangle with side lengths 4 and 6, divided into four equal parts by two segments emanating from one vertex -/
structure DividedRectangle where
  /-- The length of the rectangle -/
  length : ℝ
  /-- The width of the rectangle -/
  width : ℝ
  /-- The length of the first dividing segment -/
  segment1 : ℝ
  /-- The length of the second dividing segment -/
  segment2 : ℝ
  /-- The rectangle has side lengths 4 and 6 -/
  dim_constraint : length = 4 ∧ width = 6
  /-- The two segments divide the rectangle into four equal parts -/
  division_constraint : ∃ (a b c d : ℝ), a + b + c + d = length * width ∧ 
                        a = b ∧ b = c ∧ c = d

/-- The theorem stating that one of the dividing segments has length √18.25 -/
theorem segment_length (r : DividedRectangle) : r.segment1 = Real.sqrt 18.25 ∨ r.segment2 = Real.sqrt 18.25 := by
  sorry

end NUMINAMATH_CALUDE_segment_length_l3866_386672


namespace NUMINAMATH_CALUDE_square_of_binomial_form_l3866_386665

theorem square_of_binomial_form (x y : ℝ) :
  ∃ (a b : ℝ), (1/3 * x + y) * (y - 1/3 * x) = (a - b)^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_binomial_form_l3866_386665


namespace NUMINAMATH_CALUDE_mod_sum_powers_seven_l3866_386667

theorem mod_sum_powers_seven : (9^5 + 4^6 + 5^7) % 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_mod_sum_powers_seven_l3866_386667


namespace NUMINAMATH_CALUDE_valid_arrangements_ten_four_l3866_386657

/-- The number of ways to arrange n people in a row -/
def totalArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange k people in a row within a group -/
def groupArrangements (k : ℕ) : ℕ := Nat.factorial k

/-- The number of ways to arrange n people in a row, where k specific people are not allowed to sit in k consecutive seats -/
def validArrangements (n k : ℕ) : ℕ :=
  totalArrangements n - totalArrangements (n - k + 1) * groupArrangements k

theorem valid_arrangements_ten_four :
  validArrangements 10 4 = 3507840 := by sorry

end NUMINAMATH_CALUDE_valid_arrangements_ten_four_l3866_386657


namespace NUMINAMATH_CALUDE_min_distance_ABCD_l3866_386679

/-- Given four points A, B, C, and D on a line, with AB = 12, BC = 6, and CD = 5,
    the minimum possible distance between A and D is 1. -/
theorem min_distance_ABCD (A B C D : ℝ) : 
  abs (B - A) = 12 →
  abs (C - B) = 6 →
  abs (D - C) = 5 →
  ∃ (A' B' C' D' : ℝ), 
    abs (B' - A') = 12 ∧
    abs (C' - B') = 6 ∧
    abs (D' - C') = 5 ∧
    abs (D' - A') = 1 ∧
    ∀ (A'' B'' C'' D'' : ℝ),
      abs (B'' - A'') = 12 →
      abs (C'' - B'') = 6 →
      abs (D'' - C'') = 5 →
      abs (D'' - A'') ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_min_distance_ABCD_l3866_386679


namespace NUMINAMATH_CALUDE_fruit_basket_count_l3866_386602

/-- The number of fruit baskets -/
def num_baskets : ℕ := 4

/-- The number of apples in each of the first three baskets -/
def apples_per_basket : ℕ := 9

/-- The number of oranges in each of the first three baskets -/
def oranges_per_basket : ℕ := 15

/-- The number of bananas in each of the first three baskets -/
def bananas_per_basket : ℕ := 14

/-- The number of fruits that are reduced in the fourth basket -/
def reduction : ℕ := 2

/-- The total number of fruits in all baskets -/
def total_fruits : ℕ := 146

theorem fruit_basket_count :
  (3 * (apples_per_basket + oranges_per_basket + bananas_per_basket)) +
  ((apples_per_basket - reduction) + (oranges_per_basket - reduction) + (bananas_per_basket - reduction)) =
  total_fruits := by sorry

end NUMINAMATH_CALUDE_fruit_basket_count_l3866_386602


namespace NUMINAMATH_CALUDE_product_of_three_consecutive_integers_divisible_by_six_l3866_386627

theorem product_of_three_consecutive_integers_divisible_by_six (n : ℕ) (h : n > 0) :
  ∃ k : ℕ, n * (n + 1) * (n + 2) = 6 * k := by
sorry

end NUMINAMATH_CALUDE_product_of_three_consecutive_integers_divisible_by_six_l3866_386627


namespace NUMINAMATH_CALUDE_pythagorean_triple_divisibility_l3866_386670

theorem pythagorean_triple_divisibility (x y z : ℕ) (h : x^2 + y^2 = z^2) :
  3 ∣ x ∨ 3 ∣ y ∨ 3 ∣ z := by
  sorry

end NUMINAMATH_CALUDE_pythagorean_triple_divisibility_l3866_386670


namespace NUMINAMATH_CALUDE_shaded_sectors_ratio_l3866_386630

/-- Given three semicircular protractors with radii 1, 3, and 5,
    whose centers coincide and diameters align,
    prove that the ratio of the areas of the shaded sectors is 48 : 40 : 15 -/
theorem shaded_sectors_ratio (r₁ r₂ r₃ : ℝ) (S_A S_B S_C : ℝ) :
  r₁ = 1 → r₂ = 3 → r₃ = 5 →
  S_A = (π / 10) * (r₃^2 - r₂^2) →
  S_B = (π / 6) * (r₂^2 - r₁^2) →
  S_C = (π / 2) * r₁^2 →
  ∃ (k : ℝ), k > 0 ∧ S_A = 48 * k ∧ S_B = 40 * k ∧ S_C = 15 * k :=
by sorry

end NUMINAMATH_CALUDE_shaded_sectors_ratio_l3866_386630


namespace NUMINAMATH_CALUDE_intersection_point_on_lines_unique_intersection_l3866_386618

/-- The intersection point of two lines -/
def intersection_point : ℚ × ℚ := (39/17, 53/17)

/-- First line equation: 8x - 3y = 9 -/
def line1 (x y : ℚ) : Prop := 8*x - 3*y = 9

/-- Second line equation: 6x + 2y = 20 -/
def line2 (x y : ℚ) : Prop := 6*x + 2*y = 20

/-- Theorem stating that the intersection_point satisfies both line equations -/
theorem intersection_point_on_lines : 
  let (x, y) := intersection_point
  line1 x y ∧ line2 x y :=
by sorry

/-- Theorem stating that the intersection_point is the unique solution -/
theorem unique_intersection :
  ∀ (x y : ℚ), line1 x y ∧ line2 x y → (x, y) = intersection_point :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_on_lines_unique_intersection_l3866_386618


namespace NUMINAMATH_CALUDE_square_cut_in_half_l3866_386633

/-- A square with side length 8 is cut in half to create two congruent rectangles. -/
theorem square_cut_in_half (square_side : ℝ) (rect_width rect_height : ℝ) : 
  square_side = 8 →
  rect_width * rect_height = square_side * square_side / 2 →
  rect_width = square_side ∨ rect_height = square_side →
  (rect_width = 4 ∧ rect_height = 8) ∨ (rect_width = 8 ∧ rect_height = 4) := by
  sorry

end NUMINAMATH_CALUDE_square_cut_in_half_l3866_386633


namespace NUMINAMATH_CALUDE_figure2_total_length_l3866_386683

/-- A rectangle-like shape composed of perpendicular line segments -/
structure RectangleShape :=
  (left : ℝ)
  (bottom : ℝ)
  (right : ℝ)
  (top : ℝ)

/-- Calculate the total length of segments in the shape -/
def total_length (shape : RectangleShape) : ℝ :=
  shape.left + shape.bottom + shape.right + shape.top

/-- The theorem stating that the total length of segments in Figure 2 is 23 units -/
theorem figure2_total_length :
  let figure2 : RectangleShape := {
    left := 10,
    bottom := 5,
    right := 7,
    top := 1
  }
  total_length figure2 = 23 := by sorry

end NUMINAMATH_CALUDE_figure2_total_length_l3866_386683


namespace NUMINAMATH_CALUDE_cliff_rock_collection_l3866_386649

theorem cliff_rock_collection (igneous sedimentary : ℕ) : 
  igneous = sedimentary / 2 →
  igneous / 3 = 30 →
  igneous + sedimentary = 270 :=
by
  sorry

end NUMINAMATH_CALUDE_cliff_rock_collection_l3866_386649


namespace NUMINAMATH_CALUDE_smallest_AAB_l3866_386600

/-- Represents a digit from 1 to 9 -/
def Digit := {n : ℕ | 1 ≤ n ∧ n ≤ 9}

/-- Represents a two-digit integer AB -/
def TwoDigitInt (A B : Digit) : ℕ := 10 * A + B

/-- Represents a three-digit integer AAB -/
def ThreeDigitInt (A B : Digit) : ℕ := 100 * A + 10 * A + B

theorem smallest_AAB : 
  ∃ (A B : Digit), 
    A ≠ B ∧ 
    (TwoDigitInt A B : ℚ) = (1 / 7 : ℚ) * (ThreeDigitInt A B : ℚ) ∧
    ThreeDigitInt A B = 332 ∧
    (∀ (A' B' : Digit), 
      A' ≠ B' → 
      (TwoDigitInt A' B' : ℚ) = (1 / 7 : ℚ) * (ThreeDigitInt A' B' : ℚ) → 
      ThreeDigitInt A' B' ≥ 332) := by
  sorry

end NUMINAMATH_CALUDE_smallest_AAB_l3866_386600


namespace NUMINAMATH_CALUDE_emily_days_off_l3866_386650

/-- The number of holidays Emily took in a year -/
def total_holidays : ℕ := 24

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- The number of times Emily took a day off each month -/
def days_off_per_month : ℚ := total_holidays / months_in_year

theorem emily_days_off : days_off_per_month = 2 := by
  sorry

end NUMINAMATH_CALUDE_emily_days_off_l3866_386650


namespace NUMINAMATH_CALUDE_bills_score_l3866_386631

theorem bills_score (john sue bill : ℕ) 
  (h1 : bill = john + 20)
  (h2 : bill * 2 = sue)
  (h3 : john + bill + sue = 160) :
  bill = 45 := by
sorry

end NUMINAMATH_CALUDE_bills_score_l3866_386631


namespace NUMINAMATH_CALUDE_polynomial_characterization_l3866_386610

-- Define the polynomial type
def RealPolynomial := ℝ → ℝ

-- Define the condition for a, b, c
def SumProductZero (a b c : ℝ) : Prop := a * b + b * c + c * a = 0

-- Define the equality condition for the polynomial
def PolynomialCondition (P : RealPolynomial) : Prop :=
  ∀ (a b c : ℝ), SumProductZero a b c →
    P (a - b) + P (b - c) + P (c - a) = 2 * P (a + b + c)

-- Define the form of the polynomial we want to prove
def QuarticQuadraticForm (P : RealPolynomial) : Prop :=
  ∃ (α β : ℝ), ∀ (x : ℝ), P x = α * x^4 + β * x^2

-- The main theorem
theorem polynomial_characterization (P : RealPolynomial) :
  PolynomialCondition P → QuarticQuadraticForm P :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_characterization_l3866_386610


namespace NUMINAMATH_CALUDE_f_monotone_increasing_F_lower_bound_l3866_386696

noncomputable section

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (2 * x) - 4 * a * Real.exp x - 2 * a * x

def g (a : ℝ) (x : ℝ) : ℝ := x^2 + 5 * a^2

def F (a : ℝ) (x : ℝ) : ℝ := f a x + g a x

-- Theorem 1: f is monotonically increasing when a ≤ 0
theorem f_monotone_increasing (a : ℝ) :
  (∀ x : ℝ, Monotone (f a)) ↔ a ≤ 0 :=
sorry

-- Theorem 2: F has a lower bound
theorem F_lower_bound (a : ℝ) (x : ℝ) :
  F a x ≥ 4 * (1 - Real.log 2)^2 / 5 :=
sorry

end NUMINAMATH_CALUDE_f_monotone_increasing_F_lower_bound_l3866_386696


namespace NUMINAMATH_CALUDE_max_value_of_one_minus_sin_l3866_386601

theorem max_value_of_one_minus_sin (x : ℝ) : 
  ∃ (M : ℝ), M = 2 ∧ ∀ x, (1 - Real.sin x) ≤ M ∧ ∃ x₀, (1 - Real.sin x₀) = M :=
sorry

end NUMINAMATH_CALUDE_max_value_of_one_minus_sin_l3866_386601


namespace NUMINAMATH_CALUDE_root_implies_a_value_l3866_386635

theorem root_implies_a_value (a : ℝ) : (1 : ℝ)^2 - 2*(1 : ℝ) + a = 0 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_root_implies_a_value_l3866_386635


namespace NUMINAMATH_CALUDE_triangle_inequality_check_l3866_386609

theorem triangle_inequality_check (rods : Fin 100 → ℝ) 
  (h_sorted : ∀ i j : Fin 100, i ≤ j → rods i ≤ rods j) :
  (∀ i j k : Fin 100, i ≠ j ∧ j ≠ k ∧ i ≠ k → 
    rods i + rods j > rods k) ↔ 
  (rods 98 + rods 99 > rods 100) :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_check_l3866_386609


namespace NUMINAMATH_CALUDE_problem_statement_l3866_386615

theorem problem_statement (a b : ℝ) (h1 : a * b = 3) (h2 : a - 2 * b = 5) :
  a^2 * b - 2 * a * b^2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3866_386615


namespace NUMINAMATH_CALUDE_total_flowers_l3866_386663

def flower_collection (arwen_tulips arwen_roses : ℕ) : ℕ :=
  let elrond_tulips := 2 * arwen_tulips
  let elrond_roses := 3 * arwen_roses
  let galadriel_tulips := 3 * elrond_tulips
  let galadriel_roses := 2 * arwen_roses
  arwen_tulips + arwen_roses + elrond_tulips + elrond_roses + galadriel_tulips + galadriel_roses

theorem total_flowers : flower_collection 20 18 = 288 := by
  sorry

end NUMINAMATH_CALUDE_total_flowers_l3866_386663


namespace NUMINAMATH_CALUDE_lisa_marbles_theorem_distribution_satisfies_conditions_l3866_386675

/-- The minimum number of additional marbles needed -/
def additional_marbles_needed (num_friends : ℕ) (initial_marbles : ℕ) : ℕ :=
  let total_needed := (num_friends * (num_friends + 3)) / 2
  max (total_needed - initial_marbles) 0

/-- Proof that 40 additional marbles are needed for Lisa's scenario -/
theorem lisa_marbles_theorem :
  additional_marbles_needed 12 50 = 40 := by
  sorry

/-- Verify that the distribution satisfies the conditions -/
theorem distribution_satisfies_conditions 
  (num_friends : ℕ) 
  (initial_marbles : ℕ) 
  (h : num_friends > 0) :
  let additional := additional_marbles_needed num_friends initial_marbles
  let total := initial_marbles + additional
  (∀ i : ℕ, i > 0 ∧ i ≤ num_friends → i + 1 ≤ total / num_friends) ∧ 
  (∀ i j : ℕ, i > 0 ∧ j > 0 ∧ i ≤ num_friends ∧ j ≤ num_friends ∧ i ≠ j → i + 1 ≠ j + 1) := by
  sorry

end NUMINAMATH_CALUDE_lisa_marbles_theorem_distribution_satisfies_conditions_l3866_386675


namespace NUMINAMATH_CALUDE_last_digit_of_sum_l3866_386646

/-- Given a = 25 and b = -3, the last digit of a^1999 + b^2002 is 4 -/
theorem last_digit_of_sum (a b : ℤ) : a = 25 ∧ b = -3 → (a^1999 + b^2002) % 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_sum_l3866_386646


namespace NUMINAMATH_CALUDE_coffee_shrink_problem_l3866_386603

def shrink_ray_effect : ℝ := 0.5

theorem coffee_shrink_problem (num_cups : ℕ) (remaining_coffee : ℝ) 
  (h1 : num_cups = 5)
  (h2 : remaining_coffee = 20) : 
  (remaining_coffee / shrink_ray_effect) / num_cups = 8 := by
  sorry

end NUMINAMATH_CALUDE_coffee_shrink_problem_l3866_386603


namespace NUMINAMATH_CALUDE_min_doors_for_safety_l3866_386673

/-- Represents a spaceship with a given number of corridors -/
structure Spaceship :=
  (corridors : ℕ)

/-- Represents the state of doors in the spaceship -/
def DoorState := Fin 23 → Bool

/-- Checks if there exists a path from reactor to lounge -/
def hasPath (s : Spaceship) (state : DoorState) : Prop :=
  sorry -- Definition of path existence

/-- Counts the number of closed doors -/
def closedDoors (state : DoorState) : ℕ :=
  sorry -- Count of closed doors

/-- Theorem stating the minimum number of doors to close for safety -/
theorem min_doors_for_safety (s : Spaceship) :
  (s.corridors = 23) →
  (∀ (state : DoorState), closedDoors state ≥ 22 → ¬hasPath s state) ∧
  (∃ (state : DoorState), closedDoors state = 21 ∧ hasPath s state) :=
sorry

#check min_doors_for_safety

end NUMINAMATH_CALUDE_min_doors_for_safety_l3866_386673


namespace NUMINAMATH_CALUDE_expression_evaluation_l3866_386654

theorem expression_evaluation :
  let w : ℤ := 3
  let x : ℤ := -2
  let y : ℤ := 1
  let z : ℤ := 4
  w^2 * x^2 * y * z - w * x^2 * y * z^2 + w * y^3 * z^2 - w * y^2 * x * z^4 = 1536 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3866_386654


namespace NUMINAMATH_CALUDE_line_properties_l3866_386658

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Given the equation of a line y + 7 = -x - 3, prove it passes through (-3, -7) with slope -1 -/
theorem line_properties :
  let l : Line := { slope := -1, yIntercept := -10 }
  let p : Point := { x := -3, y := -7 }
  (p.y + 7 = -p.x - 3) ∧ 
  (l.slope = -1) ∧
  (p.y = l.slope * p.x + l.yIntercept) := by
  sorry


end NUMINAMATH_CALUDE_line_properties_l3866_386658


namespace NUMINAMATH_CALUDE_function_identity_l3866_386666

-- Define the property that f must satisfy
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + f y) + f (x * y) = y * f x + f y + f (f x)

-- State the theorem
theorem function_identity {f : ℝ → ℝ} (h : SatisfiesEquation f) :
  ∀ x : ℝ, f x = x :=
by sorry

end NUMINAMATH_CALUDE_function_identity_l3866_386666


namespace NUMINAMATH_CALUDE_proposition_q_undetermined_l3866_386629

theorem proposition_q_undetermined (p q : Prop) 
  (h1 : ¬(p ∧ q)) 
  (h2 : ¬p) : 
  (q ∨ ¬q) := by sorry

end NUMINAMATH_CALUDE_proposition_q_undetermined_l3866_386629


namespace NUMINAMATH_CALUDE_complex_quadrant_l3866_386698

theorem complex_quadrant (z : ℂ) (h : (1 + 2*I)*z = 3 - 4*I) : 
  (z.re < 0) ∧ (z.im < 0) :=
sorry

end NUMINAMATH_CALUDE_complex_quadrant_l3866_386698


namespace NUMINAMATH_CALUDE_hot_pepper_percentage_is_twenty_percent_l3866_386686

/-- Represents the total number of peppers picked by Joel over 7 days -/
def total_peppers : ℕ := 80

/-- Represents the number of non-hot peppers picked by Joel -/
def non_hot_peppers : ℕ := 64

/-- Calculates the percentage of hot peppers in Joel's garden -/
def hot_pepper_percentage : ℚ :=
  (total_peppers - non_hot_peppers : ℚ) / total_peppers * 100

/-- Proves that the percentage of hot peppers in Joel's garden is 20% -/
theorem hot_pepper_percentage_is_twenty_percent :
  hot_pepper_percentage = 20 := by
  sorry

end NUMINAMATH_CALUDE_hot_pepper_percentage_is_twenty_percent_l3866_386686


namespace NUMINAMATH_CALUDE_sqrt_neg_five_squared_l3866_386659

theorem sqrt_neg_five_squared : Real.sqrt ((-5)^2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_neg_five_squared_l3866_386659


namespace NUMINAMATH_CALUDE_peter_calories_l3866_386689

/-- Represents the number of calories Peter wants to eat -/
def calories_wanted (chip_calories : ℕ) (chips_per_bag : ℕ) (bag_cost : ℕ) (total_spent : ℕ) : ℕ :=
  (total_spent / bag_cost) * chips_per_bag * chip_calories

/-- Proves that Peter wants to eat 480 calories worth of chips -/
theorem peter_calories : calories_wanted 10 24 2 4 = 480 := by
  sorry

end NUMINAMATH_CALUDE_peter_calories_l3866_386689


namespace NUMINAMATH_CALUDE_smallest_perfect_cube_multiplier_l3866_386652

def y : ℕ := 2^3 * 3^4 * 4^3 * 5^4 * 6^6 * 7^7 * 8^8 * 9^9

theorem smallest_perfect_cube_multiplier (n : ℕ) :
  (∀ m : ℕ, m < 29400 → ¬ ∃ k : ℕ, m * y = k^3) ∧
  ∃ k : ℕ, 29400 * y = k^3 :=
sorry

end NUMINAMATH_CALUDE_smallest_perfect_cube_multiplier_l3866_386652


namespace NUMINAMATH_CALUDE_loss_percentage_calculation_l3866_386623

theorem loss_percentage_calculation (cost_price selling_price : ℝ) : 
  cost_price = 1500 →
  selling_price = 1290 →
  (cost_price - selling_price) / cost_price * 100 = 14 := by
sorry

end NUMINAMATH_CALUDE_loss_percentage_calculation_l3866_386623


namespace NUMINAMATH_CALUDE_systematic_sampling_method_l3866_386688

theorem systematic_sampling_method (population_size : ℕ) (sample_size : ℕ) 
  (h1 : population_size = 102) (h2 : sample_size = 9) : 
  ∃ (excluded : ℕ) (interval : ℕ), 
    excluded = 3 ∧ 
    interval = 11 ∧ 
    (population_size - excluded) % sample_size = 0 ∧
    (population_size - excluded) / sample_size = interval :=
by sorry

end NUMINAMATH_CALUDE_systematic_sampling_method_l3866_386688


namespace NUMINAMATH_CALUDE_algebraic_expression_equality_l3866_386687

theorem algebraic_expression_equality : 
  Real.sqrt (5 - 2 * Real.sqrt 6) + Real.sqrt (7 - 4 * Real.sqrt 3) = 2 - Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_algebraic_expression_equality_l3866_386687


namespace NUMINAMATH_CALUDE_house_legs_l3866_386626

/-- The number of legs in a house with humans and various pets -/
def total_legs (humans dogs cats parrots goldfish : ℕ) : ℕ :=
  humans * 2 + dogs * 4 + cats * 4 + parrots * 2 + goldfish * 0

/-- Theorem: The total number of legs in the house is 38 -/
theorem house_legs : total_legs 5 2 3 4 5 = 38 := by
  sorry

end NUMINAMATH_CALUDE_house_legs_l3866_386626


namespace NUMINAMATH_CALUDE_f_xy_second_derivative_not_exists_l3866_386620

noncomputable def f (x y : ℝ) : ℝ :=
  if x^2 + y^4 ≠ 0 then (x * y^2) / (x^2 + y^4) else 0

theorem f_xy_second_derivative_not_exists :
  ¬ ∃ (L : ℝ), ∀ ε > 0, ∃ δ > 0, ∀ x y : ℝ,
    x^2 + y^2 < δ^2 → |((f (x + y) y - f x y) / y - (f x y - f x 0) / y) / x - L| < ε :=
sorry

end NUMINAMATH_CALUDE_f_xy_second_derivative_not_exists_l3866_386620


namespace NUMINAMATH_CALUDE_perpendicular_bisector_equation_l3866_386607

/-- Given two points A (-2, 0) and B (0, 4), prove that the equation x + 2y - 3 = 0
    represents the perpendicular bisector of the line segment AB. -/
theorem perpendicular_bisector_equation (A B : ℝ × ℝ) (h1 : A = (-2, 0)) (h2 : B = (0, 4)) :
  ∀ (x y : ℝ), (x + 2*y - 3 = 0) ↔ 
    (x - (-2))^2 + (y - 0)^2 = (x - 0)^2 + (y - 4)^2 ∧ 
    (x + 1) * (4 - 0) + (y - 2) * (0 - (-2)) = 0 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_equation_l3866_386607


namespace NUMINAMATH_CALUDE_simple_interest_rate_example_l3866_386634

/-- Calculates the simple interest rate given principal, amount, and time -/
def simple_interest_rate (principal amount : ℚ) (time : ℕ) : ℚ :=
  ((amount - principal) * 100) / (principal * time)

theorem simple_interest_rate_example :
  simple_interest_rate 750 900 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_rate_example_l3866_386634


namespace NUMINAMATH_CALUDE_line_equation_correctness_l3866_386632

/-- A line passing through a point with a given direction vector -/
structure DirectedLine (n : ℕ) where
  point : Fin n → ℝ
  direction : Fin n → ℝ

/-- The equation of a line in 2D space -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point satisfies a line equation -/
def satisfiesEquation (p : Fin 2 → ℝ) (eq : LineEquation) : Prop :=
  eq.a * p 0 + eq.b * p 1 + eq.c = 0

/-- Check if a vector is parallel to a line equation -/
def isParallel (v : Fin 2 → ℝ) (eq : LineEquation) : Prop :=
  eq.a * v 0 + eq.b * v 1 = 0

theorem line_equation_correctness (l : DirectedLine 2) (eq : LineEquation) :
  (l.point 0 = -3 ∧ l.point 1 = 1) →
  (l.direction 0 = 2 ∧ l.direction 1 = -3) →
  (eq.a = 3 ∧ eq.b = 2 ∧ eq.c = -11) →
  satisfiesEquation l.point eq ∧ isParallel l.direction eq :=
sorry

end NUMINAMATH_CALUDE_line_equation_correctness_l3866_386632


namespace NUMINAMATH_CALUDE_min_hypotenuse_right_triangle_l3866_386604

theorem min_hypotenuse_right_triangle (k : ℝ) (h : k > 0) :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
  a + b + c = k ∧
  a^2 + b^2 = c^2 ∧
  c = (Real.sqrt 2 - 1) * k ∧
  ∀ (a' b' c' : ℝ), a' > 0 → b' > 0 → c' > 0 →
    a' + b' + c' = k → a'^2 + b'^2 = c'^2 → c' ≥ (Real.sqrt 2 - 1) * k := by
  sorry

end NUMINAMATH_CALUDE_min_hypotenuse_right_triangle_l3866_386604


namespace NUMINAMATH_CALUDE_focus_of_standard_parabola_l3866_386676

/-- The focus of the parabola y = x^2 is at the point (0, 1/4). -/
theorem focus_of_standard_parabola :
  let f : ℝ × ℝ := (0, 1/4)
  let parabola := {(x, y) : ℝ × ℝ | y = x^2}
  f ∈ parabola ∧ ∀ p ∈ parabola, dist p f = dist p (0, -1/4) :=
by sorry

end NUMINAMATH_CALUDE_focus_of_standard_parabola_l3866_386676


namespace NUMINAMATH_CALUDE_harkamal_payment_l3866_386605

/-- The total amount Harkamal paid for grapes and mangoes -/
def total_amount (grape_quantity : ℕ) (grape_rate : ℕ) (mango_quantity : ℕ) (mango_rate : ℕ) : ℕ :=
  grape_quantity * grape_rate + mango_quantity * mango_rate

/-- Theorem stating that Harkamal paid 965 for his purchase -/
theorem harkamal_payment : total_amount 8 70 9 45 = 965 := by
  sorry

end NUMINAMATH_CALUDE_harkamal_payment_l3866_386605


namespace NUMINAMATH_CALUDE_circle_tangent_condition_l3866_386613

-- Define a circle using its equation coefficients
structure Circle where
  D : ℝ
  E : ℝ
  F : ℝ

-- Define what it means for a circle to be tangent to the x-axis at the origin
def tangent_to_x_axis_at_origin (c : Circle) : Prop :=
  ∃ (y : ℝ), y ≠ 0 ∧ 0^2 + y^2 + c.D*0 + c.E*y + c.F = 0 ∧
  ∀ (x : ℝ), x ≠ 0 → (∀ (y : ℝ), x^2 + y^2 + c.D*x + c.E*y + c.F ≠ 0)

-- The main theorem
theorem circle_tangent_condition (c : Circle) :
  tangent_to_x_axis_at_origin c ↔ c.E ≠ 0 ∧ c.D = 0 ∧ c.F = 0 := by
  sorry

end NUMINAMATH_CALUDE_circle_tangent_condition_l3866_386613


namespace NUMINAMATH_CALUDE_range_of_a_l3866_386693

theorem range_of_a (p q : Prop) 
  (hp : ∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - a ≥ 0)
  (hq : ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0) :
  a ≤ -2 ∨ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3866_386693


namespace NUMINAMATH_CALUDE_class_representatives_count_l3866_386680

/-- Represents the number of boys in the class -/
def num_boys : ℕ := 5

/-- Represents the number of girls in the class -/
def num_girls : ℕ := 3

/-- Represents the number of subjects needing representatives -/
def num_subjects : ℕ := 5

/-- Calculates the number of ways to select representatives with fewer girls than boys -/
def count_fewer_girls : ℕ := sorry

/-- Calculates the number of ways to select representatives with Boy A as a representative but not for mathematics -/
def count_boy_a_not_math : ℕ := sorry

/-- Calculates the number of ways to select representatives with Girl B for Chinese and Boy A as a representative but not for mathematics -/
def count_girl_b_chinese_boy_a_not_math : ℕ := sorry

/-- Theorem stating the correct number of ways for each condition -/
theorem class_representatives_count :
  count_fewer_girls = 5520 ∧
  count_boy_a_not_math = 3360 ∧
  count_girl_b_chinese_boy_a_not_math = 360 := by sorry

end NUMINAMATH_CALUDE_class_representatives_count_l3866_386680


namespace NUMINAMATH_CALUDE_triangle_inequalities_l3866_386608

theorem triangle_inequalities (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  let p := (a + b + c) / 2
  (2 * (a + b + c) * (a^2 + b^2 + c^2) ≥ 3 * (a^3 + b^3 + c^3 + 3 * a * b * c)) ∧
  ((a + b + c)^3 ≤ 5 * (b * c * (b + c) + c * a * (c + a) + a * b * (a + b)) - 3 * a * b * c) ∧
  (a * b * c < a^2 * (p - a) + b^2 * (p - b) + c^2 * (p - c) ∧
    a^2 * (p - a) + b^2 * (p - b) + c^2 * (p - c) ≤ 3/2 * a * b * c) ∧
  (1 < Real.cos (π - Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c))) +
       Real.cos (π - Real.arccos ((a^2 + c^2 - b^2) / (2 * a * c))) +
       Real.cos (π - Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))) ∧
   Real.cos (π - Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c))) +
       Real.cos (π - Real.arccos ((a^2 + c^2 - b^2) / (2 * a * c))) +
       Real.cos (π - Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))) ≤ 3/2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequalities_l3866_386608


namespace NUMINAMATH_CALUDE_binomial_coefficient_sum_l3866_386612

theorem binomial_coefficient_sum (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x, (1 - 2*x)^7 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| + |a₇| = 3^7 - 1 := by
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_sum_l3866_386612


namespace NUMINAMATH_CALUDE_geometric_sequence_q_value_l3866_386669

/-- A geometric sequence {a_n} -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_q_value
  (a : ℕ → ℝ)
  (h_monotone : ∀ n : ℕ, a n ≤ a (n + 1))
  (h_geometric : geometric_sequence a)
  (h_sum : a 3 + a 7 = 5)
  (h_product : a 6 * a 4 = 6) :
  ∃ q : ℝ, q > 1 ∧ q^4 = 3/2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_q_value_l3866_386669


namespace NUMINAMATH_CALUDE_bed_price_ratio_bed_to_frame_ratio_l3866_386690

/-- Given a bed frame price, a bed price multiple, a discount rate, and a final price,
    calculate the ratio of the bed's price to the bed frame's price. -/
theorem bed_price_ratio
  (bed_frame_price : ℝ)
  (bed_price_multiple : ℝ)
  (discount_rate : ℝ)
  (final_price : ℝ)
  (h1 : bed_frame_price = 75)
  (h2 : discount_rate = 0.2)
  (h3 : final_price = 660)
  (h4 : (1 - discount_rate) * (bed_frame_price + bed_frame_price * bed_price_multiple) = final_price) :
  bed_price_multiple = 10 := by
sorry

/-- The ratio of the bed's price to the bed frame's price is 10:1. -/
theorem bed_to_frame_ratio (bed_price_multiple : ℝ) 
  (h : bed_price_multiple = 10) : 
  bed_price_multiple / 1 = 10 / 1 := by
sorry

end NUMINAMATH_CALUDE_bed_price_ratio_bed_to_frame_ratio_l3866_386690


namespace NUMINAMATH_CALUDE_order_of_numbers_l3866_386643

def base_to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * base ^ i) 0

def a : Nat := base_to_decimal [1,1,1,1,1,1] 2
def b : Nat := base_to_decimal [0,1,2] 6
def c : Nat := base_to_decimal [0,0,0,1] 4
def d : Nat := base_to_decimal [0,1,1] 8

theorem order_of_numbers : b > d ∧ d > a ∧ a > c := by
  sorry

end NUMINAMATH_CALUDE_order_of_numbers_l3866_386643


namespace NUMINAMATH_CALUDE_aquarium_original_price_l3866_386668

/-- Proves that the original price of an aquarium is $120 given the conditions of the problem -/
theorem aquarium_original_price (P : ℝ) : 
  (0.5 * P + 0.05 * (0.5 * P) = 63) → P = 120 := by
  sorry

end NUMINAMATH_CALUDE_aquarium_original_price_l3866_386668


namespace NUMINAMATH_CALUDE_triangle_area_inequality_l3866_386697

-- Define a triangle
structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define the area of a triangle
def area (t : Triangle) : ℝ := sorry

-- Define an interior point of a triangle
def is_interior_point (P : Point) (t : Triangle) : Prop := sorry

-- Define parallel lines
def parallel_line (P : Point) (l : Line) : Line := sorry

-- Define the division of a triangle by parallel lines
def divide_triangle (t : Triangle) (P : Point) : Prop := sorry

-- Define the areas of the smaller triangles
def small_triangle_areas (t : Triangle) (P : Point) : ℝ × ℝ × ℝ := sorry

-- Theorem statement
theorem triangle_area_inequality (ABC : Triangle) (P : Point) :
  is_interior_point P ABC →
  divide_triangle ABC P →
  let (S1, S2, S3) := small_triangle_areas ABC P
  area ABC ≤ 3 * (S1 + S2 + S3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_inequality_l3866_386697


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l3866_386637

def U : Set Nat := {1,2,3,4,5,6,7,8}
def A : Set Nat := {2,5,8}
def B : Set Nat := {1,3,5,7}

theorem complement_intersection_theorem : 
  (U \ A) ∩ B = {1,3,7} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l3866_386637


namespace NUMINAMATH_CALUDE_bird_count_problem_l3866_386655

/-- Represents the number of birds in a group -/
structure BirdGroup where
  adults : ℕ
  offspring_per_adult : ℕ

/-- Calculates the total number of birds in a group -/
def total_birds (group : BirdGroup) : ℕ :=
  group.adults * (group.offspring_per_adult + 1)

/-- The problem statement -/
theorem bird_count_problem (duck_group1 duck_group2 duck_group3 geese_group swan_group : BirdGroup)
  (h1 : duck_group1 = { adults := 2, offspring_per_adult := 5 })
  (h2 : duck_group2 = { adults := 6, offspring_per_adult := 3 })
  (h3 : duck_group3 = { adults := 9, offspring_per_adult := 6 })
  (h4 : geese_group = { adults := 4, offspring_per_adult := 7 })
  (h5 : swan_group = { adults := 3, offspring_per_adult := 4 }) :
  (total_birds duck_group1 + total_birds duck_group2 + total_birds duck_group3 +
   total_birds geese_group + total_birds swan_group) * 3 = 438 := by
  sorry

end NUMINAMATH_CALUDE_bird_count_problem_l3866_386655


namespace NUMINAMATH_CALUDE_distance_to_plane_l3866_386660

/-- The distance from a point to a plane in 3D space -/
def distance_point_to_plane (P : ℝ × ℝ × ℝ) (n : ℝ × ℝ × ℝ) : ℝ :=
  sorry

theorem distance_to_plane :
  let P : ℝ × ℝ × ℝ := (-1, 3, 2)
  let n : ℝ × ℝ × ℝ := (2, -2, 1)
  distance_point_to_plane P n = 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_plane_l3866_386660


namespace NUMINAMATH_CALUDE_meeting_at_163rd_streetlight_l3866_386691

/-- The number of streetlights along the alley -/
def num_streetlights : ℕ := 400

/-- The position where Alla and Boris meet -/
def meeting_point : ℕ := 163

/-- Alla's position when observation is made -/
def alla_observed_pos : ℕ := 55

/-- Boris's position when observation is made -/
def boris_observed_pos : ℕ := 321

/-- The theorem stating that Alla and Boris meet at the 163rd streetlight -/
theorem meeting_at_163rd_streetlight :
  let alla_distance := alla_observed_pos - 1
  let boris_distance := num_streetlights - boris_observed_pos
  let total_observed_distance := alla_distance + boris_distance
  let scaling_factor := (num_streetlights - 1) / total_observed_distance
  (1 : ℚ) + scaling_factor * alla_distance = meeting_point := by
  sorry

end NUMINAMATH_CALUDE_meeting_at_163rd_streetlight_l3866_386691


namespace NUMINAMATH_CALUDE_min_distance_theorem_l3866_386624

theorem min_distance_theorem (a b x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  x + y + Real.sqrt ((a - x)^2 + (b - y)^2) ≥ Real.sqrt (a^2 + b^2) := by
  sorry

end NUMINAMATH_CALUDE_min_distance_theorem_l3866_386624


namespace NUMINAMATH_CALUDE_triangle_equality_condition_l3866_386685

/-- In a triangle ABC, the sum of squares of its sides is equal to 4√3 times its area 
    if and only if the triangle is equilateral. -/
theorem triangle_equality_condition (a b c : ℝ) (Δ : ℝ) :
  (a > 0) → (b > 0) → (c > 0) → (Δ > 0) →
  (a^2 + b^2 + c^2 = 4 * Real.sqrt 3 * Δ) ↔ (a = b ∧ b = c) :=
sorry

end NUMINAMATH_CALUDE_triangle_equality_condition_l3866_386685


namespace NUMINAMATH_CALUDE_magazine_subscription_l3866_386614

theorem magazine_subscription (total_students : ℕ) 
  (boys_first_half : ℕ) (girls_first_half : ℕ)
  (boys_second_half : ℕ) (girls_second_half : ℕ)
  (boys_whole_year : ℕ) 
  (h1 : total_students = 56)
  (h2 : boys_first_half = 25)
  (h3 : girls_first_half = 15)
  (h4 : boys_second_half = 26)
  (h5 : girls_second_half = 25)
  (h6 : boys_whole_year = 23) :
  girls_first_half - (girls_first_half + girls_second_half - (total_students - (boys_first_half + boys_second_half - boys_whole_year))) = 3 := by
  sorry

end NUMINAMATH_CALUDE_magazine_subscription_l3866_386614


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3866_386682

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 1 + 3 * a 6 + a 11 = 10) →
  a 5 + a 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3866_386682


namespace NUMINAMATH_CALUDE_min_intersection_size_l3866_386625

theorem min_intersection_size (total students_green_eyes students_own_lunch : ℕ)
  (h_total : total = 25)
  (h_green : students_green_eyes = 15)
  (h_lunch : students_own_lunch = 18)
  : ∃ (intersection : ℕ), 
    intersection ≤ students_green_eyes ∧ 
    intersection ≤ students_own_lunch ∧
    intersection ≥ students_green_eyes + students_own_lunch - total ∧
    intersection = 8 := by
  sorry

end NUMINAMATH_CALUDE_min_intersection_size_l3866_386625


namespace NUMINAMATH_CALUDE_jeff_pickup_cost_l3866_386651

/-- The cost of last year's costume in dollars -/
def last_year_cost : ℝ := 250

/-- The percentage increase in cost compared to last year -/
def cost_increase_percent : ℝ := 0.4

/-- The deposit percentage -/
def deposit_percent : ℝ := 0.1

/-- The total cost of this year's costume -/
def total_cost : ℝ := last_year_cost * (1 + cost_increase_percent)

/-- The amount of the deposit -/
def deposit : ℝ := total_cost * deposit_percent

/-- The amount Jeff paid when picking up the costume -/
def pickup_cost : ℝ := total_cost - deposit

theorem jeff_pickup_cost : pickup_cost = 315 := by
  sorry

end NUMINAMATH_CALUDE_jeff_pickup_cost_l3866_386651


namespace NUMINAMATH_CALUDE_find_e_l3866_386684

-- Define the functions f and g
def f (c : ℝ) (x : ℝ) : ℝ := 5 * x + 2 * c
def g (c : ℝ) (x : ℝ) : ℝ := c * x^2 + 3

-- State the theorem
theorem find_e (c : ℝ) :
  (∀ x, f c (g c x) = 15 * x^2 + 21) :=
by sorry

end NUMINAMATH_CALUDE_find_e_l3866_386684


namespace NUMINAMATH_CALUDE_triangle_theorem_l3866_386639

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_theorem (t : Triangle) 
  (h1 : t.b * Real.sin t.A = Real.sqrt 3 * t.a * Real.cos t.B)
  (h2 : t.b = 3)
  (h3 : Real.sin t.C = 2 * Real.sin t.A) :
  t.B = π / 3 ∧ t.a = Real.sqrt 3 ∧ t.c = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l3866_386639
