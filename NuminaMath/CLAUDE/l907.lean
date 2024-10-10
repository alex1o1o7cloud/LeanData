import Mathlib

namespace sqrt_gt_sufficient_not_necessary_for_exp_gt_l907_90788

theorem sqrt_gt_sufficient_not_necessary_for_exp_gt (a b : ℝ) :
  (∀ a b : ℝ, Real.sqrt a > Real.sqrt b → Real.exp a > Real.exp b) ∧
  ¬(∀ a b : ℝ, Real.exp a > Real.exp b → Real.sqrt a > Real.sqrt b) :=
by sorry

end sqrt_gt_sufficient_not_necessary_for_exp_gt_l907_90788


namespace tan_is_odd_l907_90743

-- Define the tangent function
noncomputable def tan (x : ℝ) : ℝ := Real.tan x

-- State the theorem
theorem tan_is_odd : ∀ x : ℝ, tan (-x) = -tan x := by sorry

end tan_is_odd_l907_90743


namespace max_value_f2019_l907_90753

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 2*x

-- Define the recursive function f_n
def f_n : ℕ → (ℝ → ℝ)
  | 0 => f
  | n + 1 => λ x => f (f_n n x)

-- State the theorem
theorem max_value_f2019 :
  ∀ x ∈ Set.Icc 1 2,
  f_n 2019 x ≤ 3^(2^2019) - 1 ∧
  ∃ y ∈ Set.Icc 1 2, f_n 2019 y = 3^(2^2019) - 1 :=
sorry

end max_value_f2019_l907_90753


namespace total_area_is_62_l907_90708

/-- The area of a figure composed of three rectangles -/
def figure_area (area1 area2 area3 : ℕ) : ℕ := area1 + area2 + area3

/-- Theorem: The total area of the figure is 62 square units -/
theorem total_area_is_62 (area1 area2 area3 : ℕ) 
  (h1 : area1 = 30) 
  (h2 : area2 = 12) 
  (h3 : area3 = 20) : 
  figure_area area1 area2 area3 = 62 := by
  sorry

#eval figure_area 30 12 20

end total_area_is_62_l907_90708


namespace sum_of_abc_l907_90739

theorem sum_of_abc (a b c : ℕ+) 
  (h1 : a.val * b.val + c.val = 47)
  (h2 : b.val * c.val + a.val = 47)
  (h3 : a.val * c.val + b.val = 47) :
  a.val + b.val + c.val = 48 := by
  sorry

end sum_of_abc_l907_90739


namespace square_difference_ratio_l907_90706

theorem square_difference_ratio : (2045^2 - 2030^2) / (2050^2 - 2025^2) = 3/5 := by
  sorry

end square_difference_ratio_l907_90706


namespace line_passes_through_fixed_point_l907_90763

theorem line_passes_through_fixed_point :
  ∀ (k : ℝ), (k * (-3) - (-2) + 3 * k - 2 = 0) := by
  sorry

end line_passes_through_fixed_point_l907_90763


namespace certain_number_proof_l907_90796

theorem certain_number_proof (x : ℝ) : 0.15 * x + 0.12 * 45 = 9.15 ↔ x = 25 := by
  sorry

end certain_number_proof_l907_90796


namespace quadratic_two_unequal_real_roots_l907_90775

theorem quadratic_two_unequal_real_roots :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (2 * x₁^2 - 3 * x₁ - 4 = 0) ∧ (2 * x₂^2 - 3 * x₂ - 4 = 0) := by
  sorry

end quadratic_two_unequal_real_roots_l907_90775


namespace greatest_three_digit_number_l907_90783

theorem greatest_three_digit_number : ∃ n : ℕ,
  n = 793 ∧
  100 ≤ n ∧ n < 1000 ∧
  ∃ k₁ : ℕ, n = 9 * k₁ + 1 ∧
  ∃ k₂ : ℕ, n = 5 * k₂ + 3 ∧
  ∃ k₃ : ℕ, n = 7 * k₃ + 2 ∧
  ∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧
    ∃ l₁ : ℕ, m = 9 * l₁ + 1 ∧
    ∃ l₂ : ℕ, m = 5 * l₂ + 3 ∧
    ∃ l₃ : ℕ, m = 7 * l₃ + 2) → m ≤ n :=
by sorry

end greatest_three_digit_number_l907_90783


namespace paige_folders_l907_90721

theorem paige_folders (initial_files : ℕ) (deleted_files : ℕ) (files_per_folder : ℕ) : 
  initial_files = 27 →
  deleted_files = 9 →
  files_per_folder = 6 →
  (initial_files - deleted_files) / files_per_folder = 3 :=
by
  sorry

end paige_folders_l907_90721


namespace complement_union_equals_set_l907_90747

def U : Set Nat := {1,2,3,4,5,6}
def A : Set Nat := {2,4,5}
def B : Set Nat := {1,2,5}

theorem complement_union_equals_set : 
  (U \ (A ∪ B)) = {3,6} := by sorry

end complement_union_equals_set_l907_90747


namespace coin_ratio_l907_90792

/-- Given the total number of coins, the fraction Amalie spends, and the number of coins
    Amalie has left, prove the ratio of Elsa's coins to Amalie's original coins. -/
theorem coin_ratio (total : ℕ) (amalie_spent_fraction : ℚ) (amalie_left : ℕ)
  (h1 : total = 440)
  (h2 : amalie_spent_fraction = 3/4)
  (h3 : amalie_left = 90) :
  (total - (amalie_left / (1 - amalie_spent_fraction))) / (amalie_left / (1 - amalie_spent_fraction)) = 8/9 := by
  sorry

end coin_ratio_l907_90792


namespace one_thirds_in_nine_thirds_l907_90719

theorem one_thirds_in_nine_thirds : (9 : ℚ) / 3 / (1 / 3) = 9 := by sorry

end one_thirds_in_nine_thirds_l907_90719


namespace root_expression_value_l907_90740

theorem root_expression_value (r s : ℝ) : 
  (3 * r^2 + 4 * r - 18 = 0) →
  (3 * s^2 + 4 * s - 18 = 0) →
  r ≠ s →
  (3 * r^3 - 3 * s^3) / (r - s) = 70/3 := by
sorry

end root_expression_value_l907_90740


namespace p_iff_q_l907_90700

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := x - y - 1 = 0
def l₂ (a x y : ℝ) : Prop := x + a*y - 2 = 0

-- Define parallel lines
def parallel (f g : ℝ → ℝ → Prop) : Prop :=
  ∀ x₁ y₁ x₂ y₂ : ℝ, f x₁ y₁ ∧ f x₂ y₂ ∧ g x₁ y₁ ∧ g x₂ y₂ →
    (y₂ - y₁) / (x₂ - x₁) = (y₂ - y₁) / (x₂ - x₁)

-- Define the propositions p and q
def p (a : ℝ) : Prop := parallel (l₁) (l₂ a)
def q (a : ℝ) : Prop := a = -1

-- State the theorem
theorem p_iff_q : ∀ a : ℝ, p a ↔ q a := by sorry

end p_iff_q_l907_90700


namespace isosceles_triangle_base_length_l907_90703

theorem isosceles_triangle_base_length 
  (equilateral_perimeter : ℝ) 
  (isosceles_perimeter : ℝ) 
  (h_equilateral : equilateral_perimeter = 60) 
  (h_isosceles : isosceles_perimeter = 50) 
  (h_shared_side : equilateral_perimeter / 3 = (isosceles_perimeter - isosceles_base) / 2) : 
  isosceles_base = 10 :=
sorry

end isosceles_triangle_base_length_l907_90703


namespace first_grade_allocation_l907_90749

theorem first_grade_allocation (total : ℕ) (ratio_first : ℕ) (ratio_second : ℕ) (ratio_third : ℕ) 
  (h_total : total = 160)
  (h_ratio : ratio_first = 6 ∧ ratio_second = 5 ∧ ratio_third = 5) :
  (total * ratio_first) / (ratio_first + ratio_second + ratio_third) = 60 := by
  sorry

end first_grade_allocation_l907_90749


namespace total_students_l907_90798

theorem total_students (boys girls : ℕ) (h1 : boys * 5 = girls * 8) (h2 : girls = 160) : 
  boys + girls = 416 := by
sorry

end total_students_l907_90798


namespace proposition_implication_l907_90729

theorem proposition_implication (m : ℝ) : 
  (∀ x, -2 ≤ x ∧ x ≤ 10 → 1 - m ≤ x ∧ x ≤ 1 + m) ∧ 
  (∃ x, 1 - m ≤ x ∧ x ≤ 1 + m ∧ (x < -2 ∨ x > 10)) ∧
  (m > 0) →
  m ≥ 9 := by sorry

end proposition_implication_l907_90729


namespace berry_ratio_l907_90712

/-- Given the distribution of berries among Stacy, Steve, and Sylar, 
    prove that the ratio of Stacy's berries to Steve's berries is 4:1 -/
theorem berry_ratio (total berries_stacy berries_steve berries_sylar : ℕ) :
  total = 1100 →
  berries_stacy = 800 →
  berries_steve = 2 * berries_sylar →
  total = berries_stacy + berries_steve + berries_sylar →
  berries_stacy / berries_steve = 4 := by
  sorry

#check berry_ratio

end berry_ratio_l907_90712


namespace manufacturing_expenses_calculation_l907_90755

/-- Calculates the monthly manufacturing expenses for a textile firm. -/
def monthly_manufacturing_expenses (
  total_looms : ℕ)
  (aggregate_sales : ℕ)
  (establishment_charges : ℕ)
  (profit_decrease : ℕ) : ℕ :=
  let sales_per_loom := aggregate_sales / total_looms
  let expenses_per_loom := sales_per_loom - profit_decrease
  expenses_per_loom * total_looms

/-- Proves that the monthly manufacturing expenses are 150000 given the specified conditions. -/
theorem manufacturing_expenses_calculation :
  monthly_manufacturing_expenses 80 500000 75000 4375 = 150000 := by
  sorry

end manufacturing_expenses_calculation_l907_90755


namespace ruble_payment_l907_90748

theorem ruble_payment (n : ℤ) (h : n > 7) : ∃ x y : ℕ, 3 * x + 5 * y = n := by
  sorry

end ruble_payment_l907_90748


namespace solution_set_of_even_monotonic_function_l907_90717

def f (a b x : ℝ) := (x - 2) * (a * x + b)

theorem solution_set_of_even_monotonic_function 
  (a b : ℝ) 
  (h_even : ∀ x, f a b x = f a b (-x))
  (h_increasing : ∀ x y, 0 < x → x < y → f a b x < f a b y) :
  {x : ℝ | f a b (2 - x) > 0} = {x : ℝ | x < 0 ∨ x > 4} := by
sorry


end solution_set_of_even_monotonic_function_l907_90717


namespace reading_time_l907_90768

theorem reading_time (total_pages : ℕ) (first_half_speed second_half_speed : ℕ) : 
  total_pages = 500 → 
  first_half_speed = 10 → 
  second_half_speed = 5 → 
  (total_pages / 2 / first_half_speed + total_pages / 2 / second_half_speed) = 75 := by
sorry

end reading_time_l907_90768


namespace polynomial_roots_magnitude_l907_90742

theorem polynomial_roots_magnitude (c : ℂ) : 
  (∃ (Q : ℂ → ℂ), 
    Q = (fun x => (x^2 - 3*x + 3) * (x^2 - c*x + 9) * (x^2 - 5*x + 15)) ∧
    (∃ (r1 r2 r3 : ℂ), 
      r1 ≠ r2 ∧ r2 ≠ r3 ∧ r1 ≠ r3 ∧
      (∀ x : ℂ, Q x = 0 ↔ x = r1 ∨ x = r2 ∨ x = r3))) →
  Complex.abs c = 6 := by
sorry

end polynomial_roots_magnitude_l907_90742


namespace infinite_commuting_functions_l907_90762

/-- Given a bijective function f from R to R, there exists an infinite number of functions g 
    from R to R such that f(g(x)) = g(f(x)) for all x in R. -/
theorem infinite_commuting_functions 
  (f : ℝ → ℝ) 
  (hf : Function.Bijective f) : 
  ∃ (S : Set (ℝ → ℝ)), Set.Infinite S ∧ ∀ g ∈ S, ∀ x, f (g x) = g (f x) := by
  sorry

end infinite_commuting_functions_l907_90762


namespace prob_at_least_one_defective_l907_90701

/-- The probability of drawing a defective box from each large box -/
def p_defective : ℝ := 0.01

/-- The probability of drawing a non-defective box from each large box -/
def p_non_defective : ℝ := 1 - p_defective

/-- The number of boxes drawn -/
def n : ℕ := 3

theorem prob_at_least_one_defective :
  1 - p_non_defective ^ n = 1 - 0.99 ^ 3 :=
sorry

end prob_at_least_one_defective_l907_90701


namespace min_value_of_expression_l907_90702

/-- Given a moving straight line ax + by + c - 2 = 0 where a > 0, c > 0,
    that always passes through point (1, m), and the maximum distance
    from (4, 0) to the line is 3, the minimum value of 1/(2a) + 2/c is 9/4. -/
theorem min_value_of_expression (a b c m : ℝ) : 
  a > 0 → c > 0 → 
  (∀ x y, a * x + b * y + c - 2 = 0 → x = 1 → y = m) →
  (∃ x y, a * x + b * y + c - 2 = 0 ∧ 
    Real.sqrt ((x - 4)^2 + y^2) = 3) →
  (∀ x y, a * x + b * y + c - 2 = 0 → 
    Real.sqrt ((x - 4)^2 + y^2) ≤ 3) →
  (1 / (2 * a) + 2 / c) ≥ 9/4 :=
by sorry

end min_value_of_expression_l907_90702


namespace proportional_relationship_l907_90765

theorem proportional_relationship (k : ℝ) (x z : ℝ → ℝ) :
  (∀ t, x t = k / (z t * Real.sqrt (z t))) →
  x 9 = 8 →
  x 64 = 27 / 64 := by
sorry

end proportional_relationship_l907_90765


namespace yao_ming_shots_l907_90779

/-- Represents the scoring details of a basketball player in a game -/
structure ScoringDetails where
  total_shots_made : ℕ
  total_points : ℕ
  three_pointers_made : ℕ

/-- Calculates the number of 2-point shots and free throws made given the scoring details -/
def calculate_shots (details : ScoringDetails) : ℕ × ℕ :=
  let two_pointers := (details.total_points - 3 * details.three_pointers_made) / 2
  let free_throws := details.total_shots_made - details.three_pointers_made - two_pointers
  (two_pointers, free_throws)

/-- Theorem stating that given Yao Ming's scoring details, he made 8 2-point shots and 3 free throws -/
theorem yao_ming_shots :
  let details : ScoringDetails := {
    total_shots_made := 14,
    total_points := 28,
    three_pointers_made := 3
  }
  calculate_shots details = (8, 3) := by sorry

end yao_ming_shots_l907_90779


namespace rangers_apprentice_reading_l907_90716

theorem rangers_apprentice_reading (total_books : Nat) (pages_per_book : Nat) 
  (books_read_first_month : Nat) (pages_left_to_finish : Nat) :
  total_books = 14 →
  pages_per_book = 200 →
  books_read_first_month = 4 →
  pages_left_to_finish = 1000 →
  (((total_books - books_read_first_month) * pages_per_book - pages_left_to_finish) / pages_per_book) / 
  (total_books - books_read_first_month) = 1 / 2 := by
  sorry

end rangers_apprentice_reading_l907_90716


namespace unique_function_satisfying_equation_l907_90778

theorem unique_function_satisfying_equation :
  ∃! f : ℝ → ℝ, ∀ x y : ℝ, f (x - f y) = x - y ∧ f = id := by
  sorry

end unique_function_satisfying_equation_l907_90778


namespace soccer_team_selection_l907_90744

/-- The total number of players in the soccer team -/
def total_players : ℕ := 16

/-- The number of quadruplets in the team -/
def num_quadruplets : ℕ := 4

/-- The number of players to be chosen as starters -/
def num_starters : ℕ := 6

/-- The number of quadruplets to be chosen as starters -/
def num_quadruplets_chosen : ℕ := 1

/-- The number of ways to choose the starting lineup -/
def num_ways : ℕ := 3168

theorem soccer_team_selection :
  (num_quadruplets * Nat.choose (total_players - num_quadruplets) (num_starters - num_quadruplets_chosen)) = num_ways :=
sorry

end soccer_team_selection_l907_90744


namespace sum_remainder_modulo_11_l907_90772

theorem sum_remainder_modulo_11 : (123456 + 123457 + 123458 + 123459 + 123460 + 123461) % 11 = 5 := by
  sorry

end sum_remainder_modulo_11_l907_90772


namespace remainder_2_power_2015_mod_20_l907_90754

theorem remainder_2_power_2015_mod_20 : ∃ (seq : Fin 4 → Nat),
  (∀ (n : Nat), (2^n : Nat) % 20 = seq (n % 4)) ∧
  (seq 0 = 4 ∧ seq 1 = 8 ∧ seq 2 = 16 ∧ seq 3 = 12) →
  (2^2015 : Nat) % 20 = 8 := by
sorry

end remainder_2_power_2015_mod_20_l907_90754


namespace fgh_supermarkets_in_us_l907_90720

theorem fgh_supermarkets_in_us (total : ℕ) (difference : ℕ) : 
  total = 84 → difference = 10 → (total / 2 + difference / 2) = 47 := by
  sorry

end fgh_supermarkets_in_us_l907_90720


namespace product_equals_900_l907_90782

theorem product_equals_900 (a : ℝ) (h : (a + 25)^2 = 1000) : (a + 15) * (a + 35) = 900 := by
  sorry

end product_equals_900_l907_90782


namespace max_attendance_difference_l907_90709

-- Define the estimates and error margins
def chloe_estimate : ℝ := 40000
def derek_estimate : ℝ := 55000
def emma_estimate : ℝ := 75000

def chloe_error : ℝ := 0.05
def derek_error : ℝ := 0.15
def emma_error : ℝ := 0.10

-- Define the ranges for actual attendances
def chicago_range : Set ℝ := {x | chloe_estimate * (1 - chloe_error) ≤ x ∧ x ≤ chloe_estimate * (1 + chloe_error)}
def denver_range : Set ℝ := {x | derek_estimate / (1 + derek_error) ≤ x ∧ x ≤ derek_estimate / (1 - derek_error)}
def miami_range : Set ℝ := {x | emma_estimate * (1 - emma_error) ≤ x ∧ x ≤ emma_estimate * (1 + emma_error)}

-- Define the theorem
theorem max_attendance_difference :
  ∃ (c d m : ℝ),
    c ∈ chicago_range ∧
    d ∈ denver_range ∧
    m ∈ miami_range ∧
    (⌊(max c (max d m) - min c (min d m) + 500) / 1000⌋ * 1000 = 45000) :=
sorry

end max_attendance_difference_l907_90709


namespace repair_cost_theorem_l907_90745

def new_shoes_cost : ℝ := 28
def new_shoes_lifespan : ℝ := 2
def used_shoes_lifespan : ℝ := 1
def percentage_difference : ℝ := 0.2173913043478261

theorem repair_cost_theorem :
  ∃ (repair_cost : ℝ),
    repair_cost = 11.50 ∧
    (new_shoes_cost / new_shoes_lifespan) = repair_cost * (1 + percentage_difference) :=
by sorry

end repair_cost_theorem_l907_90745


namespace quadratic_two_distinct_real_roots_l907_90781

theorem quadratic_two_distinct_real_roots :
  ∃ x y : ℝ, x ≠ y ∧ (2 * x^2 + 3 * x - 4 = 0) ∧ (2 * y^2 + 3 * y - 4 = 0) := by
  sorry

end quadratic_two_distinct_real_roots_l907_90781


namespace sum_of_sequences_l907_90728

theorem sum_of_sequences : (2+12+22+32+42) + (10+20+30+40+50) = 260 := by
  sorry

end sum_of_sequences_l907_90728


namespace equilateral_triangle_area_perimeter_ratio_l907_90764

/-- The ratio of the area to the perimeter of an equilateral triangle with side length 6 -/
theorem equilateral_triangle_area_perimeter_ratio :
  let side_length : ℝ := 6
  let perimeter : ℝ := 3 * side_length
  let area : ℝ := (side_length^2 * Real.sqrt 3) / 4
  area / perimeter = Real.sqrt 3 / 2 := by
sorry

end equilateral_triangle_area_perimeter_ratio_l907_90764


namespace factorization_and_difference_l907_90750

theorem factorization_and_difference (y : ℤ) : ∃ (a b : ℤ), 
  (4 * y^2 - 3 * y - 28 = (4 * y + a) * (y + b)) ∧ (a - b = 11) := by
  sorry

end factorization_and_difference_l907_90750


namespace projection_of_vectors_l907_90785

/-- Given two vectors in ℝ², prove that the projection of one onto the other is as specified. -/
theorem projection_of_vectors (a b : ℝ × ℝ) (h1 : a = (0, 1)) (h2 : b = (1, Real.sqrt 3)) :
  (a • b / (b • b)) • b = (Real.sqrt 3 / 4) • b :=
sorry

end projection_of_vectors_l907_90785


namespace losing_candidate_vote_percentage_l907_90793

/-- Given a total number of votes and a loss margin, calculate the percentage of votes received by the losing candidate. -/
def calculate_vote_percentage (total_votes : ℕ) (loss_margin : ℕ) : ℚ :=
  let candidate_votes := (total_votes - loss_margin) / 2
  (candidate_votes : ℚ) / total_votes * 100

/-- Theorem stating that given 7000 total votes and a loss margin of 2100 votes, the losing candidate received 35% of the votes. -/
theorem losing_candidate_vote_percentage :
  calculate_vote_percentage 7000 2100 = 35 := by
  sorry

end losing_candidate_vote_percentage_l907_90793


namespace arithmetic_geometric_sequence_l907_90714

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) :
  (∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence condition
  a 3 = 3 →  -- given condition
  a 2 ^ 2 = a 1 * a 4 →  -- geometric sequence condition
  a 5 = 5 ∨ a 5 = 3 := by
  sorry

end arithmetic_geometric_sequence_l907_90714


namespace spy_arrangement_exists_l907_90773

-- Define the board
def Board := Fin 6 → Fin 6 → Bool

-- Define the direction a spy can face
inductive Direction
| North
| East
| South
| West

-- Define a spy's position and direction
structure Spy where
  row : Fin 6
  col : Fin 6
  dir : Direction

-- Define the visibility function for a spy
def canSee (s : Spy) (r : Fin 6) (c : Fin 6) : Prop :=
  match s.dir with
  | Direction.North => 
      (s.row > r && s.row - r ≤ 2 && s.col = c) || 
      (s.row = r && (s.col = c + 1 || s.col + 1 = c))
  | Direction.East => 
      (s.col < c && c - s.col ≤ 2 && s.row = r) || 
      (s.col = c && (s.row = r + 1 || s.row + 1 = r))
  | Direction.South => 
      (s.row < r && r - s.row ≤ 2 && s.col = c) || 
      (s.row = r && (s.col = c + 1 || s.col + 1 = c))
  | Direction.West => 
      (s.col > c && s.col - c ≤ 2 && s.row = r) || 
      (s.col = c && (s.row = r + 1 || s.row + 1 = r))

-- Define a valid arrangement of spies
def validArrangement (spies : List Spy) : Prop :=
  spies.length = 18 ∧
  ∀ s1 s2, s1 ∈ spies → s2 ∈ spies → s1 ≠ s2 →
    ¬(canSee s1 s2.row s2.col) ∧ ¬(canSee s2 s1.row s1.col)

-- Theorem: There exists a valid arrangement of 18 spies
theorem spy_arrangement_exists : ∃ spies : List Spy, validArrangement spies := by
  sorry

end spy_arrangement_exists_l907_90773


namespace laura_biathlon_l907_90795

/-- Laura's biathlon training problem -/
theorem laura_biathlon (x : ℝ) : x > 0 → (25 / (3*x + 2) + 4 / x + 8/60 = 140/60) → (6.6*x^2 - 32.6*x - 8 = 0) := by
  sorry

end laura_biathlon_l907_90795


namespace train_passing_jogger_l907_90707

/-- Time for a train to pass a jogger given their speeds and initial positions -/
theorem train_passing_jogger (jogger_speed train_speed : ℝ) (initial_distance train_length : ℝ) : 
  jogger_speed = 9 →
  train_speed = 45 →
  initial_distance = 360 →
  train_length = 180 →
  (initial_distance + train_length) / (train_speed - jogger_speed) * (3600 / 1000) = 54 :=
by sorry

end train_passing_jogger_l907_90707


namespace joes_total_lift_weight_l907_90718

-- Define the weights of the lifts
def first_lift : ℕ := 400
def second_lift : ℕ := 2 * first_lift - 300

-- Define the total weight
def total_weight : ℕ := first_lift + second_lift

-- Theorem statement
theorem joes_total_lift_weight : total_weight = 900 := by
  sorry

end joes_total_lift_weight_l907_90718


namespace condition_analysis_l907_90730

theorem condition_analysis (x y : ℝ) : 
  (∀ x y : ℝ, (x - 1)^2 + (y - 2)^2 = 0 → (x - 1) * (y - 2) = 0) ∧ 
  (∃ x y : ℝ, (x - 1) * (y - 2) = 0 ∧ (x - 1)^2 + (y - 2)^2 ≠ 0) := by
  sorry

end condition_analysis_l907_90730


namespace fraction_comparison_l907_90752

theorem fraction_comparison (m n : ℕ) (h1 : m > 0) (h2 : n > 0) (h3 : m < n) :
  (m + 3 : ℚ) / (n + 3) > (m : ℚ) / n := by
  sorry

end fraction_comparison_l907_90752


namespace M_equals_reals_l907_90780

def M : Set ℂ := {z : ℂ | Complex.abs ((z - 1)^2) = Complex.abs (z - 1)^2}

theorem M_equals_reals : M = {z : ℂ | z.im = 0} := by sorry

end M_equals_reals_l907_90780


namespace arithmetic_sequence_inequality_l907_90776

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

/-- The theorem statement -/
theorem arithmetic_sequence_inequality (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a)
  (h_inequality : (a 5 + a 6 + a 7 + a 8) * (a 6 + a 7 + a 8) < 0) :
  |a 6| > |a 7| := by
  sorry

end arithmetic_sequence_inequality_l907_90776


namespace smallest_y_value_l907_90789

theorem smallest_y_value (x y : ℤ) (h : x * y + 3 * x + 2 * y = 1) : 
  ∀ (z : ℤ), z ≥ -10 ∨ ¬∃ (w : ℤ), w * z + 3 * w + 2 * z = 1 :=
sorry

end smallest_y_value_l907_90789


namespace train_passing_time_l907_90760

/-- The time taken for a train to pass a telegraph post -/
theorem train_passing_time (train_length : ℝ) (train_speed_kmph : ℝ) : 
  train_length = 60 →
  train_speed_kmph = 36 →
  (train_length / (train_speed_kmph * (5/18))) = 6 := by
  sorry

end train_passing_time_l907_90760


namespace circle_diameter_from_area_l907_90771

theorem circle_diameter_from_area (A : ℝ) (h : A = 196 * Real.pi) :
  ∃ (d : ℝ), d = 28 ∧ A = Real.pi * (d / 2)^2 := by
  sorry

end circle_diameter_from_area_l907_90771


namespace extremum_implies_a_eq_neg_two_l907_90737

/-- The function f(x) = a ln x + x^2 has an extremum at x = 1 -/
def has_extremum_at_one (a : ℝ) : Prop :=
  let f := fun (x : ℝ) => a * Real.log x + x^2
  ∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), x ≠ 1 ∧ |x - 1| < ε → f x ≤ f 1 ∨ f x ≥ f 1

/-- If f(x) = a ln x + x^2 has an extremum at x = 1, then a = -2 -/
theorem extremum_implies_a_eq_neg_two (a : ℝ) :
  has_extremum_at_one a → a = -2 :=
by sorry

end extremum_implies_a_eq_neg_two_l907_90737


namespace line_m_equation_l907_90757

/-- Two distinct lines in the xy-plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflection of a point about a line -/
def reflect (p : Point) (l : Line) : Point :=
  sorry

theorem line_m_equation (ℓ m : Line) (Q Q'' : Point) :
  ℓ.a = 1 ∧ ℓ.b = 3 ∧ ℓ.c = 7 ∧  -- Equation of line ℓ: x + 3y = 7
  Q.x = 2 ∧ Q.y = 5 ∧  -- Coordinates of Q
  Q''.x = 5 ∧ Q''.y = 0 ∧  -- Coordinates of Q''
  (1 : ℝ) * ℓ.a + 2 * ℓ.b = ℓ.c ∧  -- ℓ passes through (1, 2)
  (1 : ℝ) * m.a + 2 * m.b = m.c ∧  -- m passes through (1, 2)
  Q'' = reflect (reflect Q ℓ) m →  -- Q'' is the result of reflecting Q about ℓ and then m
  m.a = 2 ∧ m.b = -1 ∧ m.c = 2  -- Equation of line m: 2x - y = 2
  := by sorry

end line_m_equation_l907_90757


namespace interior_angle_sum_l907_90766

theorem interior_angle_sum (n : ℕ) : 
  (180 * (n - 2) = 1800) → (180 * ((n + 4) - 2) = 2520) :=
by
  sorry

end interior_angle_sum_l907_90766


namespace distinct_digit_numbers_count_l907_90732

/-- A function that counts the number of integers between 1000 and 9999 with four distinct digits -/
def count_distinct_digit_numbers : ℕ :=
  9 * 9 * 8 * 7

/-- The theorem stating that the count of integers between 1000 and 9999 with four distinct digits is 4536 -/
theorem distinct_digit_numbers_count :
  count_distinct_digit_numbers = 4536 := by
  sorry

end distinct_digit_numbers_count_l907_90732


namespace product_comparison_l907_90715

theorem product_comparison (a : Fin 10 → ℝ) 
  (h_pos : ∀ i, a i > 0) 
  (h_distinct : ∀ i j, i ≠ j → a i ≠ a j) : 
  (∃ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 
    (∀ l m, l ≠ i ∧ l ≠ j ∧ l ≠ k ∧ m ≠ i ∧ m ≠ j ∧ m ≠ k ∧ l ≠ m → 
      a i * a j * a k > a l * a m)) ∨
  (∃ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 
    (∀ l m n o, l ≠ i ∧ l ≠ j ∧ l ≠ k ∧ m ≠ i ∧ m ≠ j ∧ m ≠ k ∧ 
      n ≠ i ∧ n ≠ j ∧ n ≠ k ∧ o ≠ i ∧ o ≠ j ∧ o ≠ k ∧ 
      l ≠ m ∧ l ≠ n ∧ l ≠ o ∧ m ≠ n ∧ m ≠ o ∧ n ≠ o → 
      a i * a j * a k > a l * a m * a n * a o)) := by
sorry

end product_comparison_l907_90715


namespace roots_sum_reciprocal_cubes_l907_90735

theorem roots_sum_reciprocal_cubes (r s : ℝ) : 
  (3 * r^2 + 5 * r + 2 = 0) → 
  (3 * s^2 + 5 * s + 2 = 0) → 
  (r ≠ s) →
  (1 / r^3 + 1 / s^3 = -27 / 35) :=
by sorry

end roots_sum_reciprocal_cubes_l907_90735


namespace total_interest_calculation_l907_90724

/-- Calculate total interest over 10 years with principal trebling after 5 years -/
theorem total_interest_calculation (P R : ℝ) 
  (h1 : P * R * 10 / 100 = 600) : 
  P * R * 5 / 100 + 3 * P * R * 5 / 100 = 1140 := by
  sorry

end total_interest_calculation_l907_90724


namespace intersection_implies_a_equals_one_l907_90704

def A : Set ℝ := {x | x ≤ 1}
def B (a : ℝ) : Set ℝ := {x | x ≥ a}

theorem intersection_implies_a_equals_one (a : ℝ) :
  A ∩ B a = {1} → a = 1 := by sorry

end intersection_implies_a_equals_one_l907_90704


namespace unifying_sqrt_plus_m_range_l907_90733

/-- A function is unifying on [a,b] if it's monotonic and maps [a,b] onto itself --/
def IsUnifying (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  a < b ∧ 
  Monotone f ∧ 
  (∀ y ∈ Set.Icc a b, ∃ x ∈ Set.Icc a b, f x = y)

/-- The theorem stating the range of m for which f(x) = √(x+1) + m is a unifying function --/
theorem unifying_sqrt_plus_m_range :
  ∃ a b : ℝ, a < b ∧ 
  (∃ m : ℝ, IsUnifying (fun x ↦ Real.sqrt (x + 1) + m) a b) ↔ 
  m ∈ Set.Ioo (-5/4) (-1) ∪ {-1} :=
sorry

end unifying_sqrt_plus_m_range_l907_90733


namespace work_efficiency_ratio_l907_90774

/-- Given two workers A and B, their combined work efficiency, and A's individual efficiency,
    prove the ratio of their efficiencies. -/
theorem work_efficiency_ratio 
  (total_days : ℝ) 
  (a_days : ℝ) 
  (h1 : total_days = 12) 
  (h2 : a_days = 16) : 
  (1 / a_days) / ((1 / total_days) - (1 / a_days)) = 3 := by
  sorry

#check work_efficiency_ratio

end work_efficiency_ratio_l907_90774


namespace functional_equation_proof_l907_90759

open Real

theorem functional_equation_proof (x : ℝ) (hx : x ≠ 0) :
  let f : ℝ → ℝ := λ x => (x / 3) + (2 / (3 * x))
  2 * f x - f (1 / x) = 1 / x := by sorry

end functional_equation_proof_l907_90759


namespace equation_solution_l907_90786

theorem equation_solution (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) :
  (3 / (x - 1) - 2 / x = 0) ↔ (x = -2) := by
  sorry

end equation_solution_l907_90786


namespace sqrt_equation_solution_l907_90769

theorem sqrt_equation_solution :
  ∃ x : ℝ, (Real.sqrt (5 * x + 15) = 15) ∧ (x = 42) := by sorry

end sqrt_equation_solution_l907_90769


namespace spider_movement_limit_l907_90711

/-- Represents the spider's position and movement on the wall --/
structure SpiderPosition :=
  (height : ℝ)  -- Current height of the spider
  (day : ℕ)     -- Current day

/-- Defines the daily movement of the spider --/
def daily_movement (sp : SpiderPosition) : SpiderPosition :=
  ⟨sp.height + 2, sp.day + 1⟩

/-- Checks if the spider can be moved up 3 feet --/
def can_move_up (sp : SpiderPosition) (wall_height : ℝ) : Prop :=
  sp.height + 3 ≤ wall_height

/-- Theorem: Tony runs out of room after 8 days --/
theorem spider_movement_limit :
  ∀ (wall_height : ℝ) (initial_height : ℝ),
  wall_height = 18 → initial_height = 3 →
  ∃ (n : ℕ), n = 8 ∧
  ¬(can_move_up (n.iterate daily_movement ⟨initial_height, 0⟩) wall_height) ∧
  ∀ (m : ℕ), m < n →
  can_move_up (m.iterate daily_movement ⟨initial_height, 0⟩) wall_height :=
by sorry

end spider_movement_limit_l907_90711


namespace function_range_theorem_l907_90713

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem function_range_theorem (f : ℝ → ℝ) (a : ℝ) :
  (∀ x ≠ 0, is_odd_function f) →
  (∀ x ≠ 0, f (x + 5/2) * f x = 1) →
  f (-1) > 1 →
  f 2016 = (a + 3) / (a - 3) →
  0 < a ∧ a < 3 := by sorry

end function_range_theorem_l907_90713


namespace factorization_equality_l907_90777

theorem factorization_equality (x y : ℝ) : x^2 - 1 + 2*x*y + y^2 = (x+y+1)*(x+y-1) := by sorry

end factorization_equality_l907_90777


namespace fraction_product_result_l907_90758

def fraction_product (n : ℕ) : ℚ :=
  let seq (k : ℕ) := 2 + 3 * k
  (seq 0) / (seq n)

theorem fraction_product_result :
  fraction_product 667 = 2 / 2007 := by sorry

end fraction_product_result_l907_90758


namespace quadratic_parabola_properties_l907_90767

/-- Represents a quadratic equation of the form ax^2 + bx + c = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a parabola of the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Given quadratic equation and parabola have two distinct real roots and specific form when intersecting x-axis symmetrically -/
theorem quadratic_parabola_properties (m : ℝ) :
  let q : QuadraticEquation := ⟨1, -2*m, m^2 - 4⟩
  let p : Parabola := ⟨1, -2*m, m^2 - 4⟩
  -- The quadratic equation has two distinct real roots
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ q.a * x₁^2 + q.b * x₁ + q.c = 0 ∧ q.a * x₂^2 + q.b * x₂ + q.c = 0 ∧
  -- When the parabola intersects x-axis symmetrically, it has the form y = x^2 - 4
  (∃ (x₁ x₂ : ℝ), x₁ < 0 ∧ x₂ > 0 ∧ x₁ = -x₂ ∧ 
   p.a * x₁^2 + p.b * x₁ + p.c = 0 ∧ p.a * x₂^2 + p.b * x₂ + p.c = 0) →
  p = ⟨1, 0, -4⟩ := by
  sorry

end quadratic_parabola_properties_l907_90767


namespace new_average_after_removing_scores_l907_90722

theorem new_average_after_removing_scores (n : ℕ) (original_avg : ℚ) (score1 score2 : ℕ) :
  n = 60 →
  original_avg = 82 →
  score1 = 95 →
  score2 = 97 →
  let total_sum := n * original_avg
  let remaining_sum := total_sum - (score1 + score2)
  let new_avg := remaining_sum / (n - 2)
  new_avg = 81.52 := by sorry

end new_average_after_removing_scores_l907_90722


namespace product_zero_l907_90784

theorem product_zero (b : ℤ) (h : b = 3) : 
  (b - 12) * (b - 11) * (b - 10) * (b - 9) * (b - 8) * (b - 7) * (b - 6) * (b - 5) * (b - 4) * (b - 3) * (b - 2) * (b - 1) * b = 0 := by
  sorry

end product_zero_l907_90784


namespace speed_of_car_C_l907_90731

/-- Proves that given the conditions of the problem, the speed of car C is 26 km/h --/
theorem speed_of_car_C (v_A v_B : ℝ) (t_A t_B t_C : ℝ) :
  v_A = 24 →
  v_B = 20 →
  t_A = 5 / 60 →
  t_B = 10 / 60 →
  t_C = 12 / 60 →
  v_A * t_A = v_B * t_B →
  ∃ (v_C : ℝ), v_C * t_C = v_A * t_A ∧ v_C = 26 :=
by sorry

#check speed_of_car_C

end speed_of_car_C_l907_90731


namespace negation_of_cube_odd_l907_90736

theorem negation_of_cube_odd (P : ℕ → Prop) :
  (¬ ∀ x : ℕ, Odd x → Odd (x^3)) ↔ (∃ x : ℕ, Odd x ∧ Even (x^3)) :=
by sorry

end negation_of_cube_odd_l907_90736


namespace number_relationship_l907_90770

theorem number_relationship (a b c : ℤ) : 
  (a + b + c = 264) → 
  (a = 2 * b) → 
  (b = 72) → 
  (c = a - 96) := by
sorry

end number_relationship_l907_90770


namespace floor_product_equals_45_l907_90761

theorem floor_product_equals_45 (x : ℝ) : 
  ⌊x * ⌊x⌋⌋ = 45 ↔ x ∈ Set.Icc (7.5) (7 + 2/3) := by sorry

end floor_product_equals_45_l907_90761


namespace negation_of_existential_l907_90756

theorem negation_of_existential (P : α → Prop) : 
  (¬∃ x, P x) ↔ (∀ x, ¬P x) := by sorry

end negation_of_existential_l907_90756


namespace triangle_angle_calculation_l907_90710

/-- Proves that in a triangle ABC, if angle C is triple angle B and angle B is 18°, then angle A is 108° -/
theorem triangle_angle_calculation (A B C : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C → -- Angles are positive
  A + B + C = 180 → -- Sum of angles in a triangle
  C = 3 * B → -- Angle C is triple angle B
  B = 18 → -- Angle B is 18°
  A = 108 := by sorry

end triangle_angle_calculation_l907_90710


namespace angle_with_same_terminal_side_as_negative_950_degrees_l907_90799

theorem angle_with_same_terminal_side_as_negative_950_degrees :
  ∃ θ : ℝ, 0 ≤ θ ∧ θ < 180 ∧ ∃ k : ℤ, θ = -950 + 360 * k ∧ θ = 130 := by
  sorry

end angle_with_same_terminal_side_as_negative_950_degrees_l907_90799


namespace break_even_price_l907_90734

/-- Calculates the minimum selling price per component to break even -/
def minimum_selling_price (production_cost shipping_cost : ℚ) (fixed_costs : ℚ) (volume : ℕ) : ℚ :=
  (production_cost + shipping_cost + fixed_costs / volume)

theorem break_even_price 
  (production_cost : ℚ) 
  (shipping_cost : ℚ) 
  (fixed_costs : ℚ) 
  (volume : ℕ) 
  (h1 : production_cost = 80)
  (h2 : shipping_cost = 2)
  (h3 : fixed_costs = 16200)
  (h4 : volume = 150) :
  minimum_selling_price production_cost shipping_cost fixed_costs volume = 190 := by
  sorry

#eval minimum_selling_price 80 2 16200 150

end break_even_price_l907_90734


namespace expected_bullets_remaining_l907_90797

/-- The probability of hitting the target -/
def hit_probability : ℝ := 0.6

/-- The total number of bullets -/
def total_bullets : ℕ := 4

/-- The expected number of bullets remaining after stopping the shooting -/
def expected_remaining_bullets : ℝ := 2.376

/-- Theorem stating that the expected number of bullets remaining is 2.376 -/
theorem expected_bullets_remaining :
  let p := hit_probability
  let n := total_bullets
  let E := expected_remaining_bullets
  E = (0 * (1 - p)^3 + 1 * p * (1 - p)^2 + 2 * p * (1 - p) + 3 * p) := by
  sorry

end expected_bullets_remaining_l907_90797


namespace intersection_of_A_and_B_l907_90705

def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {x | 0 < x ∧ x < 3}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by sorry

end intersection_of_A_and_B_l907_90705


namespace l_shaped_area_is_23_l907_90741

-- Define the side lengths
def large_square_side : ℝ := 8
def medium_square_side : ℝ := 4
def small_square_side : ℝ := 3

-- Define the areas
def large_square_area : ℝ := large_square_side ^ 2
def medium_square_area : ℝ := medium_square_side ^ 2
def small_square_area : ℝ := small_square_side ^ 2

-- Define the L-shaped area
def l_shaped_area : ℝ := large_square_area - (2 * medium_square_area + small_square_area)

-- Theorem statement
theorem l_shaped_area_is_23 : l_shaped_area = 23 := by
  sorry

end l_shaped_area_is_23_l907_90741


namespace package_weight_sum_l907_90725

theorem package_weight_sum (x y z : ℝ) 
  (h1 : x + y = 112)
  (h2 : y + z = 118)
  (h3 : z + x = 120) :
  x + y + z = 175 := by
sorry

end package_weight_sum_l907_90725


namespace garden_area_unchanged_l907_90751

/-- Represents a rectangular garden with given length and width -/
structure RectangularGarden where
  length : ℝ
  width : ℝ

/-- Represents a square garden with a given side length -/
structure SquareGarden where
  side : ℝ

/-- Calculates the area of a rectangular garden -/
def area_rectangular (g : RectangularGarden) : ℝ := g.length * g.width

/-- Calculates the perimeter of a rectangular garden -/
def perimeter_rectangular (g : RectangularGarden) : ℝ := 2 * (g.length + g.width)

/-- Calculates the area of a square garden -/
def area_square (g : SquareGarden) : ℝ := g.side * g.side

/-- Calculates the perimeter of a square garden -/
def perimeter_square (g : SquareGarden) : ℝ := 4 * g.side

theorem garden_area_unchanged 
  (rect : RectangularGarden) 
  (sq : SquareGarden) 
  (partition_length : ℝ) :
  rect.length = 60 →
  rect.width = 15 →
  partition_length = 30 →
  perimeter_rectangular rect = perimeter_square sq + partition_length →
  area_rectangular rect = area_square sq :=
by sorry

end garden_area_unchanged_l907_90751


namespace abs_inequality_l907_90723

theorem abs_inequality (a b c : ℝ) (h : |a - c| < |b|) : |a| < |b| + |c| := by
  sorry

end abs_inequality_l907_90723


namespace max_reverse_digit_diff_l907_90726

/-- Given two two-digit positive integers with the same digits in reverse order and
    their positive difference less than 60, the maximum difference is 54 -/
theorem max_reverse_digit_diff :
  ∀ q r : ℕ,
  10 ≤ q ∧ q < 100 →  -- q is a two-digit number
  10 ≤ r ∧ r < 100 →  -- r is a two-digit number
  ∃ a b : ℕ,
    0 ≤ a ∧ a ≤ 9 ∧   -- a is a digit
    0 ≤ b ∧ b ≤ 9 ∧   -- b is a digit
    q = 10 * a + b ∧  -- q's representation
    r = 10 * b + a ∧  -- r's representation
    (q > r → q - r < 60) ∧  -- positive difference less than 60
    (r > q → r - q < 60) →
  (∀ q' r' : ℕ,
    (∃ a' b' : ℕ,
      0 ≤ a' ∧ a' ≤ 9 ∧
      0 ≤ b' ∧ b' ≤ 9 ∧
      q' = 10 * a' + b' ∧
      r' = 10 * b' + a' ∧
      (q' > r' → q' - r' < 60) ∧
      (r' > q' → r' - q' < 60)) →
    q' - r' ≤ 54) ∧
  ∃ q₀ r₀ : ℕ, q₀ - r₀ = 54 ∧
    (∃ a₀ b₀ : ℕ,
      0 ≤ a₀ ∧ a₀ ≤ 9 ∧
      0 ≤ b₀ ∧ b₀ ≤ 9 ∧
      q₀ = 10 * a₀ + b₀ ∧
      r₀ = 10 * b₀ + a₀ ∧
      q₀ - r₀ < 60) :=
by sorry

end max_reverse_digit_diff_l907_90726


namespace dans_apples_l907_90727

theorem dans_apples (benny_apples total_apples : ℕ) 
  (h1 : benny_apples = 2)
  (h2 : total_apples = 11) :
  total_apples - benny_apples = 9 :=
by sorry

end dans_apples_l907_90727


namespace arithmetic_expression_equality_l907_90790

theorem arithmetic_expression_equality : 8 + 12 / 3 - 2^3 + 1 = 5 := by
  sorry

end arithmetic_expression_equality_l907_90790


namespace power_equation_solution_l907_90794

theorem power_equation_solution (m : ℝ) : (7 : ℝ) ^ (4 * m) = (1 / 7) ^ (2 * m - 18) → m = 3 := by
  sorry

end power_equation_solution_l907_90794


namespace max_value_expression_l907_90738

theorem max_value_expression (n : ℕ) (h : n = 15000) :
  let factorization := 2^3 * 3 * 5^4
  ∃ (x y : ℕ), 
    (2*x - y = 0 ∨ 3*x - y = 0) ∧ 
    (x ∣ n) ∧
    ∀ (x' y' : ℕ), (2*x' - y' = 0 ∨ 3*x' - y' = 0) ∧ (x' ∣ n) → 
      2*x + 3*y ≥ 2*x' + 3*y' ∧
      2*x + 3*y = 60000 := by
  sorry

end max_value_expression_l907_90738


namespace equal_pieces_after_exchanges_l907_90791

theorem equal_pieces_after_exchanges (initial_white : ℕ) (initial_black : ℕ) 
  (exchange_count : ℕ) (pieces_per_exchange : ℕ) :
  initial_white = 80 →
  initial_black = 50 →
  pieces_per_exchange = 3 →
  exchange_count = 5 →
  initial_white - exchange_count * pieces_per_exchange = 
  initial_black + exchange_count * pieces_per_exchange :=
by
  sorry

#check equal_pieces_after_exchanges

end equal_pieces_after_exchanges_l907_90791


namespace proposition_equivalence_l907_90746

theorem proposition_equivalence (p q : Prop) : 
  (¬(p ∨ q)) → ((¬p) ∧ (¬q)) := by
  sorry

end proposition_equivalence_l907_90746


namespace six_balls_removal_ways_l907_90787

/-- Represents the number of ways to remove n balls from a box, removing at least one at a time. -/
def removalWays (n : ℕ) : ℕ :=
  if n = 0 then 1
  else sorry  -- The actual implementation would go here

/-- The number of ways to remove 6 balls is 32. -/
theorem six_balls_removal_ways : removalWays 6 = 32 := by
  sorry  -- The proof would go here

end six_balls_removal_ways_l907_90787
