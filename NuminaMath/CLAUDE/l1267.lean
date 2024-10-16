import Mathlib

namespace NUMINAMATH_CALUDE_probability_of_listening_second_class_l1267_126772

/-- Represents the duration of a class in minutes -/
def class_duration : ℕ := 40

/-- Represents the duration of a break between classes in minutes -/
def break_duration : ℕ := 10

/-- Represents the start time of the first class in minutes after midnight -/
def first_class_start : ℕ := 8 * 60

/-- Represents the earliest arrival time of the student in minutes after midnight -/
def earliest_arrival : ℕ := 9 * 60 + 10

/-- Represents the latest arrival time of the student in minutes after midnight -/
def latest_arrival : ℕ := 10 * 60

/-- Represents the duration of the arrival window in minutes -/
def arrival_window : ℕ := latest_arrival - earliest_arrival

/-- Represents the duration of the favorable arrival window in minutes -/
def favorable_window : ℕ := 10

/-- The probability of the student listening to the second class for no less than 10 minutes -/
theorem probability_of_listening_second_class :
  (favorable_window : ℚ) / arrival_window = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_probability_of_listening_second_class_l1267_126772


namespace NUMINAMATH_CALUDE_yangbajing_largest_in_1975_l1267_126720

/-- Represents a geothermal power station -/
structure GeothermalStation where
  name : String
  capacity : ℕ  -- capacity in kilowatts
  country : String
  year_established : ℕ

/-- The set of all geothermal power stations in China in 1975 -/
def china_geothermal_stations_1975 : Set GeothermalStation :=
  sorry

/-- The Yangbajing Geothermal Power Station -/
def yangbajing : GeothermalStation :=
  { name := "Yangbajing Geothermal Power Station"
  , capacity := 50  -- 50 kilowatts in 1975
  , country := "China"
  , year_established := 1975 }

/-- Theorem: Yangbajing was the largest geothermal power station in China in 1975 -/
theorem yangbajing_largest_in_1975 :
  yangbajing ∈ china_geothermal_stations_1975 ∧
  ∀ s ∈ china_geothermal_stations_1975, s.capacity ≤ yangbajing.capacity :=
by
  sorry

end NUMINAMATH_CALUDE_yangbajing_largest_in_1975_l1267_126720


namespace NUMINAMATH_CALUDE_jelly_bean_difference_l1267_126732

theorem jelly_bean_difference (total : ℕ) (vanilla : ℕ) (grape : ℕ) 
  (h1 : total = 770)
  (h2 : vanilla = 120)
  (h3 : total = grape + vanilla)
  (h4 : grape > 5 * vanilla) :
  grape - 5 * vanilla = 50 := by
  sorry

end NUMINAMATH_CALUDE_jelly_bean_difference_l1267_126732


namespace NUMINAMATH_CALUDE_intersection_chord_length_l1267_126745

/-- The length of the chord formed by the intersection of ρ = 4sin θ and ρ cos θ = 1 -/
theorem intersection_chord_length :
  ∃ (A B : ℝ × ℝ),
    (∃ θ₁, (4 * Real.sin θ₁ * Real.cos θ₁, 4 * Real.sin θ₁ * Real.sin θ₁) = A) ∧
    (∃ θ₂, (4 * Real.sin θ₂ * Real.cos θ₂, 4 * Real.sin θ₂ * Real.sin θ₂) = B) ∧
    (A.1 = 1) ∧ (B.1 = 1) ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_intersection_chord_length_l1267_126745


namespace NUMINAMATH_CALUDE_q_satisfies_conditions_l1267_126713

/-- A quadratic polynomial satisfying specific conditions -/
def q (x : ℚ) : ℚ := (20/7) * x^2 + (40/7) * x - 300/7

/-- Proof that q satisfies the given conditions -/
theorem q_satisfies_conditions :
  q (-5) = 0 ∧ q 3 = 0 ∧ q 2 = -20 :=
by sorry


end NUMINAMATH_CALUDE_q_satisfies_conditions_l1267_126713


namespace NUMINAMATH_CALUDE_smallest_factor_sum_factorization_exists_l1267_126770

theorem smallest_factor_sum (b : ℤ) : 
  (∃ (p q : ℤ), x^2 + b*x + 2007 = (x + p) * (x + q)) →
  b ≥ 232 :=
by sorry

theorem factorization_exists : 
  ∃ (b p q : ℤ), (b = p + q) ∧ (p * q = 2007) ∧ 
  (x^2 + b*x + 2007 = (x + p) * (x + q)) ∧
  (b = 232) :=
by sorry

end NUMINAMATH_CALUDE_smallest_factor_sum_factorization_exists_l1267_126770


namespace NUMINAMATH_CALUDE_unique_solution_l1267_126709

/-- The sequence x_n defined by x_n = n / (n + 2016) -/
def x (n : ℕ) : ℚ := n / (n + 2016)

/-- Theorem stating the unique solution for m and n -/
theorem unique_solution :
  ∃! (m n : ℕ), m > 0 ∧ n > 0 ∧ x 2016 = x m * x n ∧ m = 6048 ∧ n = 4032 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l1267_126709


namespace NUMINAMATH_CALUDE_incorrect_expression_l1267_126716

theorem incorrect_expression (x y : ℝ) (h : x / y = 5 / 6) : 
  ¬((2 * x - y) / y = 4 / 3) := by
sorry

end NUMINAMATH_CALUDE_incorrect_expression_l1267_126716


namespace NUMINAMATH_CALUDE_sequence_2014_term_l1267_126749

/-- A positive sequence satisfying the given recurrence relation -/
def PositiveSequence (a : ℕ+ → ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ+, n * a (n + 1) = (n + 1) * a n ∧ 0 < a n

/-- The 2014th term of the sequence is equal to 2014 -/
theorem sequence_2014_term (a : ℕ+ → ℝ) (h : PositiveSequence a) : a 2014 = 2014 := by
  sorry

end NUMINAMATH_CALUDE_sequence_2014_term_l1267_126749


namespace NUMINAMATH_CALUDE_unique_k_value_l1267_126757

theorem unique_k_value : ∃! k : ℝ, ∀ x : ℝ, x * (2 * x + 3) < k ↔ -5/2 < x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_k_value_l1267_126757


namespace NUMINAMATH_CALUDE_common_roots_product_l1267_126742

-- Define the cubic equations
def cubic1 (C : ℝ) (x : ℝ) : ℝ := x^3 + C*x + 20
def cubic2 (D : ℝ) (x : ℝ) : ℝ := x^3 + D*x^2 + 80

-- Define the theorem
theorem common_roots_product (C D : ℝ) (u v : ℝ) :
  (∃ w, cubic1 C u = 0 ∧ cubic1 C v = 0 ∧ cubic1 C w = 0) →
  (∃ t, cubic2 D u = 0 ∧ cubic2 D v = 0 ∧ cubic2 D t = 0) →
  u * v = 10 * Real.rpow 4 (1/3) :=
by sorry

end NUMINAMATH_CALUDE_common_roots_product_l1267_126742


namespace NUMINAMATH_CALUDE_least_years_to_double_l1267_126734

-- Define the interest rate
def interest_rate : ℝ := 0.5

-- Define the function for the amount after t years
def amount (t : ℕ) : ℝ := (1 + interest_rate) ^ t

-- Theorem statement
theorem least_years_to_double :
  ∀ t : ℕ, t < 2 → amount t ≤ 2 ∧ 2 < amount 2 :=
by sorry

end NUMINAMATH_CALUDE_least_years_to_double_l1267_126734


namespace NUMINAMATH_CALUDE_price_reduction_percentage_l1267_126714

/-- Given a price reduction scenario, prove that the first reduction percentage is 25% -/
theorem price_reduction_percentage (P : ℝ) (x : ℝ) : 
  P * (1 - x / 100) * (1 - 60 / 100) = P * (1 - 70 / 100) → x = 25 := by
  sorry

#check price_reduction_percentage

end NUMINAMATH_CALUDE_price_reduction_percentage_l1267_126714


namespace NUMINAMATH_CALUDE_history_book_pages_l1267_126702

theorem history_book_pages (novel_pages science_pages history_pages : ℕ) : 
  novel_pages = history_pages / 2 →
  science_pages = 4 * novel_pages →
  science_pages = 600 →
  history_pages = 300 := by
sorry

end NUMINAMATH_CALUDE_history_book_pages_l1267_126702


namespace NUMINAMATH_CALUDE_christina_distance_to_friend_l1267_126746

/-- The distance Christina walks to school (in km) -/
def distance_to_school : ℝ := 7

/-- The number of days Christina walks to school in a week -/
def days_per_week : ℕ := 5

/-- The total distance Christina covered in the week (in km) -/
def total_weekly_distance : ℝ := 74

/-- The distance to Christina's mother's friend's place from school (in km) -/
def distance_to_friend : ℝ := 4

theorem christina_distance_to_friend :
  distance_to_friend = total_weekly_distance - (2 * distance_to_school * (days_per_week - 1)) - distance_to_school := by
  sorry

end NUMINAMATH_CALUDE_christina_distance_to_friend_l1267_126746


namespace NUMINAMATH_CALUDE_roots_of_quadratic_equation_l1267_126797

theorem roots_of_quadratic_equation (α β : ℝ) : 
  (∀ x : ℝ, x^2 - 3*x + 1 = 0 ↔ x = α ∨ x = β) →
  7 * α^3 + 10 * β^4 = 697 := by
  sorry

end NUMINAMATH_CALUDE_roots_of_quadratic_equation_l1267_126797


namespace NUMINAMATH_CALUDE_face_value_from_discounts_l1267_126790

/-- Face value calculation given banker's discount and true discount -/
theorem face_value_from_discounts
  (BD : ℚ) -- Banker's discount
  (TD : ℚ) -- True discount
  (h1 : BD = 42)
  (h2 : TD = 36)
  : BD = TD + (BD - TD) :=
by
  sorry

#check face_value_from_discounts

end NUMINAMATH_CALUDE_face_value_from_discounts_l1267_126790


namespace NUMINAMATH_CALUDE_lucy_fraction_of_edna_l1267_126741

-- Define the field length
def field_length : ℚ := 24

-- Define Mary's distance as a fraction of the field length
def mary_distance : ℚ := 3/8 * field_length

-- Define Edna's distance as a fraction of Mary's distance
def edna_distance : ℚ := 2/3 * mary_distance

-- Define Lucy's distance as mary_distance - 4
def lucy_distance : ℚ := mary_distance - 4

-- Theorem to prove
theorem lucy_fraction_of_edna : lucy_distance / edna_distance = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_lucy_fraction_of_edna_l1267_126741


namespace NUMINAMATH_CALUDE_hotel_room_cost_l1267_126793

theorem hotel_room_cost (original_friends : ℕ) (additional_friends : ℕ) (cost_decrease : ℚ) :
  original_friends = 5 →
  additional_friends = 2 →
  cost_decrease = 15 →
  ∃ total_cost : ℚ,
    total_cost / original_friends - total_cost / (original_friends + additional_friends) = cost_decrease ∧
    total_cost = 262.5 := by
  sorry

end NUMINAMATH_CALUDE_hotel_room_cost_l1267_126793


namespace NUMINAMATH_CALUDE_product_evaluation_l1267_126788

theorem product_evaluation (a : ℤ) (h : a = 1) : (a - 3) * (a - 2) * (a - 1) * a * (a + 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_evaluation_l1267_126788


namespace NUMINAMATH_CALUDE_initial_milk_water_ratio_l1267_126733

/-- Proves that the initial ratio of milk to water is 3:1 given the conditions -/
theorem initial_milk_water_ratio 
  (total_volume : ℝ) 
  (added_water : ℝ) 
  (final_ratio : ℝ) :
  total_volume = 50 →
  added_water = 100 →
  final_ratio = 1/3 →
  ∃ (initial_milk initial_water : ℝ),
    initial_milk + initial_water = total_volume ∧
    initial_milk / (initial_water + added_water) = final_ratio ∧
    initial_milk / initial_water = 3 := by
  sorry

end NUMINAMATH_CALUDE_initial_milk_water_ratio_l1267_126733


namespace NUMINAMATH_CALUDE_bucket_capacity_problem_l1267_126763

/-- Proves that if reducing a bucket's capacity to 4/5 of its original requires 250 buckets to fill a tank, 
    then the number of buckets needed with the original capacity is 200. -/
theorem bucket_capacity_problem (tank_volume : ℝ) (original_capacity : ℝ) 
  (h1 : tank_volume > 0) (h2 : original_capacity > 0) :
  (tank_volume = 250 * (4/5 * original_capacity)) → 
  (tank_volume = 200 * original_capacity) :=
by
  sorry

#check bucket_capacity_problem

end NUMINAMATH_CALUDE_bucket_capacity_problem_l1267_126763


namespace NUMINAMATH_CALUDE_max_sum_of_cubes_l1267_126754

theorem max_sum_of_cubes (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h₁ : 0 ≤ x₁ ∧ x₁ ≤ 1) 
  (h₂ : 0 ≤ x₂ ∧ x₂ ≤ 1) 
  (h₃ : 0 ≤ x₃ ∧ x₃ ≤ 1) 
  (h₄ : 0 ≤ x₄ ∧ x₄ ≤ 1) 
  (h₅ : 0 ≤ x₅ ∧ x₅ ≤ 1) :
  |x₁ - x₂|^3 + |x₂ - x₃|^3 + |x₃ - x₄|^3 + |x₄ - x₅|^3 + |x₅ - x₁|^3 ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_cubes_l1267_126754


namespace NUMINAMATH_CALUDE_solution_sets_imply_a_minus_b_eq_neg_seven_l1267_126707

-- Define the solution sets A and B
def A : Set ℝ := {x | x^2 - x - 6 < 0}
def B : Set ℝ := {x | x^2 - 5*x + 4 < 0}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := A ∩ B

-- Define the coefficients a and b
def a : ℝ := -4
def b : ℝ := 3

-- Theorem statement
theorem solution_sets_imply_a_minus_b_eq_neg_seven :
  (A_intersect_B = {x | x^2 + a*x + b < 0}) →
  a - b = -7 := by sorry

end NUMINAMATH_CALUDE_solution_sets_imply_a_minus_b_eq_neg_seven_l1267_126707


namespace NUMINAMATH_CALUDE_projection_problem_l1267_126701

def vector1 : ℝ × ℝ := (3, -2)
def vector2 : ℝ × ℝ := (2, 5)

def is_projection (v p : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), p = (k * v.1, k * v.2)

theorem projection_problem (v : ℝ × ℝ) (p : ℝ × ℝ) :
  is_projection v p ∧ is_projection v p →
  p = (133/50, 49/50) := by sorry

end NUMINAMATH_CALUDE_projection_problem_l1267_126701


namespace NUMINAMATH_CALUDE_not_always_divisible_by_19_l1267_126730

theorem not_always_divisible_by_19 : ∃ (a b : ℤ), ¬(19 ∣ ((3*a + 2)^3 - (3*b + 2)^3)) := by
  sorry

end NUMINAMATH_CALUDE_not_always_divisible_by_19_l1267_126730


namespace NUMINAMATH_CALUDE_matrix_power_2023_l1267_126783

def A : Matrix (Fin 2) (Fin 2) ℤ := !![1, 0; 2, 1]

theorem matrix_power_2023 :
  A ^ 2023 = !![1, 0; 4046, 1] := by sorry

end NUMINAMATH_CALUDE_matrix_power_2023_l1267_126783


namespace NUMINAMATH_CALUDE_max_intersections_10_points_l1267_126736

/-- The maximum number of intersection points of perpendicular bisectors for n points -/
def max_intersections (n : ℕ) : ℕ :=
  Nat.choose n 3 + 3 * Nat.choose n 4

/-- Theorem stating the maximum number of intersection points for 10 points -/
theorem max_intersections_10_points :
  max_intersections 10 = 750 := by sorry

end NUMINAMATH_CALUDE_max_intersections_10_points_l1267_126736


namespace NUMINAMATH_CALUDE_prob_non_blue_specific_cube_l1267_126705

/-- A cube with colored faces -/
structure ColoredCube where
  green_faces : ℕ
  yellow_faces : ℕ
  blue_faces : ℕ

/-- The probability of rolling a non-blue face on a colored cube -/
def prob_non_blue (cube : ColoredCube) : ℚ :=
  (cube.green_faces + cube.yellow_faces) / (cube.green_faces + cube.yellow_faces + cube.blue_faces)

/-- Theorem: The probability of rolling a non-blue face on a cube with 3 green faces, 2 yellow faces, and 1 blue face is 5/6 -/
theorem prob_non_blue_specific_cube :
  prob_non_blue ⟨3, 2, 1⟩ = 5/6 := by sorry

end NUMINAMATH_CALUDE_prob_non_blue_specific_cube_l1267_126705


namespace NUMINAMATH_CALUDE_hollow_block_3_9_5_cubes_l1267_126744

/-- Calculates the number of unit cubes needed to construct the outer shell of a hollow rectangular block. -/
def hollow_block_cubes (length width depth : ℕ) : ℕ :=
  2 * (length * width) +  -- top and bottom
  2 * ((width * depth) - (width * 2)) +  -- longer sides
  2 * ((length * depth) - (length * 2) - 2)  -- shorter sides

/-- Theorem stating that a hollow rectangular block with dimensions 3 x 9 x 5 requires 122 unit cubes. -/
theorem hollow_block_3_9_5_cubes : 
  hollow_block_cubes 3 9 5 = 122 := by
  sorry

#eval hollow_block_cubes 3 9 5  -- Should output 122

end NUMINAMATH_CALUDE_hollow_block_3_9_5_cubes_l1267_126744


namespace NUMINAMATH_CALUDE_num_subcommittee_pairs_l1267_126710

/-- Represents the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The total number of committee members -/
def total_members : ℕ := 12

/-- The number of teachers in the committee -/
def teachers : ℕ := 5

/-- The size of each subcommittee -/
def subcommittee_size : ℕ := 4

/-- The number of subcommittees to form -/
def num_subcommittees : ℕ := 2

/-- Calculates the number of subcommittees with at least one teacher -/
def subcommittees_with_teacher (members teachers : ℕ) : ℕ :=
  choose members subcommittee_size - choose (members - teachers) subcommittee_size

/-- The main theorem stating the number of distinct pairs of subcommittees -/
theorem num_subcommittee_pairs : 
  subcommittees_with_teacher total_members teachers * 
  subcommittees_with_teacher (total_members - subcommittee_size) (teachers - 1) = 29900 := by
  sorry

end NUMINAMATH_CALUDE_num_subcommittee_pairs_l1267_126710


namespace NUMINAMATH_CALUDE_sum_of_powers_of_i_l1267_126737

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem sum_of_powers_of_i :
  i^255 + i^256 + i^257 + i^258 + i^259 = -i :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_of_i_l1267_126737


namespace NUMINAMATH_CALUDE_function_expression_l1267_126758

theorem function_expression (f : ℝ → ℝ) (h : ∀ x, f (x + 1) = x^2 - 1) :
  ∀ x, f x = x^2 - 2*x :=
by sorry

end NUMINAMATH_CALUDE_function_expression_l1267_126758


namespace NUMINAMATH_CALUDE_sum_of_seven_consecutive_odds_mod_16_l1267_126795

theorem sum_of_seven_consecutive_odds_mod_16 (n : ℕ) (h : n = 12001) :
  (List.sum (List.map (λ i => n + 2 * i) (List.range 7))) % 16 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_seven_consecutive_odds_mod_16_l1267_126795


namespace NUMINAMATH_CALUDE_managing_team_selection_l1267_126739

def society_size : ℕ := 20
def team_size : ℕ := 3

theorem managing_team_selection :
  Nat.choose society_size team_size = 1140 := by
  sorry

end NUMINAMATH_CALUDE_managing_team_selection_l1267_126739


namespace NUMINAMATH_CALUDE_nine_distinct_values_of_z_l1267_126792

/-- Given two integers x and y between 100 and 999 inclusive, where y is formed by swapping
    the hundreds and tens digits of x (units digit remains the same), prove that the absolute
    difference z = |x - y| can have exactly 9 distinct values. -/
theorem nine_distinct_values_of_z (x y : ℤ) (z : ℕ) :
  (100 ≤ x ∧ x ≤ 999) →
  (100 ≤ y ∧ y ≤ 999) →
  (∃ a b c : ℕ, x = 100 * a + 10 * b + c ∧ y = 10 * a + 100 * b + c ∧ 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9) →
  z = |x - y| →
  ∃ (S : Finset ℕ), S.card = 9 ∧ z ∈ S ∧ ∀ w ∈ S, ∃ k : ℕ, w = 90 * k ∧ k ≤ 8 :=
sorry

#check nine_distinct_values_of_z

end NUMINAMATH_CALUDE_nine_distinct_values_of_z_l1267_126792


namespace NUMINAMATH_CALUDE_sum_of_divisors_156_l1267_126703

/-- The sum of positive whole number divisors of a natural number n -/
def sum_of_divisors (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of positive whole number divisors of 156 is 392 -/
theorem sum_of_divisors_156 : sum_of_divisors 156 = 392 := by sorry

end NUMINAMATH_CALUDE_sum_of_divisors_156_l1267_126703


namespace NUMINAMATH_CALUDE_largest_digit_divisible_by_four_l1267_126780

theorem largest_digit_divisible_by_four :
  ∀ n : ℕ, 
    (n = 4969794) → 
    (∀ m : ℕ, m % 4 = 0 ↔ (m % 100) % 4 = 0) → 
    n % 4 = 0 :=
by sorry

end NUMINAMATH_CALUDE_largest_digit_divisible_by_four_l1267_126780


namespace NUMINAMATH_CALUDE_system_solutions_correct_l1267_126762

theorem system_solutions_correct : 
  -- System 1
  (∃ (x y : ℝ), x - 2*y = 1 ∧ 3*x + 2*y = 7 ∧ x = 2 ∧ y = 1/2) ∧
  -- System 2
  (∃ (x y : ℝ), x - y = 3 ∧ (x - y - 3)/2 - y/3 = -1 ∧ x = 6 ∧ y = 3) :=
by sorry

end NUMINAMATH_CALUDE_system_solutions_correct_l1267_126762


namespace NUMINAMATH_CALUDE_fraction_change_l1267_126756

theorem fraction_change (x : ℚ) : 
  (1.2 * (5 : ℚ)) / (7 * (1 - x / 100)) = 20 / 21 → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_fraction_change_l1267_126756


namespace NUMINAMATH_CALUDE_cos_270_degrees_l1267_126799

theorem cos_270_degrees : Real.cos (270 * π / 180) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_270_degrees_l1267_126799


namespace NUMINAMATH_CALUDE_fifteen_tomorrow_fishers_l1267_126723

/-- Represents the fishing schedule in the coastal village -/
structure FishingSchedule where
  daily : Nat
  everyOtherDay : Nat
  everyThreeDay : Nat
  yesterday : Nat
  today : Nat

/-- Calculates the number of people fishing tomorrow based on the given schedule -/
def tomorrowFishers (schedule : FishingSchedule) : Nat :=
  sorry

/-- Theorem stating that given the specific fishing schedule, 15 people will fish tomorrow -/
theorem fifteen_tomorrow_fishers (schedule : FishingSchedule) 
  (h1 : schedule.daily = 7)
  (h2 : schedule.everyOtherDay = 8)
  (h3 : schedule.everyThreeDay = 3)
  (h4 : schedule.yesterday = 12)
  (h5 : schedule.today = 10) :
  tomorrowFishers schedule = 15 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_tomorrow_fishers_l1267_126723


namespace NUMINAMATH_CALUDE_final_tree_count_l1267_126725

/-- The number of dogwood trees in the park after planting -/
def total_trees (initial : ℕ) (planted_today : ℕ) (planted_tomorrow : ℕ) : ℕ :=
  initial + planted_today + planted_tomorrow

/-- Theorem stating the final number of dogwood trees in the park -/
theorem final_tree_count :
  total_trees 7 5 4 = 16 := by
  sorry

end NUMINAMATH_CALUDE_final_tree_count_l1267_126725


namespace NUMINAMATH_CALUDE_max_notebooks_proof_l1267_126704

/-- The maximum number of notebooks that can be bought given the constraints -/
def max_notebooks : ℕ := 5

/-- The total budget in yuan -/
def total_budget : ℚ := 30

/-- The total number of books -/
def total_books : ℕ := 30

/-- The cost of each notebook in yuan -/
def notebook_cost : ℚ := 4

/-- The cost of each exercise book in yuan -/
def exercise_book_cost : ℚ := 0.4

theorem max_notebooks_proof :
  (∀ n : ℕ, n ≤ total_books →
    n * notebook_cost + (total_books - n) * exercise_book_cost ≤ total_budget) →
  (max_notebooks * notebook_cost + (total_books - max_notebooks) * exercise_book_cost ≤ total_budget) ∧
  (∀ m : ℕ, m > max_notebooks →
    m * notebook_cost + (total_books - m) * exercise_book_cost > total_budget) :=
by sorry

end NUMINAMATH_CALUDE_max_notebooks_proof_l1267_126704


namespace NUMINAMATH_CALUDE_triangle_equilateral_l1267_126766

/-- 
Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
if (a+b+c)(b+c-a) = 3bc and sin A = 2sin B cos C, 
then the triangle is equilateral with A = B = C = 60°
-/
theorem triangle_equilateral (a b c A B C : ℝ) : 
  (a + b + c) * (b + c - a) = 3 * b * c →
  Real.sin A = 2 * Real.sin B * Real.cos C →
  A = 60 * π / 180 ∧ B = 60 * π / 180 ∧ C = 60 * π / 180 := by
  sorry


end NUMINAMATH_CALUDE_triangle_equilateral_l1267_126766


namespace NUMINAMATH_CALUDE_at_least_one_alarm_probability_l1267_126779

theorem at_least_one_alarm_probability (p_A p_B : ℝ) 
  (h_A : 0 ≤ p_A ∧ p_A ≤ 1) 
  (h_B : 0 ≤ p_B ∧ p_B ≤ 1) : 
  1 - (1 - p_A) * (1 - p_B) = p_A + p_B - p_A * p_B :=
by sorry

end NUMINAMATH_CALUDE_at_least_one_alarm_probability_l1267_126779


namespace NUMINAMATH_CALUDE_donation_problem_l1267_126798

theorem donation_problem (first_total second_total : ℕ) 
  (donor_ratio : ℚ) (avg_diff : ℕ) :
  first_total = 60000 →
  second_total = 150000 →
  donor_ratio = 3/2 →
  avg_diff = 20 →
  ∃ (first_donors : ℕ),
    first_donors = 2000 ∧
    (donor_ratio * first_donors : ℚ) = 3000 ∧
    (second_total : ℚ) / (donor_ratio * first_donors) - 
    (first_total : ℚ) / first_donors = avg_diff :=
by sorry

end NUMINAMATH_CALUDE_donation_problem_l1267_126798


namespace NUMINAMATH_CALUDE_library_average_disk_space_per_hour_l1267_126765

/-- Represents a digital music library -/
structure MusicLibrary where
  days : ℕ
  diskSpace : ℕ

/-- Calculates the average disk space usage per hour for a given music library -/
def averageDiskSpacePerHour (library : MusicLibrary) : ℚ :=
  library.diskSpace / (library.days * 24)

/-- Theorem stating that for the given library, the average disk space per hour is 50 MB -/
theorem library_average_disk_space_per_hour :
  let library : MusicLibrary := { days := 15, diskSpace := 18000 }
  averageDiskSpacePerHour library = 50 := by
  sorry

end NUMINAMATH_CALUDE_library_average_disk_space_per_hour_l1267_126765


namespace NUMINAMATH_CALUDE_stocker_wait_time_l1267_126700

def total_shopping_time : ℕ := 90
def shopping_time : ℕ := 42
def cart_wait_time : ℕ := 3
def employee_wait_time : ℕ := 13
def checkout_wait_time : ℕ := 18

theorem stocker_wait_time :
  total_shopping_time - shopping_time - (cart_wait_time + employee_wait_time + checkout_wait_time) = 14 := by
  sorry

end NUMINAMATH_CALUDE_stocker_wait_time_l1267_126700


namespace NUMINAMATH_CALUDE_molecular_weight_C7H6O2_correct_l1267_126728

/-- The molecular weight of C7H6O2 -/
def molecular_weight_C7H6O2 : ℝ := 122

/-- The number of moles in the given sample -/
def num_moles : ℝ := 9

/-- The total molecular weight of the given sample -/
def total_molecular_weight : ℝ := 1098

/-- Theorem stating that the molecular weight of C7H6O2 is correct -/
theorem molecular_weight_C7H6O2_correct :
  molecular_weight_C7H6O2 = total_molecular_weight / num_moles :=
by sorry

end NUMINAMATH_CALUDE_molecular_weight_C7H6O2_correct_l1267_126728


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1267_126740

theorem quadratic_equation_roots (a b c : ℝ) (h1 : a = 1) (h2 : b = 2023) (h3 : c = 2035) :
  ∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1267_126740


namespace NUMINAMATH_CALUDE_goods_train_speed_l1267_126764

theorem goods_train_speed 
  (man_train_speed : ℝ) 
  (goods_train_length : ℝ) 
  (passing_time : ℝ) 
  (h1 : man_train_speed = 70) 
  (h2 : goods_train_length = 0.28) 
  (h3 : passing_time = 9 / 3600) : 
  ∃ (goods_train_speed : ℝ), 
    goods_train_speed = 42 ∧ 
    (goods_train_speed + man_train_speed) * passing_time = goods_train_length :=
by sorry

end NUMINAMATH_CALUDE_goods_train_speed_l1267_126764


namespace NUMINAMATH_CALUDE_new_machine_rate_proof_l1267_126785

/-- The rate of the old machine in bolts per hour -/
def old_machine_rate : ℝ := 100

/-- The time both machines work together in minutes -/
def work_time : ℝ := 84

/-- The total number of bolts produced by both machines -/
def total_bolts : ℝ := 350

/-- The rate of the new machine in bolts per hour -/
def new_machine_rate : ℝ := 150

theorem new_machine_rate_proof :
  (old_machine_rate * work_time / 60 + new_machine_rate * work_time / 60) = total_bolts :=
by sorry

end NUMINAMATH_CALUDE_new_machine_rate_proof_l1267_126785


namespace NUMINAMATH_CALUDE_correct_num_episodes_l1267_126735

/-- The number of episodes in a TV mini series -/
def num_episodes : ℕ := 6

/-- The length of each episode in minutes -/
def episode_length : ℕ := 50

/-- The total watching time in hours -/
def total_watching_time : ℕ := 5

/-- Theorem stating that the number of episodes is correct -/
theorem correct_num_episodes :
  num_episodes * episode_length = total_watching_time * 60 := by
  sorry

end NUMINAMATH_CALUDE_correct_num_episodes_l1267_126735


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1267_126719

-- Define sets A and B
def A : Set ℝ := {x : ℝ | -1 < x ∧ x ≤ 4}
def B : Set ℝ := {x : ℝ | 2 < x ∧ x ≤ 5}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 2 < x ∧ x ≤ 4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1267_126719


namespace NUMINAMATH_CALUDE_lunch_cakes_count_l1267_126708

def total_cakes : ℕ := 15
def dinner_cakes : ℕ := 9

theorem lunch_cakes_count : total_cakes - dinner_cakes = 6 := by
  sorry

end NUMINAMATH_CALUDE_lunch_cakes_count_l1267_126708


namespace NUMINAMATH_CALUDE_problem_solution_l1267_126712

def solution : Set (ℤ × ℤ × ℤ × ℤ × ℤ × ℤ) :=
  {(3, 2, 1, 3, 2, 1), (6, 1, 1, 2, 2, 2), (7, 1, 1, 3, 3, 1), (8, 1, 1, 5, 2, 1),
   (2, 2, 2, 6, 1, 1), (3, 3, 1, 7, 1, 1), (5, 2, 1, 8, 1, 1)}

def satisfies_conditions (t : ℤ × ℤ × ℤ × ℤ × ℤ × ℤ) : Prop :=
  let (a, b, c, x, y, z) := t
  a + b + c = x * y * z ∧
  x + y + z = a * b * c ∧
  a ≥ b ∧ b ≥ c ∧ c ≥ 1 ∧
  x ≥ y ∧ y ≥ z ∧ z ≥ 1

theorem problem_solution :
  ∀ t : ℤ × ℤ × ℤ × ℤ × ℤ × ℤ, satisfies_conditions t ↔ t ∈ solution := by
  sorry

#check problem_solution

end NUMINAMATH_CALUDE_problem_solution_l1267_126712


namespace NUMINAMATH_CALUDE_sum_from_difference_and_squares_l1267_126771

theorem sum_from_difference_and_squares (x y : ℝ) 
  (h1 : x^2 - y^2 = 21) 
  (h2 : x - y = 3) : 
  x + y = 7 := by
sorry

end NUMINAMATH_CALUDE_sum_from_difference_and_squares_l1267_126771


namespace NUMINAMATH_CALUDE_min_unhappiness_theorem_l1267_126750

/-- Represents the unhappiness levels of students -/
def unhappiness_levels : List ℝ := List.range 2017

/-- The number of groups to split the students into -/
def num_groups : ℕ := 15

/-- Calculates the minimum possible sum of average unhappiness levels -/
def min_unhappiness (levels : List ℝ) (groups : ℕ) : ℝ :=
  sorry

/-- The theorem stating the minimum unhappiness of the class -/
theorem min_unhappiness_theorem :
  min_unhappiness unhappiness_levels num_groups = 1120.5 := by
  sorry

end NUMINAMATH_CALUDE_min_unhappiness_theorem_l1267_126750


namespace NUMINAMATH_CALUDE_treehouse_planks_l1267_126738

/-- The total number of planks Charlie and his father have -/
def total_planks (initial_planks charlie_planks father_planks : ℕ) : ℕ :=
  initial_planks + charlie_planks + father_planks

/-- Theorem stating that the total number of planks is 35 -/
theorem treehouse_planks : total_planks 15 10 10 = 35 := by
  sorry

end NUMINAMATH_CALUDE_treehouse_planks_l1267_126738


namespace NUMINAMATH_CALUDE_shortest_altitude_right_triangle_l1267_126776

theorem shortest_altitude_right_triangle (a b c h : ℝ) : 
  a = 8 ∧ b = 15 ∧ c = 17 →
  a^2 + b^2 = c^2 →
  h * c = 2 * (1/2 * a * b) →
  h = 120/17 := by
sorry

end NUMINAMATH_CALUDE_shortest_altitude_right_triangle_l1267_126776


namespace NUMINAMATH_CALUDE_quadratic_max_value_l1267_126768

/-- Given a quadratic function y = x² + 2x - 2, prove that if the maximum value of y is 1
    when a ≤ x ≤ 1/2, then a = -3. -/
theorem quadratic_max_value (y : ℝ → ℝ) (a : ℝ) :
  (∀ x, y x = x^2 + 2*x - 2) →
  (∀ x, a ≤ x → x ≤ 1/2 → y x ≤ 1) →
  (∃ x, a ≤ x ∧ x ≤ 1/2 ∧ y x = 1) →
  a = -3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_max_value_l1267_126768


namespace NUMINAMATH_CALUDE_square_area_in_triangle_l1267_126752

/-- A triangle in a 2D plane -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- A square in a 2D plane -/
structure Square where
  corners : Fin 4 → ℝ × ℝ

/-- The area of a triangle -/
def Triangle.area (t : Triangle) : ℝ := sorry

/-- The area of a square -/
def Square.area (s : Square) : ℝ := sorry

/-- Predicate to check if a square lies within a triangle -/
def Square.liesWithin (s : Square) (t : Triangle) : Prop := sorry

/-- Theorem: The area of any square lying within a triangle does not exceed half of the area of that triangle -/
theorem square_area_in_triangle (t : Triangle) (s : Square) :
  s.liesWithin t → s.area ≤ (1/2) * t.area := by sorry

end NUMINAMATH_CALUDE_square_area_in_triangle_l1267_126752


namespace NUMINAMATH_CALUDE_B_subset_A_l1267_126791

-- Define the sets A and B
def A : Set (ℝ × ℝ) := {p | p.2 = p.1}
def B : Set (ℝ × ℝ) := {p | 2 * p.1 - p.2 = 2 ∧ p.1 + 2 * p.2 = 6}

-- Theorem statement
theorem B_subset_A : B ⊆ A := by
  sorry

end NUMINAMATH_CALUDE_B_subset_A_l1267_126791


namespace NUMINAMATH_CALUDE_game_probability_limit_l1267_126794

/-- Represents the state of money distribution among players -/
inductive GameState
  | AllOne
  | TwoOneZero

/-- Transition probability matrix for the game -/
def transitionMatrix : Matrix GameState GameState ℝ := sorry

/-- The probability of all players having $1 after n bell rings -/
def prob_all_one (n : ℕ) : ℝ := sorry

/-- The limit of prob_all_one as n approaches infinity -/
def limit_prob_all_one : ℝ := sorry

theorem game_probability_limit :
  limit_prob_all_one = 1/4 := by sorry

end NUMINAMATH_CALUDE_game_probability_limit_l1267_126794


namespace NUMINAMATH_CALUDE_carol_wins_probability_l1267_126796

-- Define the probability of rolling an eight on an 8-sided die
def prob_eight : ℚ := 1 / 8

-- Define the probability of not rolling an eight
def prob_not_eight : ℚ := 1 - prob_eight

-- Define the probability of Carol winning in the first cycle
def prob_carol_first_cycle : ℚ := prob_not_eight * prob_not_eight * prob_eight

-- Define the probability of no one winning in a single cycle
def prob_no_winner_cycle : ℚ := prob_not_eight * prob_not_eight * prob_not_eight

-- State the theorem
theorem carol_wins_probability :
  (prob_carol_first_cycle / (1 - prob_no_winner_cycle)) = 49 / 169 := by
  sorry

end NUMINAMATH_CALUDE_carol_wins_probability_l1267_126796


namespace NUMINAMATH_CALUDE_profit_thirty_for_thirtyfive_l1267_126767

/-- Calculates the profit percentage when selling a different number of articles than the cost price basis -/
def profit_percentage (sold : ℕ) (cost_basis : ℕ) : ℚ :=
  let profit := cost_basis - sold
  (profit / sold) * 100

/-- Theorem stating that selling 30 articles at the price of 35 articles' cost results in a profit of 1/6 * 100% -/
theorem profit_thirty_for_thirtyfive :
  profit_percentage 30 35 = 100 / 6 := by
  sorry

end NUMINAMATH_CALUDE_profit_thirty_for_thirtyfive_l1267_126767


namespace NUMINAMATH_CALUDE_fraction_subtraction_l1267_126729

theorem fraction_subtraction (d : ℝ) : (5 + 4 * d) / 9 - 3 = (4 * d - 22) / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l1267_126729


namespace NUMINAMATH_CALUDE_mustard_bottles_sum_l1267_126731

theorem mustard_bottles_sum : 0.25 + 0.25 + 0.38 = 0.88 := by
  sorry

end NUMINAMATH_CALUDE_mustard_bottles_sum_l1267_126731


namespace NUMINAMATH_CALUDE_not_all_distinct_l1267_126721

/-- A sequence of non-negative rational numbers satisfying a(m) + a(n) = a(mn) -/
def NonNegativeSequence (a : ℕ → ℚ) : Prop :=
  (∀ n, a n ≥ 0) ∧ (∀ m n, a m + a n = a (m * n))

/-- The theorem stating that not all elements of the sequence can be distinct -/
theorem not_all_distinct (a : ℕ → ℚ) (h : NonNegativeSequence a) :
  ∃ i j, i ≠ j ∧ a i = a j :=
sorry

end NUMINAMATH_CALUDE_not_all_distinct_l1267_126721


namespace NUMINAMATH_CALUDE_pets_lost_l1267_126747

/-- Proves the number of pets Anthony lost when he forgot to lock the door -/
theorem pets_lost (initial_pets : ℕ) (final_pets : ℕ) : 
  initial_pets = 16 → 
  final_pets = 8 → 
  (initial_pets - (initial_pets - (initial_pets - final_pets) * 4 / 5)) = final_pets →
  initial_pets - (initial_pets - final_pets) * 5 / 4 = 6 :=
by
  sorry

#check pets_lost

end NUMINAMATH_CALUDE_pets_lost_l1267_126747


namespace NUMINAMATH_CALUDE_cube_split_and_stack_l1267_126786

/-- The number of millimeters in a meter -/
def mm_per_m : ℕ := 1000

/-- The number of meters in a kilometer -/
def m_per_km : ℕ := 1000

/-- The edge length of the original cube in meters -/
def cube_edge_m : ℕ := 1

/-- The edge length of small cubes in millimeters -/
def small_cube_edge_mm : ℕ := 1

/-- The height of the column in kilometers -/
def column_height_km : ℕ := 1000

theorem cube_split_and_stack :
  (cube_edge_m * mm_per_m)^3 / small_cube_edge_mm = column_height_km * m_per_km * mm_per_m :=
sorry

end NUMINAMATH_CALUDE_cube_split_and_stack_l1267_126786


namespace NUMINAMATH_CALUDE_f_is_odd_g_is_even_l1267_126773

-- Define the functions
def f (x : ℝ) : ℝ := x + x^3 + x^5
def g (x : ℝ) : ℝ := x^2 + 1

-- Define odd and even functions
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Theorem statements
theorem f_is_odd : IsOdd f := by sorry

theorem g_is_even : IsEven g := by sorry

end NUMINAMATH_CALUDE_f_is_odd_g_is_even_l1267_126773


namespace NUMINAMATH_CALUDE_classify_books_count_l1267_126774

/-- The number of ways to classify 6 distinct books into two groups -/
def classify_books : ℕ :=
  let total_books : ℕ := 6
  let intersection_size : ℕ := 3
  let remaining_books : ℕ := total_books - intersection_size
  let ways_to_choose_intersection : ℕ := Nat.choose total_books intersection_size
  let ways_to_distribute_remaining : ℕ := 3^remaining_books
  (ways_to_choose_intersection * ways_to_distribute_remaining) / 2

/-- Theorem stating that the number of ways to classify the books is 270 -/
theorem classify_books_count : classify_books = 270 := by
  sorry

end NUMINAMATH_CALUDE_classify_books_count_l1267_126774


namespace NUMINAMATH_CALUDE_inequality_proof_l1267_126743

theorem inequality_proof (x y z u : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (hu : u > 0)
  (hx1 : x ≠ 1) (hy1 : y ≠ 1) (hz1 : z ≠ 1) (hu1 : u ≠ 1)
  (h_all_gt : (x > 1 ∧ y > 1 ∧ z > 1 ∧ u > 1) ∨ (x < 1 ∧ y < 1 ∧ z < 1 ∧ u < 1)) :
  (Real.log x ^ 3 / Real.log y) / (x + y + z) +
  (Real.log y ^ 3 / Real.log z) / (y + z + u) +
  (Real.log z ^ 3 / Real.log u) / (z + u + x) +
  (Real.log u ^ 3 / Real.log x) / (u + x + y) ≥
  16 / (x + y + z + u) :=
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1267_126743


namespace NUMINAMATH_CALUDE_total_cost_proof_l1267_126759

/-- The cost of a single ticket in dollars -/
def ticket_cost : ℝ := 44

/-- The number of tickets purchased -/
def num_tickets : ℕ := 7

/-- The total cost of tickets in dollars -/
def total_cost : ℝ := ticket_cost * num_tickets

theorem total_cost_proof : total_cost = 308 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_proof_l1267_126759


namespace NUMINAMATH_CALUDE_like_terms_exponent_equality_l1267_126777

theorem like_terms_exponent_equality (a b : ℝ) (m : ℝ) : 
  (∃ k : ℝ, -2 * a^(2-m) * b^3 = k * (-2 * a^(4-3*m) * b^3)) → m = 1 := by
sorry

end NUMINAMATH_CALUDE_like_terms_exponent_equality_l1267_126777


namespace NUMINAMATH_CALUDE_calvin_winning_condition_l1267_126778

/-- The game state represents the current configuration of coins on the circle. -/
structure GameState where
  n : ℕ
  coins : Fin (2 * n + 1) → Bool

/-- A player's move in the game. -/
inductive Move
  | calvin : Fin (2 * n + 1) → Move
  | hobbes : Option (Fin (2 * n + 1)) → Move

/-- Applies a move to the current game state. -/
def applyMove (state : GameState) (move : Move) : GameState :=
  sorry

/-- Counts the number of tails in the current game state. -/
def countTails (state : GameState) : ℕ :=
  sorry

/-- Determines if a player has a winning strategy for the game. -/
def hasWinningStrategy (n k : ℕ) (player : Bool) : Prop :=
  sorry

/-- The main theorem stating the conditions for Calvin's victory. -/
theorem calvin_winning_condition (n k : ℕ) (h1 : n > 1) (h2 : k ≥ 1) :
  hasWinningStrategy n k true ↔ k ≤ n + 1 :=
  sorry

end NUMINAMATH_CALUDE_calvin_winning_condition_l1267_126778


namespace NUMINAMATH_CALUDE_min_value_quadratic_l1267_126727

theorem min_value_quadratic (x : ℝ) : 
  (∀ x, x^2 - 4*x + 5 ≥ 1) ∧ (∃ x, x^2 - 4*x + 5 = 1) :=
sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l1267_126727


namespace NUMINAMATH_CALUDE_pencil_distribution_l1267_126717

/- Given conditions -/
def total_pencils : ℕ := 10
def num_friends : ℕ := 4

/- Function to calculate binomial coefficient -/
def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/- Theorem statement -/
theorem pencil_distribution :
  binomial (total_pencils - num_friends + num_friends - 1) (num_friends - 1) = 84 :=
by sorry

end NUMINAMATH_CALUDE_pencil_distribution_l1267_126717


namespace NUMINAMATH_CALUDE_amount_distribution_l1267_126711

theorem amount_distribution (A : ℝ) : 
  (A / 14 = A / 18 + 80) → A = 5040 := by
  sorry

end NUMINAMATH_CALUDE_amount_distribution_l1267_126711


namespace NUMINAMATH_CALUDE_room_entry_exit_ways_l1267_126753

/-- The number of doors in the room -/
def num_doors : ℕ := 4

/-- The number of times the person enters the room -/
def num_entries : ℕ := 1

/-- The number of times the person exits the room -/
def num_exits : ℕ := 1

/-- The total number of ways to enter and exit the room -/
def total_ways : ℕ := num_doors ^ (num_entries + num_exits)

theorem room_entry_exit_ways :
  total_ways = 16 := by sorry

end NUMINAMATH_CALUDE_room_entry_exit_ways_l1267_126753


namespace NUMINAMATH_CALUDE_work_completion_equality_prove_new_group_size_l1267_126761

/-- The number of persons in the original group -/
def original_group : ℕ := 15

/-- The number of days the original group takes to complete the work -/
def original_days : ℕ := 18

/-- The fraction of work done by the new group -/
def new_group_work_fraction : ℚ := 1/3

/-- The number of days the new group takes to complete their fraction of work -/
def new_group_days : ℕ := 21

/-- The multiplier for the number of persons in the new group -/
def new_group_multiplier : ℚ := 5/2

/-- The number of persons in the new group -/
def new_group_size : ℕ := 7

theorem work_completion_equality :
  (original_group : ℚ) / original_days = 
  (new_group_multiplier * new_group_size) * new_group_work_fraction / new_group_days :=
by sorry

/-- The main theorem proving the size of the new group -/
theorem prove_new_group_size : 
  ∃ (n : ℕ), n = new_group_size ∧
  (original_group : ℚ) / original_days = 
  (new_group_multiplier * n) * new_group_work_fraction / new_group_days :=
by sorry

end NUMINAMATH_CALUDE_work_completion_equality_prove_new_group_size_l1267_126761


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l1267_126748

theorem cubic_equation_solution : 
  ∀ x y : ℕ, x^3 - y^3 = x * y + 61 → x = 6 ∧ y = 5 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l1267_126748


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1267_126789

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (a > b + 1 → a > b) ∧ ¬(a > b → a > b + 1) := by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1267_126789


namespace NUMINAMATH_CALUDE_complex_power_magnitude_l1267_126722

theorem complex_power_magnitude : 
  Complex.abs ((2/3 : ℂ) + (5/6 : ℂ) * Complex.I)^8 = 2825761/1679616 := by
sorry

end NUMINAMATH_CALUDE_complex_power_magnitude_l1267_126722


namespace NUMINAMATH_CALUDE_apollonius_circle_l1267_126760

/-- The Apollonius Circle Theorem -/
theorem apollonius_circle (x y : ℝ) : 
  let A : ℝ × ℝ := (2, 0)
  let B : ℝ × ℝ := (8, 0)
  let P : ℝ × ℝ := (x, y)
  let dist (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  dist P A / dist P B = 1/2 → x^2 + y^2 = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_apollonius_circle_l1267_126760


namespace NUMINAMATH_CALUDE_triangle_circumradius_l1267_126775

theorem triangle_circumradius (a b c : ℝ) (h1 : a = 8) (h2 : b = 15) (h3 : c = 17) :
  let s := (a + b + c) / 2
  (a * b * c) / (4 * Real.sqrt (s * (s - a) * (s - b) * (s - c))) = 8.5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_circumradius_l1267_126775


namespace NUMINAMATH_CALUDE_triangle_sine_product_l1267_126724

theorem triangle_sine_product (A B C : Real) (s₁ s₂ s₃ : Real) : 
  (Real.sin A + Real.sin B + Real.sin C) / (Real.cos A + Real.cos B + Real.cos C) = 12/7 →
  Real.sin A * Real.sin B * Real.sin C = 12/25 →
  (Real.sin C = s₁ ∨ Real.sin C = s₂ ∨ Real.sin C = s₃) →
  100 * s₁ * s₂ * s₃ = 48 := by
sorry

end NUMINAMATH_CALUDE_triangle_sine_product_l1267_126724


namespace NUMINAMATH_CALUDE_lines_cannot_form_triangle_implies_a_equals_neg_one_l1267_126755

/-- Three lines in a 2D plane --/
structure ThreeLines where
  line1 : (ℝ × ℝ) → Prop
  line2 : (ℝ × ℝ) → Prop
  line3 : (ℝ × ℝ) → Prop

/-- Definition of the specific lines in the problem --/
def problemLines (a : ℝ) : ThreeLines :=
  { line1 := λ (x, y) => x + y = 1
  , line2 := λ (x, y) => x - y = 1
  , line3 := λ (x, y) => a * x + y = 1 }

/-- Definition of when three lines cannot form a triangle --/
def cannotFormTriangle (lines : ThreeLines) : Prop :=
  ∃ (p : ℝ × ℝ), (lines.line1 p ∧ lines.line2 p) ∨
                 (lines.line1 p ∧ lines.line3 p) ∨
                 (lines.line2 p ∧ lines.line3 p)

/-- The main theorem --/
theorem lines_cannot_form_triangle_implies_a_equals_neg_one :
  ∀ a : ℝ, cannotFormTriangle (problemLines a) → a = -1 :=
by sorry

end NUMINAMATH_CALUDE_lines_cannot_form_triangle_implies_a_equals_neg_one_l1267_126755


namespace NUMINAMATH_CALUDE_matrix_multiplication_result_l1267_126718

/-- Given real numbers d, e, and f, prove that the matrix multiplication of 
    A = [[0, d, -e], [-d, 0, f], [e, -f, 0]] and 
    B = [[f^2, fd, fe], [fd, d^2, de], [fe, de, e^2]] 
    results in 
    C = [[d^2 - e^2, 2fd, 0], [0, f^2 - d^2, de-fe], [0, e^2 - d^2, fe-df]] -/
theorem matrix_multiplication_result (d e f : ℝ) : 
  let A : Matrix (Fin 3) (Fin 3) ℝ := !![0, d, -e; -d, 0, f; e, -f, 0]
  let B : Matrix (Fin 3) (Fin 3) ℝ := !![f^2, f*d, f*e; f*d, d^2, d*e; f*e, d*e, e^2]
  let C : Matrix (Fin 3) (Fin 3) ℝ := !![d^2 - e^2, 2*f*d, 0; 0, f^2 - d^2, d*e - f*e; 0, e^2 - d^2, f*e - d*f]
  A * B = C := by sorry

end NUMINAMATH_CALUDE_matrix_multiplication_result_l1267_126718


namespace NUMINAMATH_CALUDE_gunny_bag_capacity_is_13_tons_l1267_126769

/-- Represents the weight of a packet in pounds -/
def packet_weight : ℚ := 16 + 4 / 16

/-- Represents the number of packets -/
def num_packets : ℕ := 1680

/-- Represents the number of pounds in a ton -/
def pounds_per_ton : ℕ := 2100

/-- Represents the capacity of the gunny bag in tons -/
def gunny_bag_capacity : ℚ := (num_packets * packet_weight) / pounds_per_ton

theorem gunny_bag_capacity_is_13_tons : gunny_bag_capacity = 13 := by
  sorry

end NUMINAMATH_CALUDE_gunny_bag_capacity_is_13_tons_l1267_126769


namespace NUMINAMATH_CALUDE_power_inequality_l1267_126782

theorem power_inequality : (1.7 : ℝ) ^ (0.3 : ℝ) > (0.9 : ℝ) ^ (0.3 : ℝ) := by sorry

end NUMINAMATH_CALUDE_power_inequality_l1267_126782


namespace NUMINAMATH_CALUDE_expression_factorization_l1267_126715

theorem expression_factorization (x : ℝ) : 
  (16 * x^6 + 49 * x^4 - 9) - (4 * x^6 - 14 * x^4 - 9) = 3 * x^4 * (4 * x^2 + 21) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l1267_126715


namespace NUMINAMATH_CALUDE_cos_180_degrees_l1267_126726

theorem cos_180_degrees : Real.cos (π) = -1 := by
  sorry

end NUMINAMATH_CALUDE_cos_180_degrees_l1267_126726


namespace NUMINAMATH_CALUDE_tens_place_of_first_ten_digit_number_l1267_126706

/-- Represents the sequence of grouped numbers -/
def groupedSequence : List (List Nat) := sorry

/-- The number of digits in the nth group -/
def groupDigits (n : Nat) : Nat := n

/-- The sum of digits in the first n groups -/
def sumDigitsUpTo (n : Nat) : Nat := sorry

/-- The first ten-digit number in the sequence -/
def firstTenDigitNumber : Nat := sorry

/-- Theorem: The tens place digit of the first ten-digit number is 2 -/
theorem tens_place_of_first_ten_digit_number :
  (firstTenDigitNumber / 1000000000) % 10 = 2 := by sorry

end NUMINAMATH_CALUDE_tens_place_of_first_ten_digit_number_l1267_126706


namespace NUMINAMATH_CALUDE_puppy_kibble_percentage_proof_l1267_126784

/-- The percentage of vets recommending Puppy Kibble -/
def puppy_kibble_percentage : ℝ := 20

/-- The percentage of vets recommending Yummy Dog Kibble -/
def yummy_kibble_percentage : ℝ := 30

/-- The total number of vets in the state -/
def total_vets : ℕ := 1000

/-- The difference in number of vets recommending Yummy Dog Kibble vs Puppy Kibble -/
def vet_difference : ℕ := 100

theorem puppy_kibble_percentage_proof :
  puppy_kibble_percentage = 20 ∧
  yummy_kibble_percentage = 30 ∧
  total_vets = 1000 ∧
  vet_difference = 100 →
  puppy_kibble_percentage * (total_vets : ℝ) / 100 + vet_difference = 
  yummy_kibble_percentage * (total_vets : ℝ) / 100 :=
by sorry

end NUMINAMATH_CALUDE_puppy_kibble_percentage_proof_l1267_126784


namespace NUMINAMATH_CALUDE_competition_finish_orders_l1267_126787

theorem competition_finish_orders (n : ℕ) (h : n = 5) : 
  Nat.factorial n = 120 := by
  sorry

end NUMINAMATH_CALUDE_competition_finish_orders_l1267_126787


namespace NUMINAMATH_CALUDE_investment_interest_rate_l1267_126751

theorem investment_interest_rate 
  (initial_investment : ℝ)
  (first_duration : ℝ)
  (first_rate : ℝ)
  (second_duration : ℝ)
  (final_value : ℝ)
  (h1 : initial_investment = 15000)
  (h2 : first_duration = 9/12)
  (h3 : first_rate = 0.09)
  (h4 : second_duration = 9/12)
  (h5 : final_value = 17218.50) :
  ∃ s : ℝ, 
    s = 0.10 ∧ 
    final_value = initial_investment * (1 + first_duration * first_rate) * (1 + second_duration * s) := by
  sorry

end NUMINAMATH_CALUDE_investment_interest_rate_l1267_126751


namespace NUMINAMATH_CALUDE_expression_evaluation_l1267_126781

theorem expression_evaluation :
  let f : ℝ → ℝ := λ x => (x^2 - 2*x - 8) / (x - 4)
  f 5 = 7 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1267_126781
