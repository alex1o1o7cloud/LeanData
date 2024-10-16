import Mathlib

namespace NUMINAMATH_CALUDE_perpendicular_planes_from_perpendicular_lines_l937_93759

-- Define the types for our geometric objects
variable (Point : Type) [NormedAddCommGroup Point] [InnerProductSpace ℝ Point]
variable (Line : Type) (Plane : Type)

-- Define the relationships between geometric objects
variable (belongs_to : Point → Line → Prop)
variable (subset_of : Line → Plane → Prop)
variable (intersect_along : Plane → Plane → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- Define our specific objects
variable (α β : Plane) (l a b : Line)

-- State the theorem
theorem perpendicular_planes_from_perpendicular_lines 
  (h1 : intersect_along α β l)
  (h2 : subset_of a α)
  (h3 : subset_of b β)
  (h4 : perpendicular a l)
  (h5 : perpendicular b l) :
  ¬(plane_perpendicular α β) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_planes_from_perpendicular_lines_l937_93759


namespace NUMINAMATH_CALUDE_hyperbola_equation_l937_93738

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃ (c : ℝ), c^2 = 5 * a^2 ∧ 
   ∃ (S : ℝ), S = 20 ∧ S = (1/2) * c * (4 * c)) →
  a^2 = 2 ∧ b^2 = 8 := by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l937_93738


namespace NUMINAMATH_CALUDE_x_equals_160_l937_93747

/-- Given a relationship between x, y, and z, prove that x equals 160 when y is 16 and z is 7. -/
theorem x_equals_160 (k : ℝ) (x y z : ℝ → ℝ) :
  (∀ t, x t = k * y t / (z t)^2) →  -- Relationship between x, y, and z
  (x 0 = 10 ∧ y 0 = 4 ∧ z 0 = 14) →  -- Initial condition
  (y 1 = 16 ∧ z 1 = 7) →  -- New condition
  x 1 = 160 := by
sorry

end NUMINAMATH_CALUDE_x_equals_160_l937_93747


namespace NUMINAMATH_CALUDE_final_clothing_count_l937_93712

/-- Calculate the remaining clothes after donations and purchases -/
def remaining_clothes (initial : ℕ) : ℕ :=
  let after_orphanages := initial - (initial / 10 + 3 * (initial / 10))
  let after_shelter := after_orphanages - (after_orphanages / 5)
  let after_purchase := after_shelter + (after_shelter / 5)
  after_purchase - (after_purchase / 8)

/-- Theorem stating the final number of clothing pieces -/
theorem final_clothing_count :
  remaining_clothes 500 = 252 := by
  sorry

end NUMINAMATH_CALUDE_final_clothing_count_l937_93712


namespace NUMINAMATH_CALUDE_equilateral_cone_central_angle_l937_93773

/-- Represents a cone with an equilateral triangle as its axial section -/
structure EquilateralCone where
  /-- The radius of the base of the cone -/
  r : ℝ
  /-- The slant height of the cone, which is twice the radius for an equilateral axial section -/
  slant_height : ℝ
  /-- Condition that the slant height is twice the radius -/
  slant_height_eq : slant_height = 2 * r

/-- The central angle of the side surface development of an equilateral cone is π radians (180°) -/
theorem equilateral_cone_central_angle (cone : EquilateralCone) :
  Real.pi = (2 * Real.pi * cone.r) / cone.slant_height :=
by sorry

end NUMINAMATH_CALUDE_equilateral_cone_central_angle_l937_93773


namespace NUMINAMATH_CALUDE_custom_op_theorem_l937_93764

-- Define the custom operation x
def customOp (M N : Set ℕ) : Set ℕ := {x | x ∈ M ∨ x ∈ N ∧ x ∉ M ∩ N}

-- Define sets M and N
def M : Set ℕ := {0, 2, 4, 6, 8, 10}
def N : Set ℕ := {0, 3, 6, 9, 12, 15}

-- Theorem statement
theorem custom_op_theorem : customOp (customOp M N) M = N := by
  sorry

end NUMINAMATH_CALUDE_custom_op_theorem_l937_93764


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_two_l937_93770

theorem fraction_zero_implies_x_negative_two (x : ℝ) :
  (abs x - 2) / (2 - x) = 0 → x = -2 :=
sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_two_l937_93770


namespace NUMINAMATH_CALUDE_sin_tan_40_deg_l937_93794

theorem sin_tan_40_deg : 4 * Real.sin (40 * π / 180) - Real.tan (40 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sin_tan_40_deg_l937_93794


namespace NUMINAMATH_CALUDE_f_odd_and_decreasing_l937_93735

def f (x : ℝ) : ℝ := -x^3 - x

theorem f_odd_and_decreasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, x < y → f x > f y) := by
  sorry

end NUMINAMATH_CALUDE_f_odd_and_decreasing_l937_93735


namespace NUMINAMATH_CALUDE_carwash_donation_percentage_l937_93782

/-- Proves that the percentage of carwash proceeds donated is 90%, given the conditions of Hank's fundraising activities. -/
theorem carwash_donation_percentage
  (carwash_amount : ℝ)
  (bake_sale_amount : ℝ)
  (lawn_mowing_amount : ℝ)
  (bake_sale_donation_percentage : ℝ)
  (lawn_mowing_donation_percentage : ℝ)
  (total_donation : ℝ)
  (h1 : carwash_amount = 100)
  (h2 : bake_sale_amount = 80)
  (h3 : lawn_mowing_amount = 50)
  (h4 : bake_sale_donation_percentage = 0.75)
  (h5 : lawn_mowing_donation_percentage = 1)
  (h6 : total_donation = 200)
  (h7 : total_donation = carwash_amount * x + bake_sale_amount * bake_sale_donation_percentage + lawn_mowing_amount * lawn_mowing_donation_percentage)
  : x = 0.9 := by
  sorry

#check carwash_donation_percentage

end NUMINAMATH_CALUDE_carwash_donation_percentage_l937_93782


namespace NUMINAMATH_CALUDE_trig_inequality_implies_range_l937_93707

open Real

theorem trig_inequality_implies_range (θ : ℝ) :
  θ ∈ Set.Icc 0 (2 * π) →
  (cos θ)^5 - (sin θ)^5 < 7 * ((sin θ)^3 - (cos θ)^3) →
  θ ∈ Set.Ioo (π / 4) (5 * π / 4) :=
by
  sorry

end NUMINAMATH_CALUDE_trig_inequality_implies_range_l937_93707


namespace NUMINAMATH_CALUDE_total_crayons_l937_93752

/-- Theorem: The total number of crayons after adding more is the sum of the initial number and the added number. -/
theorem total_crayons (initial : ℕ) (added : ℕ) : 
  initial + added = initial + added :=
by sorry

end NUMINAMATH_CALUDE_total_crayons_l937_93752


namespace NUMINAMATH_CALUDE_slower_train_speed_l937_93740

theorem slower_train_speed
  (train_length : ℝ)
  (faster_train_speed : ℝ)
  (passing_time : ℝ)
  (h1 : train_length = 250)
  (h2 : faster_train_speed = 45)
  (h3 : passing_time = 23.998080153587715) :
  ∃ (slower_train_speed : ℝ),
    slower_train_speed = 30 ∧
    slower_train_speed + faster_train_speed = (2 * train_length / 1000) / (passing_time / 3600) :=
by sorry

end NUMINAMATH_CALUDE_slower_train_speed_l937_93740


namespace NUMINAMATH_CALUDE_power_sum_equals_six_l937_93718

theorem power_sum_equals_six (a x : ℝ) (h : a^x - a^(-x) = 2) : 
  a^(2*x) + a^(-2*x) = 6 := by
sorry

end NUMINAMATH_CALUDE_power_sum_equals_six_l937_93718


namespace NUMINAMATH_CALUDE_complete_square_result_l937_93721

/-- Given a quadratic equation x^2 - 6x + 5 = 0, prove that when completing the square, 
    the resulting equation (x + c)^2 = d has d = 4 -/
theorem complete_square_result (c : ℝ) : 
  ∃ d : ℝ, (∀ x : ℝ, x^2 - 6*x + 5 = 0 ↔ (x + c)^2 = d) ∧ d = 4 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_result_l937_93721


namespace NUMINAMATH_CALUDE_metal_bars_per_set_l937_93706

theorem metal_bars_per_set (total_bars : ℕ) (num_sets : ℕ) (bars_per_set : ℕ) : 
  total_bars = 14 → num_sets = 2 → total_bars = num_sets * bars_per_set → bars_per_set = 7 := by
  sorry

end NUMINAMATH_CALUDE_metal_bars_per_set_l937_93706


namespace NUMINAMATH_CALUDE_cube_difference_implies_sum_of_squares_l937_93787

theorem cube_difference_implies_sum_of_squares (n : ℕ) (hn : n > 0) :
  (∃ x : ℕ, x > 0 ∧ (x + 1)^3 - x^3 = n^2) →
  ∃ a b : ℕ, n = a^2 + b^2 := by
sorry

end NUMINAMATH_CALUDE_cube_difference_implies_sum_of_squares_l937_93787


namespace NUMINAMATH_CALUDE_first_note_denomination_l937_93705

/-- Proves that given the conditions of the problem, the denomination of the first type of notes must be 1 rupee -/
theorem first_note_denomination (total_amount : ℕ) (total_notes : ℕ) (x : ℕ) : 
  total_amount = 400 →
  total_notes = 75 →
  total_amount = (total_notes / 3 * x) + (total_notes / 3 * 5) + (total_notes / 3 * 10) →
  x = 1 := by
  sorry

#check first_note_denomination

end NUMINAMATH_CALUDE_first_note_denomination_l937_93705


namespace NUMINAMATH_CALUDE_pencils_sold_initially_l937_93758

-- Define the number of pencils sold at 15% gain
def pencils_at_gain : ℝ := 7.391304347826086

-- Theorem statement
theorem pencils_sold_initially (x : ℝ) :
  (0.85 * x * (1 / (1.15 * pencils_at_gain)) = 1) →
  x = 10 := by
  sorry

end NUMINAMATH_CALUDE_pencils_sold_initially_l937_93758


namespace NUMINAMATH_CALUDE_unique_solution_linear_system_l937_93744

theorem unique_solution_linear_system :
  ∃! (x y : ℝ), x + y = 5 ∧ x + 2*y = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_solution_linear_system_l937_93744


namespace NUMINAMATH_CALUDE_student_calculation_l937_93700

theorem student_calculation (chosen_number : ℕ) : 
  chosen_number = 40 → chosen_number * 7 - 150 = 130 := by
  sorry

end NUMINAMATH_CALUDE_student_calculation_l937_93700


namespace NUMINAMATH_CALUDE_alpha_plus_beta_equals_two_l937_93786

theorem alpha_plus_beta_equals_two (α β : ℝ) 
  (hα : α^3 - 3*α^2 + 5*α - 17 = 0)
  (hβ : β^3 - 3*β^2 + 5*β + 11 = 0) : 
  α + β = 2 := by
sorry

end NUMINAMATH_CALUDE_alpha_plus_beta_equals_two_l937_93786


namespace NUMINAMATH_CALUDE_functional_equation_solution_l937_93745

-- Define the function type
def FunctionType (k : ℝ) := {f : ℝ → ℝ // ∀ x, x ∈ Set.Icc (-k) k → f x ∈ Set.Icc 0 k}

-- State the theorem
theorem functional_equation_solution (k : ℝ) (h_k : k > 0) :
  ∀ f : FunctionType k,
    (∀ x y, x ∈ Set.Icc (-k) k → y ∈ Set.Icc (-k) k → x + y ∈ Set.Icc (-k) k →
      (f.val x)^2 + (f.val y)^2 - 2*x*y = k^2 + (f.val (x + y))^2) →
    ∃ a c : ℝ, ∀ x ∈ Set.Icc (-k) k,
      f.val x = Real.sqrt (a * x + c - x^2) ∧
      0 ≤ a * x + c - x^2 ∧
      a * x + c - x^2 ≤ k^2 :=
by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l937_93745


namespace NUMINAMATH_CALUDE_quadratic_root_ratio_l937_93754

theorem quadratic_root_ratio (p : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ 0 ∧ x₂ / x₁ = -4 ∧ x₁^2 + p*x₁ - 16 = 0 ∧ x₂^2 + p*x₂ - 16 = 0) → 
  (p = 6 ∨ p = -6) := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_ratio_l937_93754


namespace NUMINAMATH_CALUDE_sandy_dozens_of_marbles_l937_93749

def melanie_marbles : ℕ := 84
def sandy_multiplier : ℕ := 8
def marbles_per_dozen : ℕ := 12

theorem sandy_dozens_of_marbles :
  (melanie_marbles * sandy_multiplier) / marbles_per_dozen = 56 := by
  sorry

end NUMINAMATH_CALUDE_sandy_dozens_of_marbles_l937_93749


namespace NUMINAMATH_CALUDE_division_problem_l937_93772

theorem division_problem (dividend quotient remainder divisor : ℕ) 
  (h1 : dividend = 172)
  (h2 : quotient = 10)
  (h3 : remainder = 2)
  (h4 : dividend = quotient * divisor + remainder) :
  divisor = 17 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l937_93772


namespace NUMINAMATH_CALUDE_prob_A_third_try_prob_at_least_one_success_l937_93701

/-- Probability of 甲 solving the cube within 30 seconds -/
def prob_A : ℝ := 0.8

/-- Probability of 乙 solving the cube within 30 seconds -/
def prob_B : ℝ := 0.6

/-- Each attempt is independent -/
axiom attempts_independent : True

/-- Probability of 甲 succeeding on the third try -/
theorem prob_A_third_try : 
  (1 - prob_A) * (1 - prob_A) * prob_A = 0.032 := by sorry

/-- Probability of at least one person succeeding on the first try -/
theorem prob_at_least_one_success : 
  1 - (1 - prob_A) * (1 - prob_B) = 0.92 := by sorry

end NUMINAMATH_CALUDE_prob_A_third_try_prob_at_least_one_success_l937_93701


namespace NUMINAMATH_CALUDE_walking_rate_problem_l937_93788

/-- Proves that given the conditions of the problem, the walking rate when missing the train is 4 kmph -/
theorem walking_rate_problem (distance : ℝ) (early_rate : ℝ) (early_time : ℝ) (late_time : ℝ) :
  distance = 4 →
  early_rate = 5 →
  early_time = 6 →
  late_time = 6 →
  ∃ (late_rate : ℝ),
    (distance / early_rate) * 60 + early_time = (distance / late_rate) * 60 - late_time ∧
    late_rate = 4 :=
by sorry

end NUMINAMATH_CALUDE_walking_rate_problem_l937_93788


namespace NUMINAMATH_CALUDE_friendly_pairs_complete_l937_93704

def FriendlyPair (a b c d : ℕ+) : Prop :=
  2 * (a.val + b.val) = c.val * d.val ∧ 2 * (c.val + d.val) = a.val * b.val

def AllFriendlyPairs : Set (ℕ+ × ℕ+ × ℕ+ × ℕ+) :=
  {⟨22, 5, 54, 1⟩, ⟨13, 6, 38, 1⟩, ⟨10, 7, 34, 1⟩, ⟨10, 3, 13, 2⟩,
   ⟨6, 4, 10, 2⟩, ⟨6, 3, 6, 3⟩, ⟨4, 4, 4, 4⟩}

theorem friendly_pairs_complete :
  ∀ a b c d : ℕ+, FriendlyPair a b c d ↔ (a, b, c, d) ∈ AllFriendlyPairs :=
sorry

end NUMINAMATH_CALUDE_friendly_pairs_complete_l937_93704


namespace NUMINAMATH_CALUDE_one_nonnegative_solution_l937_93756

theorem one_nonnegative_solution :
  ∃! (x : ℝ), x ≥ 0 ∧ x^2 = -9*x :=
by sorry

end NUMINAMATH_CALUDE_one_nonnegative_solution_l937_93756


namespace NUMINAMATH_CALUDE_percentage_problem_l937_93785

theorem percentage_problem (P : ℝ) : 
  (P / 100) * 600 = (50 / 100) * 960 → P = 80 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l937_93785


namespace NUMINAMATH_CALUDE_cube_volume_problem_l937_93741

theorem cube_volume_problem (s : ℝ) : 
  s > 0 →
  s^3 - ((s + 2) * (s - 3) * s) = 8 →
  s^3 = 8 :=
by sorry

end NUMINAMATH_CALUDE_cube_volume_problem_l937_93741


namespace NUMINAMATH_CALUDE_parallelogram_EFGH_area_l937_93783

-- Define the parallelogram EFGH
def E : ℝ × ℝ := (1, 3)
def F : ℝ × ℝ := (5, 3)
def G : ℝ × ℝ := (6, 1)
def H : ℝ × ℝ := (2, 1)

-- Define the area function for a parallelogram
def parallelogram_area (a b c d : ℝ × ℝ) : ℝ :=
  let base := abs (b.1 - a.1)
  let height := abs (a.2 - d.2)
  base * height

-- Theorem statement
theorem parallelogram_EFGH_area :
  parallelogram_area E F G H = 8 := by sorry

end NUMINAMATH_CALUDE_parallelogram_EFGH_area_l937_93783


namespace NUMINAMATH_CALUDE_rick_ironing_theorem_l937_93778

/-- Represents the number of dress shirts Rick can iron in an hour -/
def shirts_per_hour : ℕ := 4

/-- Represents the number of dress pants Rick can iron in an hour -/
def pants_per_hour : ℕ := 3

/-- Represents the number of hours Rick spends ironing dress shirts -/
def hours_ironing_shirts : ℕ := 3

/-- Represents the number of hours Rick spends ironing dress pants -/
def hours_ironing_pants : ℕ := 5

/-- Calculates the total number of pieces of clothing Rick has ironed -/
def total_clothes_ironed : ℕ :=
  shirts_per_hour * hours_ironing_shirts + pants_per_hour * hours_ironing_pants

theorem rick_ironing_theorem :
  total_clothes_ironed = 27 := by
  sorry

end NUMINAMATH_CALUDE_rick_ironing_theorem_l937_93778


namespace NUMINAMATH_CALUDE_line_perp_plane_and_line_implies_parallel_l937_93734

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Line → Prop)
variable (perpToPlane : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)

-- State the theorem
theorem line_perp_plane_and_line_implies_parallel
  (l m : Line) (α : Plane)
  (h1 : l ≠ m)
  (h2 : perpToPlane l α)
  (h3 : perp l m) :
  parallel m α :=
sorry

end NUMINAMATH_CALUDE_line_perp_plane_and_line_implies_parallel_l937_93734


namespace NUMINAMATH_CALUDE_problem_statement_l937_93703

theorem problem_statement (a b c d e : ℝ) 
  (h : a^2 + b^2 + c^2 + 2 = e + Real.sqrt (a + b + c + d - e)) :
  e = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l937_93703


namespace NUMINAMATH_CALUDE_f_of_2_equals_7_l937_93709

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - x^2 + 2*x - 1

-- State the theorem
theorem f_of_2_equals_7 : f 2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_f_of_2_equals_7_l937_93709


namespace NUMINAMATH_CALUDE_student_arrangement_count_l937_93792

/-- The number of ways to arrange 6 students in a line with 3 friends not adjacent -/
def arrangement_count : ℕ := 576

/-- Total number of students -/
def total_students : ℕ := 6

/-- Number of friends who refuse to stand next to each other -/
def friend_count : ℕ := 3

/-- Number of non-friend students -/
def non_friend_count : ℕ := total_students - friend_count

theorem student_arrangement_count :
  arrangement_count =
    (Nat.factorial total_students) -
    ((Nat.factorial non_friend_count) *
     (Nat.choose (non_friend_count + 1) friend_count) *
     (Nat.factorial friend_count)) :=
by sorry

end NUMINAMATH_CALUDE_student_arrangement_count_l937_93792


namespace NUMINAMATH_CALUDE_velvet_area_for_given_box_l937_93731

/-- The total area of velvet needed to line the inside of a box with given dimensions -/
def total_velvet_area (long_side_length long_side_width short_side_length short_side_width top_bottom_area : ℕ) : ℕ :=
  2 * (long_side_length * long_side_width) +
  2 * (short_side_length * short_side_width) +
  2 * top_bottom_area

/-- Theorem stating that the total area of velvet needed for the given box dimensions is 236 square inches -/
theorem velvet_area_for_given_box : total_velvet_area 8 6 5 6 40 = 236 := by
  sorry

end NUMINAMATH_CALUDE_velvet_area_for_given_box_l937_93731


namespace NUMINAMATH_CALUDE_sequence_sum_l937_93797

theorem sequence_sum (a : ℕ → ℤ) : 
  (∀ n : ℕ, a (n + 1) - a n = 2) → 
  a 1 = -5 → 
  (|a 1| + |a 2| + |a 3| + |a 4| + |a 5| + |a 6| : ℤ) = 18 := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_l937_93797


namespace NUMINAMATH_CALUDE_smallest_b_l937_93761

theorem smallest_b (a b : ℕ+) (h1 : a.val - b.val = 7) 
  (h2 : Nat.gcd ((a.val^3 + b.val^3) / (a.val + b.val)) (a.val^2 * b.val) = 12) : 
  b.val ≥ 6 ∧ ∃ (a' b' : ℕ+), b'.val = 6 ∧ a'.val - b'.val = 7 ∧ 
    Nat.gcd ((a'.val^3 + b'.val^3) / (a'.val + b'.val)) (a'.val^2 * b'.val) = 12 :=
by sorry

end NUMINAMATH_CALUDE_smallest_b_l937_93761


namespace NUMINAMATH_CALUDE_jordan_born_in_1980_l937_93795

/-- The year when the first AMC 8 was given -/
def first_amc8_year : ℕ := 1985

/-- The age of Jordan when he took the tenth AMC 8 contest -/
def jordan_age_at_tenth_amc8 : ℕ := 14

/-- The number of years between the first AMC 8 and the tenth AMC 8 -/
def years_between_first_and_tenth : ℕ := 9

/-- Jordan's birth year -/
def jordan_birth_year : ℕ := first_amc8_year + years_between_first_and_tenth - jordan_age_at_tenth_amc8

theorem jordan_born_in_1980 : jordan_birth_year = 1980 := by
  sorry

end NUMINAMATH_CALUDE_jordan_born_in_1980_l937_93795


namespace NUMINAMATH_CALUDE_smallest_solution_congruence_l937_93730

theorem smallest_solution_congruence :
  ∃ (x : ℕ), x > 0 ∧ (5 * x) % 34 = 17 % 34 ∧ ∀ (y : ℕ), y > 0 → (5 * y) % 34 = 17 % 34 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_congruence_l937_93730


namespace NUMINAMATH_CALUDE_sine_cosine_power_sum_l937_93757

theorem sine_cosine_power_sum (x : ℝ) (h : Real.sin x + Real.cos x = -1) :
  ∀ n : ℕ, (Real.sin x)^n + (Real.cos x)^n = (-1)^n := by
sorry

end NUMINAMATH_CALUDE_sine_cosine_power_sum_l937_93757


namespace NUMINAMATH_CALUDE_function_properties_l937_93779

-- Define the properties of function f
def additive (f : ℝ → ℝ) : Prop :=
  ∀ x y, f (x + y) = f x + f y

def positive_for_positive (f : ℝ → ℝ) : Prop :=
  ∀ x, x > 0 → f x > 0

-- State the theorem
theorem function_properties (f : ℝ → ℝ) 
  (h_add : additive f) (h_pos : positive_for_positive f) : 
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x y, x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l937_93779


namespace NUMINAMATH_CALUDE_custom_op_value_l937_93751

-- Define the custom operation *
def custom_op (a b : ℚ) : ℚ := 1 / a + 1 / b

-- Theorem statement
theorem custom_op_value (a b : ℚ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (sum_eq : a + b = 12) (prod_eq : a * b = 32) : 
  custom_op a b = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_value_l937_93751


namespace NUMINAMATH_CALUDE_marker_distance_l937_93798

theorem marker_distance (k : ℝ) (h_pos : k > 0) : 
  (∀ n : ℕ, ∀ m : ℕ, m - n = 4 → 
    Real.sqrt ((m - n)^2 + (m*k - n*k)^2) = 31) →
  Real.sqrt ((19 - 7)^2 + (19*k - 7*k)^2) = 93 :=
by sorry

end NUMINAMATH_CALUDE_marker_distance_l937_93798


namespace NUMINAMATH_CALUDE_lizard_feature_difference_l937_93765

/-- Represents a three-eyed lizard with wrinkles and spots -/
structure Lizard :=
  (eyes : ℕ)
  (wrinkle_factor : ℕ)
  (spot_factor : ℕ)

/-- Calculate the total number of features (eyes, wrinkles, and spots) for a lizard -/
def total_features (l : Lizard) : ℕ :=
  l.eyes + (l.eyes * l.wrinkle_factor) + (l.eyes * l.wrinkle_factor * l.spot_factor)

/-- The main theorem about the difference between total features and eyes for two lizards -/
theorem lizard_feature_difference (jan_lizard cousin_lizard : Lizard)
  (h1 : jan_lizard.eyes = 3)
  (h2 : jan_lizard.wrinkle_factor = 3)
  (h3 : jan_lizard.spot_factor = 7)
  (h4 : cousin_lizard.eyes = 3)
  (h5 : cousin_lizard.wrinkle_factor = 2)
  (h6 : cousin_lizard.spot_factor = 5) :
  (total_features jan_lizard + total_features cousin_lizard) - (jan_lizard.eyes + cousin_lizard.eyes) = 102 :=
sorry

end NUMINAMATH_CALUDE_lizard_feature_difference_l937_93765


namespace NUMINAMATH_CALUDE_polynomial_never_33_l937_93739

theorem polynomial_never_33 (x y : ℤ) : 
  x^5 + 3*x^4*y - 5*x^3*y^2 - 15*x^2*y^3 + 4*x*y^4 + 12*y^5 ≠ 33 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_never_33_l937_93739


namespace NUMINAMATH_CALUDE_three_teams_of_four_from_twelve_l937_93720

-- Define the number of participants
def n : ℕ := 12

-- Define the number of teams
def k : ℕ := 3

-- Define the number of players per team
def m : ℕ := 4

-- Theorem statement
theorem three_teams_of_four_from_twelve (h1 : n = 12) (h2 : k = 3) (h3 : m = 4) (h4 : n = k * m) : 
  (Nat.choose n m * Nat.choose (n - m) m * Nat.choose (n - 2*m) m) / (Nat.factorial k) = 5775 := by
  sorry

end NUMINAMATH_CALUDE_three_teams_of_four_from_twelve_l937_93720


namespace NUMINAMATH_CALUDE_circuit_malfunction_probability_l937_93715

/-- Represents an electronic component with a given failure rate -/
structure Component where
  failureRate : ℝ
  hFailureRate : 0 ≤ failureRate ∧ failureRate ≤ 1

/-- Represents a circuit with two components connected in series -/
structure Circuit where
  componentA : Component
  componentB : Component

/-- The probability of a circuit malfunctioning -/
def malfunctionProbability (c : Circuit) : ℝ :=
  1 - (1 - c.componentA.failureRate) * (1 - c.componentB.failureRate)

theorem circuit_malfunction_probability (c : Circuit) 
    (hA : c.componentA.failureRate = 0.2)
    (hB : c.componentB.failureRate = 0.5) :
    malfunctionProbability c = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_circuit_malfunction_probability_l937_93715


namespace NUMINAMATH_CALUDE_amusement_park_group_composition_l937_93791

theorem amusement_park_group_composition :
  let total_cost : ℕ := 720
  let adult_price : ℕ := 15
  let child_price : ℕ := 8
  let num_children : ℕ := 15
  let num_adults : ℕ := (total_cost - child_price * num_children) / adult_price
  num_adults - num_children = 25 := by
  sorry

end NUMINAMATH_CALUDE_amusement_park_group_composition_l937_93791


namespace NUMINAMATH_CALUDE_earth_angle_calculation_l937_93784

/-- The angle between two points on a spherical Earth given their coordinates --/
def spherical_angle (lat1 : Real) (lon1 : Real) (lat2 : Real) (lon2 : Real) : Real :=
  sorry

theorem earth_angle_calculation :
  let p_lat : Real := 0
  let p_lon : Real := 100
  let q_lat : Real := 30
  let q_lon : Real := -100 -- Negative for West longitude
  spherical_angle p_lat p_lon q_lat q_lon = 160 := by
  sorry

end NUMINAMATH_CALUDE_earth_angle_calculation_l937_93784


namespace NUMINAMATH_CALUDE_circles_tangent_implies_a_value_l937_93775

/-- Definition of circle C₁ -/
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 16

/-- Definition of circle C₂ -/
def C₂ (a x y : ℝ) : Prop := (x - a)^2 + y^2 = 1

/-- Two circles are tangent if the distance between their centers equals the sum or difference of their radii -/
def are_tangent (a : ℝ) : Prop := |a| = 5 ∨ |a| = 3

/-- Theorem stating that if C₁ and C₂ are tangent, then |a| = 5 or |a| = 3 -/
theorem circles_tangent_implies_a_value (a : ℝ) :
  (∃ x y : ℝ, C₁ x y ∧ C₂ a x y) → are_tangent a :=
sorry

end NUMINAMATH_CALUDE_circles_tangent_implies_a_value_l937_93775


namespace NUMINAMATH_CALUDE_men_who_left_hostel_l937_93729

/-- Proves that 50 men left the hostel given the initial and final conditions -/
theorem men_who_left_hostel 
  (initial_men : ℕ) 
  (initial_days : ℕ) 
  (final_days : ℕ) 
  (h1 : initial_men = 250)
  (h2 : initial_days = 28)
  (h3 : final_days = 35)
  (h4 : initial_men * initial_days = (initial_men - men_who_left) * final_days) :
  men_who_left = 50 := by
  sorry

#check men_who_left_hostel

end NUMINAMATH_CALUDE_men_who_left_hostel_l937_93729


namespace NUMINAMATH_CALUDE_peach_to_apricot_ratio_l937_93702

/-- Given a total number of trees and a number of apricot trees, 
    calculate the ratio of peach trees to apricot trees. -/
def tree_ratio (total : ℕ) (apricot : ℕ) : ℚ × ℚ :=
  let peach := total - apricot
  (peach, apricot)

/-- The theorem states that for 232 total trees and 58 apricot trees,
    the ratio of peach trees to apricot trees is 3:1. -/
theorem peach_to_apricot_ratio :
  tree_ratio 232 58 = (3, 1) := by sorry

end NUMINAMATH_CALUDE_peach_to_apricot_ratio_l937_93702


namespace NUMINAMATH_CALUDE_chicken_cost_is_40_cents_l937_93725

/-- The cost of chicken per plate given the total number of plates, 
    cost of rice per plate, and total spent on food. -/
def chicken_cost_per_plate (total_plates : ℕ) (rice_cost_per_plate : ℚ) (total_spent : ℚ) : ℚ :=
  (total_spent - (total_plates : ℚ) * rice_cost_per_plate) / (total_plates : ℚ)

/-- Theorem stating that the cost of chicken per plate is $0.40 
    given the specific conditions of the problem. -/
theorem chicken_cost_is_40_cents :
  chicken_cost_per_plate 100 (1/10) 50 = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_chicken_cost_is_40_cents_l937_93725


namespace NUMINAMATH_CALUDE_bookshelf_capacity_l937_93732

theorem bookshelf_capacity (num_bookshelves : ℕ) (layers_per_bookshelf : ℕ) (books_per_layer : ℕ) 
  (h1 : num_bookshelves = 8) 
  (h2 : layers_per_bookshelf = 5) 
  (h3 : books_per_layer = 85) : 
  num_bookshelves * layers_per_bookshelf * books_per_layer = 3400 := by
  sorry

end NUMINAMATH_CALUDE_bookshelf_capacity_l937_93732


namespace NUMINAMATH_CALUDE_some_multiplier_value_l937_93796

theorem some_multiplier_value : ∃ (some_multiplier : ℤ), 
  |5 - some_multiplier * (3 - 12)| - |5 - 11| = 71 ∧ some_multiplier = 8 := by
  sorry

end NUMINAMATH_CALUDE_some_multiplier_value_l937_93796


namespace NUMINAMATH_CALUDE_max_value_sum_of_fractions_l937_93780

theorem max_value_sum_of_fractions (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x^2 / (1 + x^2) + y^2 / (1 + y^2) + z^2 / (1 + z^2) = 2) :
  x / (1 + x^2) + y / (1 + y^2) + z / (1 + z^2) ≤ Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_sum_of_fractions_l937_93780


namespace NUMINAMATH_CALUDE_intersection_point_of_function_and_inverse_l937_93769

-- Define the function f
def f (b : ℤ) : ℝ → ℝ := λ x => 4 * x + b

-- State the theorem
theorem intersection_point_of_function_and_inverse
  (b : ℤ) (a : ℤ) :
  (∃ (x : ℝ), f b x = x ∧ f b (-4) = a) →
  a = -4 :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_of_function_and_inverse_l937_93769


namespace NUMINAMATH_CALUDE_exists_specific_number_l937_93750

theorem exists_specific_number : ∃ y : ℕ+, 
  (y.val % 4 = 0) ∧ 
  (y.val % 5 = 0) ∧ 
  (y.val % 7 = 0) ∧ 
  (y.val % 13 = 0) ∧ 
  (y.val % 8 ≠ 0) ∧ 
  (y.val % 15 ≠ 0) ∧ 
  (y.val % 50 ≠ 0) ∧ 
  (y.val % 10 = 0) ∧ 
  (y.val = 1820) :=
by sorry

end NUMINAMATH_CALUDE_exists_specific_number_l937_93750


namespace NUMINAMATH_CALUDE_vitamin_d_scientific_notation_l937_93716

theorem vitamin_d_scientific_notation : 0.0000046 = 4.6 * 10^(-6) := by
  sorry

end NUMINAMATH_CALUDE_vitamin_d_scientific_notation_l937_93716


namespace NUMINAMATH_CALUDE_fraction_decomposition_l937_93766

theorem fraction_decomposition :
  ∃ (C D : ℝ),
    (C = -0.1 ∧ D = 7.3) ∧
    ∀ (x : ℝ), x ≠ 2 ∧ 3*x ≠ -4 →
      (7*x - 15) / (3*x^2 + 2*x - 8) = C / (x - 2) + D / (3*x + 4) := by
  sorry

end NUMINAMATH_CALUDE_fraction_decomposition_l937_93766


namespace NUMINAMATH_CALUDE_product_of_differences_l937_93760

theorem product_of_differences (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) 
  (h₁ : x₁^3 - 3*x₁*y₁^2 = 2005) (h₂ : y₁^3 - 3*x₁^2*y₁ = 2004)
  (h₃ : x₂^3 - 3*x₂*y₂^2 = 2005) (h₄ : y₂^3 - 3*x₂^2*y₂ = 2004)
  (h₅ : x₃^3 - 3*x₃*y₃^2 = 2005) (h₆ : y₃^3 - 3*x₃^2*y₃ = 2004)
  (h₇ : y₁ ≠ 0) (h₈ : y₂ ≠ 0) (h₉ : y₃ ≠ 0) :
  (1 - x₁/y₁) * (1 - x₂/y₂) * (1 - x₃/y₃) = 1/1002 := by
sorry

end NUMINAMATH_CALUDE_product_of_differences_l937_93760


namespace NUMINAMATH_CALUDE_petya_wins_l937_93768

/-- Represents the game state -/
structure GameState where
  stones : ℕ
  playerTurn : Bool  -- true for Petya, false for Vasya

/-- Defines a valid move in the game -/
def validMove (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 9

/-- Applies a move to the game state -/
def applyMove (state : GameState) (move : ℕ) : GameState :=
  { stones := state.stones - move, playerTurn := ¬state.playerTurn }

/-- Determines if the game is over -/
def gameOver (state : GameState) : Prop := state.stones = 0

/-- Defines a winning strategy for the first player -/
def winningStrategy (strategy : GameState → ℕ) : Prop :=
  ∀ (state : GameState), 
    validMove (strategy state) ∧ 
    (gameOver (applyMove state (strategy state)) ∨ 
     ∀ (opponentMove : ℕ), validMove opponentMove → 
       ¬gameOver (applyMove (applyMove state (strategy state)) opponentMove))

/-- Theorem: The first player (Petya) has a winning strategy -/
theorem petya_wins : 
  ∃ (strategy : GameState → ℕ), winningStrategy strategy ∧ 
    (∀ (state : GameState), state.stones = 111 ∧ state.playerTurn = true → 
      gameOver (applyMove state (strategy state)) ∨ 
      ∀ (opponentMove : ℕ), validMove opponentMove → 
        ¬gameOver (applyMove (applyMove state (strategy state)) opponentMove)) :=
sorry

end NUMINAMATH_CALUDE_petya_wins_l937_93768


namespace NUMINAMATH_CALUDE_building_C_floors_l937_93728

/-- The number of floors in Building A -/
def floors_A : ℕ := 4

/-- The number of floors in Building B -/
def floors_B : ℕ := floors_A + 9

/-- The number of floors in Building C -/
def floors_C : ℕ := 5 * floors_B - 6

theorem building_C_floors : floors_C = 59 := by
  sorry

end NUMINAMATH_CALUDE_building_C_floors_l937_93728


namespace NUMINAMATH_CALUDE_inequality_proof_l937_93790

theorem inequality_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x * y + 2 * y * z + 2 * z * x) / (x^2 + y^2 + z^2) ≤ (Real.sqrt 33 + 1) / 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l937_93790


namespace NUMINAMATH_CALUDE_simplify_expression_l937_93736

theorem simplify_expression (a b : ℝ) : 120*a - 55*a + 33*b - 7*b = 65*a + 26*b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l937_93736


namespace NUMINAMATH_CALUDE_odd_positive_poly_one_real_zero_l937_93763

/-- A polynomial with positive real coefficients of odd degree -/
structure OddPositivePoly where
  degree : Nat
  coeffs : Fin degree → ℝ
  odd_degree : Odd degree
  positive_coeffs : ∀ i, coeffs i > 0

/-- A permutation of the coefficients of a polynomial -/
def PermutedPoly (p : OddPositivePoly) :=
  { σ : Equiv (Fin p.degree) (Fin p.degree) // True }

/-- The number of real zeros of a polynomial -/
noncomputable def num_real_zeros (p : OddPositivePoly) (perm : PermutedPoly p) : ℕ :=
  sorry

/-- Theorem: For any odd degree polynomial with positive coefficients,
    there exists a permutation of its coefficients such that
    the resulting polynomial has exactly one real zero -/
theorem odd_positive_poly_one_real_zero (p : OddPositivePoly) :
  ∃ perm : PermutedPoly p, num_real_zeros p perm = 1 :=
sorry

end NUMINAMATH_CALUDE_odd_positive_poly_one_real_zero_l937_93763


namespace NUMINAMATH_CALUDE_binomial_coefficient_19_11_l937_93771

theorem binomial_coefficient_19_11 :
  (Nat.choose 19 11 = 82654) ∧ (Nat.choose 17 9 = 24310) ∧ (Nat.choose 17 7 = 19448) → 
  Nat.choose 19 11 = 82654 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_19_11_l937_93771


namespace NUMINAMATH_CALUDE_fuji_fraction_l937_93724

/-- Represents an apple orchard with Fuji and Gala trees -/
structure Orchard where
  total : ℕ
  fuji : ℕ
  gala : ℕ
  crossPollinated : ℕ

/-- The conditions of the orchard as described in the problem -/
def orchard_conditions (o : Orchard) : Prop :=
  o.crossPollinated = o.total / 10 ∧
  o.fuji + o.crossPollinated = 170 ∧
  o.gala = 30 ∧
  o.total = o.fuji + o.gala + o.crossPollinated

/-- The theorem stating that 3/4 of the trees are pure Fuji -/
theorem fuji_fraction (o : Orchard) (h : orchard_conditions o) : 
  o.fuji = 3 * o.total / 4 := by
  sorry

#check fuji_fraction

end NUMINAMATH_CALUDE_fuji_fraction_l937_93724


namespace NUMINAMATH_CALUDE_solve_z_l937_93722

-- Define the complex number i
def i : ℂ := Complex.I

-- Define a predicate for purely imaginary numbers
def isPurelyImaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

-- Define the theorem
theorem solve_z (z : ℂ) 
  (h1 : isPurelyImaginary z) 
  (h2 : ((z + 2) / (1 - i)).im = 0) : 
  z = -2 * i := by
  sorry

end NUMINAMATH_CALUDE_solve_z_l937_93722


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l937_93737

theorem triangle_abc_properties (A B C : Real) (a b c : Real) :
  C = 2 * Real.pi / 3 →
  c = 5 →
  a = Real.sqrt 5 * b * Real.sin A →
  b = 2 * Real.sqrt 15 / 3 ∧
  Real.tan (B + Real.pi / 4) = 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l937_93737


namespace NUMINAMATH_CALUDE_distance_P_to_y_axis_l937_93781

/-- The distance from a point to the y-axis in a Cartesian coordinate system --/
def distance_to_y_axis (x y : ℝ) : ℝ := |x|

/-- The point P --/
def P : ℝ × ℝ := (-3, 4)

/-- Theorem: The distance from P(-3, 4) to the y-axis is 3 --/
theorem distance_P_to_y_axis :
  distance_to_y_axis P.1 P.2 = 3 := by
  sorry


end NUMINAMATH_CALUDE_distance_P_to_y_axis_l937_93781


namespace NUMINAMATH_CALUDE_cos_shift_proof_l937_93708

theorem cos_shift_proof (φ : Real) (h1 : 0 < φ) (h2 : φ < π / 2) : 
  let f := λ x : Real => 2 * Real.cos (2 * x)
  let g := λ x : Real => 2 * Real.cos (2 * x - 2 * φ)
  (∃ x₁ x₂ : Real, |f x₁ - g x₂| = 4 ∧ |x₁ - x₂| = π / 6) → φ = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_shift_proof_l937_93708


namespace NUMINAMATH_CALUDE_cos_48_degrees_l937_93777

theorem cos_48_degrees : Real.cos (48 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_48_degrees_l937_93777


namespace NUMINAMATH_CALUDE_johns_bookshop_l937_93767

/-- The total number of books sold over 5 days -/
def total_sold : ℕ := 280

/-- The percentage of books that were not sold -/
def percent_not_sold : ℚ := 54.83870967741935

/-- The initial number of books in John's bookshop -/
def initial_books : ℕ := 620

theorem johns_bookshop :
  initial_books = total_sold / ((100 - percent_not_sold) / 100) := by sorry

end NUMINAMATH_CALUDE_johns_bookshop_l937_93767


namespace NUMINAMATH_CALUDE_m_value_l937_93755

def A (m : ℝ) : Set ℝ := {-1, 3, m^2}
def B : Set ℝ := {3, 4}

theorem m_value (m : ℝ) (h : B ⊆ A m) : m = 2 ∨ m = -2 := by
  sorry

end NUMINAMATH_CALUDE_m_value_l937_93755


namespace NUMINAMATH_CALUDE_a_minus_b_plus_c_value_l937_93719

theorem a_minus_b_plus_c_value (a b c : ℝ) :
  (abs a = 1) → (abs b = 2) → (abs c = 3) → (a > b) → (b > c) →
  ((a - b + c = 0) ∨ (a - b + c = -2)) := by
sorry

end NUMINAMATH_CALUDE_a_minus_b_plus_c_value_l937_93719


namespace NUMINAMATH_CALUDE_train_speed_equation_l937_93717

theorem train_speed_equation (x : ℝ) (h1 : x > 80) : 
  (353 / (x - 80) - 353 / x = 5 / 3) ↔ 
  (353 / (x - 80) - 353 / x = 100 / 60) := by sorry

end NUMINAMATH_CALUDE_train_speed_equation_l937_93717


namespace NUMINAMATH_CALUDE_hyperbola_third_point_x_squared_l937_93789

/-- A hyperbola is defined by its center, orientation, and three points it passes through. -/
structure Hyperbola where
  center : ℝ × ℝ
  opens_horizontally : Bool
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ
  point3 : ℝ × ℝ

/-- The theorem states that for a specific hyperbola, the square of the x-coordinate of its third point is 361/36. -/
theorem hyperbola_third_point_x_squared (h : Hyperbola) 
  (h_center : h.center = (1, 0))
  (h_orientation : h.opens_horizontally = true)
  (h_point1 : h.point1 = (0, 3))
  (h_point2 : h.point2 = (1, -4))
  (h_point3 : h.point3.2 = -1) :
  (h.point3.1)^2 = 361/36 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_third_point_x_squared_l937_93789


namespace NUMINAMATH_CALUDE_real_roots_of_equation_l937_93742

theorem real_roots_of_equation :
  ∀ x : ℝ, x^4 + x^2 - 20 = 0 ↔ x = 2 ∨ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_real_roots_of_equation_l937_93742


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l937_93723

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 3 * b = 1) :
  (1 / a + 1 / b) ≥ 8 + 4 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l937_93723


namespace NUMINAMATH_CALUDE_range_of_x_l937_93733

theorem range_of_x (x : ℝ) : 
  (¬ (x ∈ Set.Icc 2 5 ∨ x < 1 ∨ x > 4)) → (1 ≤ x ∧ x < 2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_x_l937_93733


namespace NUMINAMATH_CALUDE_perimeter_of_figure_l937_93799

/-- A point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A triangle defined by three points -/
structure Triangle :=
  (A B C : Point)

/-- A square defined by four points -/
structure Square :=
  (E H I J : Point)

/-- The figure ABCDEFGHIJ -/
structure Figure :=
  (A B C D E F G H I J : Point)

/-- Definition of an equilateral triangle -/
def isEquilateral (t : Triangle) : Prop :=
  sorry

/-- Definition of a midpoint -/
def isMidpoint (M A B : Point) : Prop :=
  sorry

/-- Definition of a square -/
def isSquare (s : Square) : Prop :=
  sorry

/-- Distance between two points -/
def distance (A B : Point) : ℝ :=
  sorry

/-- Perimeter of the figure -/
def perimeter (fig : Figure) : ℝ :=
  sorry

/-- Main theorem -/
theorem perimeter_of_figure (fig : Figure) :
  isEquilateral ⟨fig.A, fig.B, fig.C⟩ →
  isEquilateral ⟨fig.A, fig.D, fig.E⟩ →
  isEquilateral ⟨fig.E, fig.F, fig.G⟩ →
  isMidpoint fig.D fig.A fig.C →
  isMidpoint fig.G fig.A fig.E →
  isSquare ⟨fig.E, fig.H, fig.I, fig.J⟩ →
  distance fig.E fig.J = distance fig.D fig.E →
  distance fig.A fig.B = 6 →
  perimeter fig = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_of_figure_l937_93799


namespace NUMINAMATH_CALUDE_unique_solution_l937_93743

/-- The polynomial P(x) = 3x^4 + ax^3 + bx^2 - 16x + 55 -/
def P (a b x : ℝ) : ℝ := 3 * x^4 + a * x^3 + b * x^2 - 16 * x + 55

/-- The first divisibility condition -/
def condition1 (a b : ℝ) : Prop :=
  P a b (-4/3) = 23

/-- The second divisibility condition -/
def condition2 (a b : ℝ) : Prop :=
  P a b 3 = 10

theorem unique_solution :
  ∃! (a b : ℝ), condition1 a b ∧ condition2 a b ∧ a = -29 ∧ b = 7 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l937_93743


namespace NUMINAMATH_CALUDE_two_sides_and_angle_unique_two_angles_and_side_unique_three_sides_unique_two_sides_and_included_angle_not_unique_l937_93727

/-- Represents a triangle -/
structure Triangle :=
  (a b c : ℝ)  -- sides
  (α β γ : ℝ)  -- angles

/-- Determines if a triangle is uniquely defined -/
def is_unique_triangle (t : Triangle) : Prop := sorry

/-- Two sides and an angle uniquely determine a triangle -/
theorem two_sides_and_angle_unique (a b : ℝ) (α : ℝ) : 
  ∃! t : Triangle, t.a = a ∧ t.b = b ∧ t.α = α := sorry

/-- Two angles and a side uniquely determine a triangle -/
theorem two_angles_and_side_unique (α β : ℝ) (a : ℝ) : 
  ∃! t : Triangle, t.α = α ∧ t.β = β ∧ t.a = a := sorry

/-- Three sides uniquely determine a triangle -/
theorem three_sides_unique (a b c : ℝ) : 
  ∃! t : Triangle, t.a = a ∧ t.b = b ∧ t.c = c := sorry

/-- Two sides and their included angle do not uniquely determine a triangle -/
theorem two_sides_and_included_angle_not_unique (a b : ℝ) (γ : ℝ) : 
  ¬(∃! t : Triangle, t.a = a ∧ t.b = b ∧ t.γ = γ) := sorry

end NUMINAMATH_CALUDE_two_sides_and_angle_unique_two_angles_and_side_unique_three_sides_unique_two_sides_and_included_angle_not_unique_l937_93727


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a7_l937_93714

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a7 (a : ℕ → ℚ) 
    (h_arith : is_arithmetic_sequence a)
    (h_a3 : a 3 = 2)
    (h_a5 : a 5 = 7) : 
  a 7 = 12 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a7_l937_93714


namespace NUMINAMATH_CALUDE_greatest_common_divisor_of_sequence_l937_93748

theorem greatest_common_divisor_of_sequence (a : ℤ) : ∃ (k : ℕ), k = 6 ∧ 
  (∀ n ∈ Finset.range 10, (k : ℤ) ∣ ((a + n)^3 + 3*(a + n)^2 + 2*(a + n))) ∧
  (∀ m > 6, ∃ n ∈ Finset.range 10, ¬((m : ℤ) ∣ ((a + n)^3 + 3*(a + n)^2 + 2*(a + n)))) :=
by sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_of_sequence_l937_93748


namespace NUMINAMATH_CALUDE_complex_multiplication_l937_93710

theorem complex_multiplication (i : ℂ) :
  i^2 = -1 →
  (6 - 7*i) * (3 + 6*i) = 60 + 15*i :=
by sorry

end NUMINAMATH_CALUDE_complex_multiplication_l937_93710


namespace NUMINAMATH_CALUDE_line_through_points_equation_l937_93762

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line in 2D space
structure Line2D where
  slope : ℝ
  yIntercept : ℝ

-- Define the two given points
def pointA : Point2D := { x := 3, y := 0 }
def pointB : Point2D := { x := -3, y := 0 }

-- Theorem: The line passing through pointA and pointB has the equation y = 0
theorem line_through_points_equation :
  ∃ (l : Line2D), l.slope = 0 ∧ l.yIntercept = 0 ∧
  (l.slope * pointA.x + l.yIntercept = pointA.y) ∧
  (l.slope * pointB.x + l.yIntercept = pointB.y) :=
by sorry

end NUMINAMATH_CALUDE_line_through_points_equation_l937_93762


namespace NUMINAMATH_CALUDE_min_socks_for_ten_pairs_l937_93793

/-- Represents the number of socks of each color in the drawer -/
structure SockDrawer :=
  (red : ℕ)
  (green : ℕ)
  (blue : ℕ)
  (black : ℕ)

/-- Calculates the minimum number of socks to draw to guarantee a certain number of pairs -/
def minSocksForPairs (drawer : SockDrawer) (numPairs : ℕ) : ℕ :=
  4 + 1 + 2 * (numPairs - 1)

/-- Theorem stating the minimum number of socks to draw for 10 pairs -/
theorem min_socks_for_ten_pairs (drawer : SockDrawer) 
  (h_red : drawer.red = 100)
  (h_green : drawer.green = 80)
  (h_blue : drawer.blue = 60)
  (h_black : drawer.black = 40) :
  minSocksForPairs drawer 10 = 23 := by
  sorry

#eval minSocksForPairs ⟨100, 80, 60, 40⟩ 10

end NUMINAMATH_CALUDE_min_socks_for_ten_pairs_l937_93793


namespace NUMINAMATH_CALUDE_lydia_road_trip_fuel_usage_l937_93713

/-- Proves that given the conditions of Lydia's road trip, the fraction of fuel used in the second third is 1/3 --/
theorem lydia_road_trip_fuel_usage 
  (total_fuel : ℝ) 
  (first_third_fuel : ℝ) 
  (h1 : total_fuel = 60) 
  (h2 : first_third_fuel = 30) 
  (h3 : ∃ (second_third_fraction : ℝ), 
    first_third_fuel + second_third_fraction * total_fuel + (second_third_fraction / 2) * total_fuel = total_fuel) :
  ∃ (second_third_fraction : ℝ), second_third_fraction = 1/3 := by
sorry


end NUMINAMATH_CALUDE_lydia_road_trip_fuel_usage_l937_93713


namespace NUMINAMATH_CALUDE_total_money_l937_93726

theorem total_money (A B C : ℕ) : 
  A + C = 200 →
  B + C = 340 →
  C = 40 →
  A + B + C = 500 := by
sorry

end NUMINAMATH_CALUDE_total_money_l937_93726


namespace NUMINAMATH_CALUDE_quadratic_sum_and_square_sum_l937_93753

theorem quadratic_sum_and_square_sum (a b c d m n : ℕ+) 
  (h1 : a^2 + b^2 + c^2 + d^2 = 1989)
  (h2 : a + b + c + d = m^2)
  (h3 : max a (max b (max c d)) = n^2) :
  m = 9 ∧ n = 6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_sum_and_square_sum_l937_93753


namespace NUMINAMATH_CALUDE_line_always_intersects_ellipse_l937_93776

/-- The range of m for which the line y = kx + 1 always intersects the ellipse x²/5 + y²/m = 1 -/
theorem line_always_intersects_ellipse (k : ℝ) (m : ℝ) :
  (∀ x y, y = k * x + 1 → x^2 / 5 + y^2 / m = 1) ↔ m ≥ 1 ∧ m ≠ 5 :=
sorry

end NUMINAMATH_CALUDE_line_always_intersects_ellipse_l937_93776


namespace NUMINAMATH_CALUDE_three_gorges_dam_capacity_l937_93711

theorem three_gorges_dam_capacity :
  (16780000 : ℝ) = 1.678 * (10 ^ 7) := by
  sorry

end NUMINAMATH_CALUDE_three_gorges_dam_capacity_l937_93711


namespace NUMINAMATH_CALUDE_valid_schedules_count_l937_93746

/-- The number of employees and days -/
def n : ℕ := 7

/-- Calculate the number of valid schedules -/
def validSchedules : ℕ :=
  n.factorial - 2 * (n - 1).factorial

/-- Theorem stating the number of valid schedules -/
theorem valid_schedules_count :
  validSchedules = 3600 := by sorry

end NUMINAMATH_CALUDE_valid_schedules_count_l937_93746


namespace NUMINAMATH_CALUDE_gcd_459_357_l937_93774

theorem gcd_459_357 : Nat.gcd 459 357 = 51 := by
  sorry

end NUMINAMATH_CALUDE_gcd_459_357_l937_93774
