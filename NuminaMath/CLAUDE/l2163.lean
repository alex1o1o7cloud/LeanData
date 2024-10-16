import Mathlib

namespace NUMINAMATH_CALUDE_multiplication_value_proof_l2163_216324

theorem multiplication_value_proof : 
  let number : ℝ := 7.5
  let divisor : ℝ := 6
  let result : ℝ := 15
  let multiplication_value : ℝ := 12
  (number / divisor) * multiplication_value = result := by
  sorry

end NUMINAMATH_CALUDE_multiplication_value_proof_l2163_216324


namespace NUMINAMATH_CALUDE_octagon_theorem_l2163_216399

def is_permutation (l : List ℕ) : Prop :=
  l.length = 8 ∧ l.toFinset = Finset.range 8

def cyclic_shift (l : List ℕ) (k : ℕ) : List ℕ :=
  (l.drop k ++ l.take k).take 8

def product_sum (l1 l2 : List ℕ) : ℕ :=
  List.sum (List.zipWith (· * ·) l1 l2)

theorem octagon_theorem (l1 l2 : List ℕ) (h1 : is_permutation l1) (h2 : is_permutation l2) :
  ∃ k, product_sum l1 (cyclic_shift l2 k) ≥ 162 := by
  sorry

end NUMINAMATH_CALUDE_octagon_theorem_l2163_216399


namespace NUMINAMATH_CALUDE_prob_15th_roll_last_correct_l2163_216304

/-- The probability of the 15th roll being the last roll when rolling an
    eight-sided die until getting the same number on consecutive rolls. -/
def prob_15th_roll_last : ℚ :=
  (7 ^ 13 : ℚ) / (8 ^ 14 : ℚ)

/-- The number of sides on the die. -/
def num_sides : ℕ := 8

/-- The number of rolls. -/
def num_rolls : ℕ := 15

theorem prob_15th_roll_last_correct :
  prob_15th_roll_last = (7 ^ (num_rolls - 2) : ℚ) / (num_sides ^ (num_rolls - 1) : ℚ) :=
sorry

end NUMINAMATH_CALUDE_prob_15th_roll_last_correct_l2163_216304


namespace NUMINAMATH_CALUDE_cistern_water_breadth_l2163_216346

/-- Proves that for a cistern with given dimensions and wet surface area, 
    the breadth of water is 1.25 meters. -/
theorem cistern_water_breadth 
  (length : ℝ) 
  (width : ℝ) 
  (wet_surface_area : ℝ) 
  (h_length : length = 6) 
  (h_width : width = 4) 
  (h_wet_area : wet_surface_area = 49) : 
  ∃ (breadth : ℝ), 
    breadth = 1.25 ∧ 
    wet_surface_area = length * width + 2 * (length + width) * breadth :=
by sorry

end NUMINAMATH_CALUDE_cistern_water_breadth_l2163_216346


namespace NUMINAMATH_CALUDE_sqrt_x_minus_one_real_l2163_216381

theorem sqrt_x_minus_one_real (x : ℝ) : (∃ y : ℝ, y^2 = x - 1) ↔ x ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_one_real_l2163_216381


namespace NUMINAMATH_CALUDE_ellipse_properties_l2163_216395

-- Define the ellipse (C)
def Ellipse (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the line (l) with slope 1 passing through F(1,0)
def Line (x y : ℝ) : Prop := y = x - 1

-- Define the perpendicular bisector of MN
def PerpendicularBisector (k : ℝ) (x y : ℝ) : Prop :=
  y + (3*k)/(3 + 4*k^2) = -(1/k)*(x - (4*k^2)/(3 + 4*k^2))

theorem ellipse_properties :
  -- Given conditions
  (Ellipse 2 0) →
  (Ellipse 1 0) →
  -- Prove the following
  (∀ x y, Ellipse x y ↔ x^2/4 + y^2/3 = 1) ∧
  (∃ x₁ y₁ x₂ y₂, 
    Ellipse x₁ y₁ ∧ Ellipse x₂ y₂ ∧ 
    Line x₁ y₁ ∧ Line x₂ y₂ ∧
    ((x₂ - x₁)^2 + (y₂ - y₁)^2)^(1/2 : ℝ) = 24/7) ∧
  (∀ k y₀, k ≠ 0 →
    PerpendicularBisector k 0 y₀ →
    -Real.sqrt 3 / 12 ≤ y₀ ∧ y₀ ≤ Real.sqrt 3 / 12) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l2163_216395


namespace NUMINAMATH_CALUDE_cone_volume_l2163_216349

/-- A cone with lateral area √5π and whose unfolded lateral area forms a sector with central angle 2√5π/5 has volume 2π/3 -/
theorem cone_volume (lateral_area : ℝ) (central_angle : ℝ) :
  lateral_area = Real.sqrt 5 * Real.pi →
  central_angle = 2 * Real.sqrt 5 * Real.pi / 5 →
  ∃ (r h : ℝ), 
    r > 0 ∧ h > 0 ∧
    lateral_area = Real.pi * r * Real.sqrt (r^2 + h^2) ∧
    (1/3) * Real.pi * r^2 * h = (2/3) * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_cone_volume_l2163_216349


namespace NUMINAMATH_CALUDE_wendy_packaging_theorem_l2163_216302

/-- Represents the number of chocolates Wendy can package in a given time -/
def chocolates_packaged (packaging_rate : ℕ) (packaging_time : ℕ) (work_time : ℕ) : ℕ :=
  (packaging_rate * 12 * (work_time * 60 / packaging_time))

/-- Proves that Wendy can package 1152 chocolates in 4 hours -/
theorem wendy_packaging_theorem :
  chocolates_packaged 2 5 240 = 1152 := by
  sorry

#eval chocolates_packaged 2 5 240

end NUMINAMATH_CALUDE_wendy_packaging_theorem_l2163_216302


namespace NUMINAMATH_CALUDE_simplify_expression_l2163_216322

theorem simplify_expression : (81^(1/2) - 144^(1/2)) / 3^(1/2) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2163_216322


namespace NUMINAMATH_CALUDE_ice_melting_problem_l2163_216338

theorem ice_melting_problem (initial_volume : ℝ) : 
  initial_volume = 3.2 →
  (1/4) * ((1/4) * initial_volume) = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_ice_melting_problem_l2163_216338


namespace NUMINAMATH_CALUDE_max_distance_between_vectors_l2163_216358

theorem max_distance_between_vectors (x y : ℝ) : 
  x^2 + y^2 = 1 → 
  (∀ a b : ℝ × ℝ, a = (x, y) ∧ b = (1, 2) → 
    ‖a - b‖ ≤ Real.sqrt 5 + 1) ∧
  (∃ a b : ℝ × ℝ, a = (x, y) ∧ b = (1, 2) ∧ 
    ‖a - b‖ = Real.sqrt 5 + 1) := by
  sorry

end NUMINAMATH_CALUDE_max_distance_between_vectors_l2163_216358


namespace NUMINAMATH_CALUDE_fair_queue_size_l2163_216355

/-- Calculates the final number of people in a queue after changes -/
def final_queue_size (initial : ℕ) (left : ℕ) (joined : ℕ) : ℕ :=
  initial - left + joined

/-- Theorem: Given the specific scenario, the final queue size is 6 -/
theorem fair_queue_size : final_queue_size 9 6 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_fair_queue_size_l2163_216355


namespace NUMINAMATH_CALUDE_billy_total_tickets_l2163_216317

/-- The number of times Billy rode the ferris wheel -/
def ferris_rides : ℕ := 7

/-- The number of times Billy rode the bumper cars -/
def bumper_rides : ℕ := 3

/-- The cost in tickets for each ride -/
def ticket_cost_per_ride : ℕ := 5

/-- Theorem: The total number of tickets Billy used is 50 -/
theorem billy_total_tickets : 
  (ferris_rides + bumper_rides) * ticket_cost_per_ride = 50 := by
  sorry

end NUMINAMATH_CALUDE_billy_total_tickets_l2163_216317


namespace NUMINAMATH_CALUDE_happiness_difference_test_l2163_216380

-- Define the data from the problem
def total_observations : ℕ := 1184
def boys_happy : ℕ := 638
def boys_unhappy : ℕ := 128
def girls_happy : ℕ := 372
def girls_unhappy : ℕ := 46
def total_happy : ℕ := 1010
def total_unhappy : ℕ := 174
def total_boys : ℕ := 766
def total_girls : ℕ := 418

-- Define the χ² calculation function
def chi_square : ℚ :=
  (total_observations : ℚ) * (boys_happy * girls_unhappy - boys_unhappy * girls_happy)^2 /
  (total_happy * total_unhappy * total_boys * total_girls)

-- Define the critical values
def critical_value_001 : ℚ := 6635 / 1000
def critical_value_0005 : ℚ := 7879 / 1000

-- Theorem statement
theorem happiness_difference_test :
  (chi_square > critical_value_001) ∧ (chi_square < critical_value_0005) :=
by sorry

end NUMINAMATH_CALUDE_happiness_difference_test_l2163_216380


namespace NUMINAMATH_CALUDE_sample_size_calculation_l2163_216325

theorem sample_size_calculation (total_parts : ℕ) (prob_sampled : ℚ) (n : ℕ) : 
  total_parts = 200 → prob_sampled = 1/4 → n = (total_parts : ℚ) * prob_sampled → n = 50 := by
sorry

end NUMINAMATH_CALUDE_sample_size_calculation_l2163_216325


namespace NUMINAMATH_CALUDE_power_function_through_point_value_l2163_216321

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, x > 0 → f x = x ^ α

-- Theorem statement
theorem power_function_through_point_value :
  ∀ f : ℝ → ℝ,
  isPowerFunction f →
  f 2 = 16 →
  f (Real.sqrt 3) = 9 := by
sorry

end NUMINAMATH_CALUDE_power_function_through_point_value_l2163_216321


namespace NUMINAMATH_CALUDE_smallest_divisible_by_18_and_35_l2163_216390

theorem smallest_divisible_by_18_and_35 : 
  ∀ n : ℕ, n > 0 → (18 ∣ n) → (35 ∣ n) → n ≥ 630 :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_18_and_35_l2163_216390


namespace NUMINAMATH_CALUDE_total_swim_time_l2163_216384

def freestyle : ℕ := 48

def backstroke (f : ℕ) : ℕ := f + 4

def butterfly (b : ℕ) : ℕ := b + 3

def breaststroke (t : ℕ) : ℕ := t + 2

theorem total_swim_time :
  freestyle + backstroke freestyle + butterfly (backstroke freestyle) + breaststroke (butterfly (backstroke freestyle)) = 212 := by
  sorry

end NUMINAMATH_CALUDE_total_swim_time_l2163_216384


namespace NUMINAMATH_CALUDE_probability_of_selecting_male_student_l2163_216331

theorem probability_of_selecting_male_student 
  (total_students : ℕ) 
  (male_students : ℕ) 
  (female_students : ℕ) 
  (h1 : total_students = male_students + female_students)
  (h2 : male_students = 2)
  (h3 : female_students = 3) :
  (male_students : ℚ) / total_students = 2 / 5 := by
sorry

end NUMINAMATH_CALUDE_probability_of_selecting_male_student_l2163_216331


namespace NUMINAMATH_CALUDE_betty_beads_l2163_216305

/-- Given a ratio of red to blue beads and a number of red beads, 
    calculate the number of blue beads -/
def blue_beads (red_ratio blue_ratio red_count : ℕ) : ℕ :=
  (blue_ratio * red_count) / red_ratio

/-- Theorem stating that given 3 red beads for every 2 blue beads,
    and 30 red beads in total, there are 20 blue beads -/
theorem betty_beads : blue_beads 3 2 30 = 20 := by
  sorry

end NUMINAMATH_CALUDE_betty_beads_l2163_216305


namespace NUMINAMATH_CALUDE_max_x_value_l2163_216352

theorem max_x_value (x y z : ℝ) (sum_eq : x + y + z = 6) (prod_sum_eq : x*y + x*z + y*z = 9) :
  x ≤ 4 ∧ ∃ (x₀ y₀ z₀ : ℝ), x₀ + y₀ + z₀ = 6 ∧ x₀*y₀ + x₀*z₀ + y₀*z₀ = 9 ∧ x₀ = 4 :=
by sorry

end NUMINAMATH_CALUDE_max_x_value_l2163_216352


namespace NUMINAMATH_CALUDE_max_value_of_function_l2163_216392

theorem max_value_of_function :
  (∀ x : ℝ, x > 1 → (2*x^2 + 7*x - 1) / (x^2 + 3*x) ≤ 19/9) ∧
  (∃ x : ℝ, x > 1 ∧ (2*x^2 + 7*x - 1) / (x^2 + 3*x) = 19/9) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_function_l2163_216392


namespace NUMINAMATH_CALUDE_f_properties_l2163_216319

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (a^x - a^(-x)) / (a - 1)

-- Theorem statement
theorem f_properties (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x : ℝ, f a (-x) = -f a x) ∧
  StrictMono (f a) :=
by sorry

end

end NUMINAMATH_CALUDE_f_properties_l2163_216319


namespace NUMINAMATH_CALUDE_salt_proportion_is_one_twenty_first_l2163_216345

/-- The proportion of salt in a saltwater solution -/
def salt_proportion (salt_mass : ℚ) (water_mass : ℚ) : ℚ :=
  salt_mass / (salt_mass + water_mass)

/-- Proof that the proportion of salt in the given saltwater solution is 1/21 -/
theorem salt_proportion_is_one_twenty_first :
  let salt_mass : ℚ := 50
  let water_mass : ℚ := 1000
  salt_proportion salt_mass water_mass = 1 / 21 := by
  sorry

end NUMINAMATH_CALUDE_salt_proportion_is_one_twenty_first_l2163_216345


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_l2163_216300

def arithmetic_sequence (a : ℕ → ℝ) (n : ℕ) : Prop :=
  ∀ i j : ℕ, i < j → j ≤ n → a j - a i = (j - i : ℝ) * (a 1 - a 0)

theorem sum_of_x_and_y (a : ℕ → ℝ) (n : ℕ) :
  arithmetic_sequence a n ∧ a 0 = 3 ∧ a n = 33 →
  a (n - 1) + a (n - 2) = 48 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_l2163_216300


namespace NUMINAMATH_CALUDE_special_function_at_zero_l2163_216393

/-- A function satisfying f(x + y) = f(x) + f(xy) for all real x and y -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x + f (x * y)

/-- Theorem: If f is a special function, then f(0) = 0 -/
theorem special_function_at_zero (f : ℝ → ℝ) (h : special_function f) : f 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_special_function_at_zero_l2163_216393


namespace NUMINAMATH_CALUDE_intersection_implies_sum_l2163_216332

/-- Given two graphs that intersect at (3,4) and (7,2), prove that a+c = 10 -/
theorem intersection_implies_sum (a b c d : ℝ) : 
  (∀ x, -|x - (a + 1)| + b = |x - (c - 1)| + (d - 1) → x = 3 ∨ x = 7) →
  -|3 - (a + 1)| + b = |3 - (c - 1)| + (d - 1) →
  -|7 - (a + 1)| + b = |7 - (c - 1)| + (d - 1) →
  -|3 - (a + 1)| + b = 4 →
  -|7 - (a + 1)| + b = 2 →
  |3 - (c - 1)| + (d - 1) = 4 →
  |7 - (c - 1)| + (d - 1) = 2 →
  a + c = 10 := by
sorry

end NUMINAMATH_CALUDE_intersection_implies_sum_l2163_216332


namespace NUMINAMATH_CALUDE_max_gcd_2015_l2163_216318

theorem max_gcd_2015 (x y : ℤ) (h : Int.gcd x y = 1) :
  (∃ a b : ℤ, Int.gcd (a + 2015 * b) (b + 2015 * a) = 4060224) ∧
  (∀ c d : ℤ, Int.gcd (c + 2015 * d) (d + 2015 * c) ≤ 4060224) := by
sorry

end NUMINAMATH_CALUDE_max_gcd_2015_l2163_216318


namespace NUMINAMATH_CALUDE_parabola_tangent_line_l2163_216301

/-- A parabola is tangent to a line if they intersect at exactly one point. -/
def is_tangent (a : ℝ) : Prop :=
  ∃! x : ℝ, a * x^2 + 3 = 2 * x + 1

/-- If the parabola y = ax^2 + 3 is tangent to the line y = 2x + 1, then a = 1/2. -/
theorem parabola_tangent_line (a : ℝ) : is_tangent a → a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_tangent_line_l2163_216301


namespace NUMINAMATH_CALUDE_A_subset_complement_B_l2163_216342

-- Define the universe set S
def S : Finset Char := {'a', 'b', 'c', 'd', 'e'}

-- Define set A
def A : Finset Char := {'a', 'c'}

-- Define set B
def B : Finset Char := {'b', 'e'}

-- Theorem statement
theorem A_subset_complement_B : A ⊆ S \ B := by sorry

end NUMINAMATH_CALUDE_A_subset_complement_B_l2163_216342


namespace NUMINAMATH_CALUDE_valid_pairs_l2163_216368

def is_valid_pair (a b : ℕ) : Prop :=
  (a^2 + b) % (b^2 - a) = 0 ∧ (b^2 + a) % (a^2 - b) = 0

theorem valid_pairs :
  ∀ a b : ℕ, is_valid_pair a b ↔ 
    ((a = 1 ∧ b = 2) ∨ 
     (a = 2 ∧ b = 1) ∨ 
     (a = 2 ∧ b = 2) ∨ 
     (a = 2 ∧ b = 3) ∨ 
     (a = 3 ∧ b = 2) ∨ 
     (a = 3 ∧ b = 3)) :=
by sorry

end NUMINAMATH_CALUDE_valid_pairs_l2163_216368


namespace NUMINAMATH_CALUDE_inequality_proof_l2163_216372

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b ≤ 4) :
  1 / a + 1 / b ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2163_216372


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l2163_216339

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x : ℝ, P x) ↔ (∀ x : ℝ, ¬ P x) := by sorry

theorem negation_of_quadratic_inequality :
  (¬ ∃ x : ℝ, x^2 + 1 > 3*x) ↔ (∀ x : ℝ, x^2 + 1 ≤ 3*x) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l2163_216339


namespace NUMINAMATH_CALUDE_max_trees_bucked_l2163_216371

/-- Represents the energy and tree-bucking strategy over time -/
structure BuckingStrategy where
  restTime : ℕ
  initialEnergy : ℕ
  timePeriod : ℕ

/-- Calculates the total number of trees bucked given a strategy -/
def totalTreesBucked (s : BuckingStrategy) : ℕ :=
  let buckingTime := s.timePeriod - s.restTime
  let finalEnergy := s.initialEnergy + s.restTime - buckingTime + 1
  (buckingTime * (s.initialEnergy + s.restTime + finalEnergy)) / 2

/-- The main theorem to prove -/
theorem max_trees_bucked :
  ∃ (s : BuckingStrategy),
    s.initialEnergy = 100 ∧
    s.timePeriod = 60 ∧
    (∀ (t : BuckingStrategy),
      t.initialEnergy = 100 ∧
      t.timePeriod = 60 →
      totalTreesBucked t ≤ totalTreesBucked s) ∧
    totalTreesBucked s = 4293 := by
  sorry


end NUMINAMATH_CALUDE_max_trees_bucked_l2163_216371


namespace NUMINAMATH_CALUDE_lcm_of_8_9_5_10_l2163_216375

theorem lcm_of_8_9_5_10 : Nat.lcm 8 (Nat.lcm 9 (Nat.lcm 5 10)) = 360 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_8_9_5_10_l2163_216375


namespace NUMINAMATH_CALUDE_parabola_y_intercepts_l2163_216348

/-- The number of y-intercepts of the parabola x = 3y^2 - 4y + 5 -/
def num_y_intercepts : ℕ := 0

/-- The parabola equation: x = 3y^2 - 4y + 5 -/
def parabola_equation (y : ℝ) : ℝ := 3 * y^2 - 4 * y + 5

theorem parabola_y_intercepts :
  (∀ y : ℝ, parabola_equation y ≠ 0) ∧ num_y_intercepts = 0 := by sorry

end NUMINAMATH_CALUDE_parabola_y_intercepts_l2163_216348


namespace NUMINAMATH_CALUDE_f_difference_f_equation_solution_l2163_216303

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 1

-- Theorem 1
theorem f_difference (a : ℝ) : f a - f (a + 1) = -2 * a - 1 := by
  sorry

-- Theorem 2
theorem f_equation_solution : {x : ℝ | f x = x + 3} = {-1, 2} := by
  sorry

end NUMINAMATH_CALUDE_f_difference_f_equation_solution_l2163_216303


namespace NUMINAMATH_CALUDE_point_on_line_iff_concyclic_l2163_216330

-- Define the points
variable (A B C D E F M : Point)

-- Define the concyclic property
def are_concyclic (P Q R S : Point) : Prop := sorry

-- Define the collinear property
def are_collinear (P Q R : Point) : Prop := sorry

-- State the theorem
theorem point_on_line_iff_concyclic :
  (are_concyclic D M F B) →
  (are_concyclic M A E F) →
  (are_collinear A M D) ↔ (are_concyclic B F E C) := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_iff_concyclic_l2163_216330


namespace NUMINAMATH_CALUDE_B_power_six_equals_81B_l2163_216312

def B : Matrix (Fin 2) (Fin 2) ℝ := !![1, -1; 4, 2]

theorem B_power_six_equals_81B : B^6 = 81 • B := by sorry

end NUMINAMATH_CALUDE_B_power_six_equals_81B_l2163_216312


namespace NUMINAMATH_CALUDE_jason_initial_cards_l2163_216333

/-- The number of Pokemon cards Jason had initially -/
def initial_cards : ℕ := sorry

/-- The number of Pokemon cards Benny bought from Jason -/
def cards_bought : ℕ := 2

/-- The number of Pokemon cards Jason has left -/
def cards_left : ℕ := 1

/-- Theorem stating that Jason's initial number of Pokemon cards was 3 -/
theorem jason_initial_cards : initial_cards = 3 := by sorry

end NUMINAMATH_CALUDE_jason_initial_cards_l2163_216333


namespace NUMINAMATH_CALUDE_quadratic_roots_equality_l2163_216383

/-- 
Given two quadratic polynomials f(x) = x^2 - 6x + 4a and g(x) = x^2 + ax + 6,
prove that a = -12 is the only value that satisfies:
1. Both f(x) and g(x) have two distinct real roots.
2. The sum of the squares of the roots of f(x) equals the sum of the squares of the roots of g(x).
-/
theorem quadratic_roots_equality (a : ℝ) : 
  (∃ x₁ x₂ y₁ y₂ : ℝ, x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧
    x₁^2 - 6*x₁ + 4*a = 0 ∧ x₂^2 - 6*x₂ + 4*a = 0 ∧
    y₁^2 + a*y₁ + 6 = 0 ∧ y₂^2 + a*y₂ + 6 = 0 ∧
    x₁^2 + x₂^2 = y₁^2 + y₂^2) ↔ 
  a = -12 := by sorry

end NUMINAMATH_CALUDE_quadratic_roots_equality_l2163_216383


namespace NUMINAMATH_CALUDE_composition_value_l2163_216378

/-- Given two functions g and f, prove that f(g(3)) = 29 -/
theorem composition_value (g f : ℝ → ℝ) 
  (hg : ∀ x, g x = x^2) 
  (hf : ∀ x, f x = 3*x + 2) : 
  f (g 3) = 29 := by
  sorry

end NUMINAMATH_CALUDE_composition_value_l2163_216378


namespace NUMINAMATH_CALUDE_g_3_equals_25_l2163_216316

-- Define the function g
def g (p q r : ℝ) (x : ℝ) : ℝ := p * x^7 + q * x^3 + r * x + 7

-- State the theorem
theorem g_3_equals_25 (p q r : ℝ) :
  (g p q r (-3) = -11) →
  (∀ x, g p q r x + g p q r (-x) = 14) →
  g p q r 3 = 25 := by
sorry

end NUMINAMATH_CALUDE_g_3_equals_25_l2163_216316


namespace NUMINAMATH_CALUDE_unique_point_in_S_l2163_216323

def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 > 0 ∧ p.2 > 0 ∧ Real.log (p.1^3 + (1/3) * p.2^3 + 1/9) = Real.log p.1 + Real.log p.2}

theorem unique_point_in_S : ∃! p : ℝ × ℝ, p ∈ S := by
  sorry

end NUMINAMATH_CALUDE_unique_point_in_S_l2163_216323


namespace NUMINAMATH_CALUDE_smallest_better_discount_l2163_216391

theorem smallest_better_discount (x : ℝ) (h : x > 0) : ∃ (n : ℕ), n = 38 ∧ 
  (∀ (m : ℕ), m < n → 
    ((1 - m / 100) * x < (1 - 0.2) * (1 - 0.2) * x ∨
     (1 - m / 100) * x < (1 - 0.12) * (1 - 0.12) * (1 - 0.12) * x ∨
     (1 - m / 100) * x < (1 - 0.3) * (1 - 0.1) * x)) ∧
  (1 - n / 100) * x > (1 - 0.2) * (1 - 0.2) * x ∧
  (1 - n / 100) * x > (1 - 0.12) * (1 - 0.12) * (1 - 0.12) * x ∧
  (1 - n / 100) * x > (1 - 0.3) * (1 - 0.1) * x :=
sorry

end NUMINAMATH_CALUDE_smallest_better_discount_l2163_216391


namespace NUMINAMATH_CALUDE_inequalities_theorem_l2163_216386

theorem inequalities_theorem (a b c : ℝ) 
  (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) : 
  (b / a > c / a) ∧ ((b - a) / c > 0) ∧ ((a - c) / (a * c) < 0) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_theorem_l2163_216386


namespace NUMINAMATH_CALUDE_credit_card_balance_ratio_l2163_216314

theorem credit_card_balance_ratio : 
  ∀ (gold_limit : ℝ) (gold_balance : ℝ) (platinum_balance : ℝ),
  gold_limit > 0 →
  platinum_balance = (1/8) * (2 * gold_limit) →
  0.7083333333333334 * (2 * gold_limit) = 2 * gold_limit - (platinum_balance + gold_balance) →
  gold_balance / gold_limit = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_credit_card_balance_ratio_l2163_216314


namespace NUMINAMATH_CALUDE_symmetric_sequence_second_term_l2163_216344

def is_symmetric (s : Fin 21 → ℕ) : Prop :=
  ∀ i : Fin 21, s i = s (20 - i)

def is_arithmetic_sequence (s : Fin 11 → ℕ) (a d : ℕ) : Prop :=
  ∀ i : Fin 11, s i = a + i * d

theorem symmetric_sequence_second_term 
  (c : Fin 21 → ℕ) 
  (h_sym : is_symmetric c) 
  (h_arith : is_arithmetic_sequence (fun i => c (i + 10)) 1 2) : 
  c 1 = 19 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_sequence_second_term_l2163_216344


namespace NUMINAMATH_CALUDE_tetrahedron_section_theorem_l2163_216334

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a tetrahedron ABCD -/
structure Tetrahedron where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Represents a plane defined by three points -/
structure Plane where
  P : Point3D
  Q : Point3D
  R : Point3D

/-- Check if a point is the midpoint of a line segment -/
def isMidpoint (M : Point3D) (A : Point3D) (D : Point3D) : Prop :=
  M.x = (A.x + D.x) / 2 ∧ M.y = (A.y + D.y) / 2 ∧ M.z = (A.z + D.z) / 2

/-- Check if a point is on the extension of a line segment -/
def isOnExtension (N : Point3D) (A : Point3D) (B : Point3D) : Prop :=
  ∃ t : ℝ, t > 1 ∧ N.x = A.x + t * (B.x - A.x) ∧
                 N.y = A.y + t * (B.y - A.y) ∧
                 N.z = A.z + t * (B.z - A.z)

/-- Calculate the ratio in which a plane divides a line segment -/
def divisionRatio (P : Plane) (A : Point3D) (B : Point3D) : ℝ × ℝ :=
  sorry

theorem tetrahedron_section_theorem (ABCD : Tetrahedron) (M N K : Point3D) :
  isMidpoint M ABCD.A ABCD.D →
  isOnExtension N ABCD.A ABCD.B →
  isOnExtension K ABCD.A ABCD.C →
  (N.x - ABCD.B.x)^2 + (N.y - ABCD.B.y)^2 + (N.z - ABCD.B.z)^2 =
    (ABCD.B.x - ABCD.A.x)^2 + (ABCD.B.y - ABCD.A.y)^2 + (ABCD.B.z - ABCD.A.z)^2 →
  (K.x - ABCD.C.x)^2 + (K.y - ABCD.C.y)^2 + (K.z - ABCD.C.z)^2 =
    4 * ((ABCD.C.x - ABCD.A.x)^2 + (ABCD.C.y - ABCD.A.y)^2 + (ABCD.C.z - ABCD.A.z)^2) →
  let P : Plane := {P := M, Q := N, R := K}
  divisionRatio P ABCD.D ABCD.B = (2, 1) ∧
  divisionRatio P ABCD.D ABCD.C = (3, 2) :=
by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_section_theorem_l2163_216334


namespace NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_2_3_5_l2163_216327

theorem smallest_perfect_square_divisible_by_2_3_5 :
  ∀ n : ℕ, n > 0 → n^2 % 2 = 0 → n^2 % 3 = 0 → n^2 % 5 = 0 → n^2 ≥ 900 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_2_3_5_l2163_216327


namespace NUMINAMATH_CALUDE_equation_solution_l2163_216376

theorem equation_solution : ∃! x : ℝ, (3 / (x - 3) = 1 / (x - 1)) ∧ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2163_216376


namespace NUMINAMATH_CALUDE_sticks_remaining_proof_l2163_216396

/-- The number of sticks originally in the yard -/
def original_sticks : ℕ := 99

/-- The number of sticks Will picked up -/
def picked_up_sticks : ℕ := 38

/-- The number of sticks left after Will picked up some -/
def remaining_sticks : ℕ := original_sticks - picked_up_sticks

theorem sticks_remaining_proof : remaining_sticks = 61 := by
  sorry

end NUMINAMATH_CALUDE_sticks_remaining_proof_l2163_216396


namespace NUMINAMATH_CALUDE_not_both_perfect_squares_l2163_216354

theorem not_both_perfect_squares (x y : ℕ+) (h1 : Nat.gcd x.val y.val = 1) 
  (h2 : ∃ k : ℕ, x.val + 3 * y.val^2 = k^2) : 
  ¬ ∃ z : ℕ, x.val^2 + 9 * y.val^4 = z^2 := by
  sorry

end NUMINAMATH_CALUDE_not_both_perfect_squares_l2163_216354


namespace NUMINAMATH_CALUDE_bailey_points_l2163_216328

/-- 
Given four basketball players and their scoring relationships, 
prove that Bailey scored 14 points when the team's total score was 54.
-/
theorem bailey_points (bailey akiko michiko chandra : ℕ) : 
  chandra = 2 * akiko →
  akiko = michiko + 4 →
  michiko = bailey / 2 →
  bailey + akiko + michiko + chandra = 54 →
  bailey = 14 := by
sorry

end NUMINAMATH_CALUDE_bailey_points_l2163_216328


namespace NUMINAMATH_CALUDE_passengers_from_other_continents_l2163_216374

theorem passengers_from_other_continents 
  (total : ℕ) 
  (north_america : ℚ)
  (europe : ℚ)
  (africa : ℚ)
  (asia : ℚ)
  (h1 : total = 108)
  (h2 : north_america = 1 / 12)
  (h3 : europe = 1 / 4)
  (h4 : africa = 1 / 9)
  (h5 : asia = 1 / 6)
  : ℕ := by
  sorry

end NUMINAMATH_CALUDE_passengers_from_other_continents_l2163_216374


namespace NUMINAMATH_CALUDE_selling_price_for_target_profit_impossibility_of_daily_profit_maximum_profit_l2163_216310

-- Define the variables and constants
variable (x : ℝ)  -- price increase
def original_price : ℝ := 40
def cost_price : ℝ := 30
def initial_sales : ℝ := 600
def sales_decrease_rate : ℝ := 10

-- Define the profit function
def profit (x : ℝ) : ℝ :=
  (original_price + x - cost_price) * (initial_sales - sales_decrease_rate * x)

-- Theorem 1: Selling price for 10,000 yuan monthly profit
theorem selling_price_for_target_profit :
  ∃ x₁ x₂, x₁ ≠ x₂ ∧ profit x₁ = 10000 ∧ profit x₂ = 10000 ∧
  (x₁ + original_price = 80 ∨ x₁ + original_price = 50) ∧
  (x₂ + original_price = 80 ∨ x₂ + original_price = 50) :=
sorry

-- Theorem 2: Impossibility of 15,000 yuan daily profit
theorem impossibility_of_daily_profit :
  ¬∃ x, profit x = 15000 * 30 :=
sorry

-- Theorem 3: Price and value for maximum profit
theorem maximum_profit :
  ∃ x_max, ∀ x, profit x ≤ profit x_max ∧
  x_max + original_price = 65 ∧ profit x_max = 12250 :=
sorry

end NUMINAMATH_CALUDE_selling_price_for_target_profit_impossibility_of_daily_profit_maximum_profit_l2163_216310


namespace NUMINAMATH_CALUDE_central_angle_is_45_degrees_l2163_216366

/-- Represents a circular dartboard divided into sectors -/
structure Dartboard where
  smallSectors : Nat
  largeSectors : Nat
  smallSectorProbability : ℝ

/-- Calculate the central angle of a smaller sector in degrees -/
def centralAngleSmallSector (d : Dartboard) : ℝ :=
  360 * d.smallSectorProbability

/-- Theorem: The central angle of a smaller sector is 45° for the given dartboard -/
theorem central_angle_is_45_degrees (d : Dartboard) 
  (h1 : d.smallSectors = 3)
  (h2 : d.largeSectors = 1)
  (h3 : d.smallSectorProbability = 1/8) :
  centralAngleSmallSector d = 45 := by
  sorry

end NUMINAMATH_CALUDE_central_angle_is_45_degrees_l2163_216366


namespace NUMINAMATH_CALUDE_equal_expressions_l2163_216377

theorem equal_expressions : 
  (2^3 ≠ 3^2) ∧ 
  (-3^3 = (-3)^3) ∧ 
  (-2^2 ≠ (-2)^2) ∧ 
  (-|-2| ≠ -(-2)) := by
  sorry

end NUMINAMATH_CALUDE_equal_expressions_l2163_216377


namespace NUMINAMATH_CALUDE_find_number_l2163_216359

theorem find_number : ∃ x : ℝ, 4 * x - 23 = 33 ∧ x = 14 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l2163_216359


namespace NUMINAMATH_CALUDE_z_purely_imaginary_iff_a_eq_neg_three_l2163_216361

/-- A complex number z is purely imaginary if its real part is zero and its imaginary part is non-zero -/
def isPurelyImaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

/-- The complex number z defined in terms of a real number a -/
def z (a : ℝ) : ℂ := Complex.mk (a^2 + 2*a - 3) (a - 1)

/-- Theorem stating that z is purely imaginary if and only if a = -3 -/
theorem z_purely_imaginary_iff_a_eq_neg_three (a : ℝ) :
  isPurelyImaginary (z a) ↔ a = -3 := by
  sorry

end NUMINAMATH_CALUDE_z_purely_imaginary_iff_a_eq_neg_three_l2163_216361


namespace NUMINAMATH_CALUDE_girls_in_first_year_l2163_216397

theorem girls_in_first_year 
  (total_students : ℕ) 
  (sample_size : ℕ) 
  (boys_in_sample : ℕ) 
  (h1 : total_students = 2400) 
  (h2 : sample_size = 80) 
  (h3 : boys_in_sample = 42) : 
  ℕ := by
  sorry

#check girls_in_first_year

end NUMINAMATH_CALUDE_girls_in_first_year_l2163_216397


namespace NUMINAMATH_CALUDE_rectangle_triangle_equality_l2163_216315

theorem rectangle_triangle_equality (AB AD DC : ℝ) (h1 : AB = 4) (h2 : AD = 8) (h3 : DC = 4) :
  let ABCD_area := AB * AD
  let DCE_area := (1 / 2) * DC * CE
  let CE := 2 * ABCD_area / DC
  ABCD_area = DCE_area → DE = 4 * Real.sqrt 17 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_triangle_equality_l2163_216315


namespace NUMINAMATH_CALUDE_wood_gathering_proof_l2163_216326

/-- The number of pieces of wood that can be contained in one sack -/
def pieces_per_sack : ℕ := 20

/-- The number of sacks filled -/
def num_sacks : ℕ := 4

/-- The total number of pieces of wood gathered -/
def total_pieces : ℕ := pieces_per_sack * num_sacks

theorem wood_gathering_proof :
  total_pieces = 80 :=
by sorry

end NUMINAMATH_CALUDE_wood_gathering_proof_l2163_216326


namespace NUMINAMATH_CALUDE_collinear_probability_l2163_216351

/-- The number of dots in one side of the square grid -/
def grid_size : ℕ := 5

/-- The total number of dots in the grid -/
def total_dots : ℕ := grid_size * grid_size

/-- The number of dots to be chosen -/
def chosen_dots : ℕ := 4

/-- The number of ways to choose 4 dots from the grid -/
def total_choices : ℕ := Nat.choose total_dots chosen_dots

/-- The number of collinear sets of 4 dots in the grid -/
def collinear_sets : ℕ := 28

/-- The probability of choosing 4 collinear dots from a 5x5 grid -/
theorem collinear_probability : 
  (collinear_sets : ℚ) / total_choices = 14 / 6325 := by sorry

end NUMINAMATH_CALUDE_collinear_probability_l2163_216351


namespace NUMINAMATH_CALUDE_perfect_square_condition_l2163_216347

theorem perfect_square_condition (W L : ℤ) : 
  (1000 < W) → (W < 2000) → (L > 1) → (W = 2 * L^3) → 
  (∃ m : ℤ, W = m^2) → (L = 8) :=
sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l2163_216347


namespace NUMINAMATH_CALUDE_age_ratio_sachin_rahul_l2163_216353

/-- Given that Sachin is 5 years old and 7 years younger than Rahul, 
    prove that the ratio of Sachin's age to Rahul's age is 5:12. -/
theorem age_ratio_sachin_rahul :
  let sachin_age : ℕ := 5
  let age_difference : ℕ := 7
  let rahul_age : ℕ := sachin_age + age_difference
  (sachin_age : ℚ) / (rahul_age : ℚ) = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_sachin_rahul_l2163_216353


namespace NUMINAMATH_CALUDE_geometric_sequence_between_9_and_243_l2163_216382

theorem geometric_sequence_between_9_and_243 :
  ∃ (a b : ℝ), 9 < a ∧ a < b ∧ b < 243 ∧
  (9 / a = a / b) ∧ (a / b = b / 243) ∧
  a = 27 ∧ b = 81 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_between_9_and_243_l2163_216382


namespace NUMINAMATH_CALUDE_sandy_total_earnings_l2163_216369

def monday_earnings : ℚ := 12 * 0.5 + 5 * 0.25 + 10 * 0.1
def tuesday_earnings : ℚ := 8 * 0.5 + 15 * 0.25 + 5 * 0.1
def wednesday_earnings : ℚ := 3 * 1 + 4 * 0.5 + 10 * 0.25 + 7 * 0.05
def thursday_earnings : ℚ := 5 * 1 + 6 * 0.5 + 8 * 0.25 + 5 * 0.1 + 12 * 0.05
def friday_earnings : ℚ := 2 * 1 + 7 * 0.5 + 20 * 0.05 + 25 * 0.1

theorem sandy_total_earnings :
  monday_earnings + tuesday_earnings + wednesday_earnings + thursday_earnings + friday_earnings = 44.45 := by
  sorry

end NUMINAMATH_CALUDE_sandy_total_earnings_l2163_216369


namespace NUMINAMATH_CALUDE_problem_statement_l2163_216365

theorem problem_statement (x : ℚ) (h : 5 * x - 8 = 15 * x - 2) : 5 * (x - 3) = -18 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2163_216365


namespace NUMINAMATH_CALUDE_race_distance_difference_l2163_216370

theorem race_distance_difference (lingling_distance mingming_distance : ℝ) 
  (h1 : lingling_distance = 380.5)
  (h2 : mingming_distance = 405.9) : 
  mingming_distance - lingling_distance = 25.4 := by
  sorry

end NUMINAMATH_CALUDE_race_distance_difference_l2163_216370


namespace NUMINAMATH_CALUDE_sales_balance_l2163_216385

/-- Represents the sales increase of product C as a percentage -/
def sales_increase_C : ℝ := 0.3

/-- Represents the proportion of total sales from product C last year -/
def last_year_C_proportion : ℝ := 0.4

/-- Represents the decrease in sales for products A and B -/
def sales_decrease_AB : ℝ := 0.2

/-- Represents the proportion of total sales from products A and B last year -/
def last_year_AB_proportion : ℝ := 1 - last_year_C_proportion

theorem sales_balance :
  last_year_C_proportion * (1 + sales_increase_C) + 
  last_year_AB_proportion * (1 - sales_decrease_AB) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sales_balance_l2163_216385


namespace NUMINAMATH_CALUDE_square_value_l2163_216307

theorem square_value (square x : ℤ) 
  (h1 : square + x = 80)
  (h2 : 3 * (square + x) - 2 * x = 164) : 
  square = 42 := by
sorry

end NUMINAMATH_CALUDE_square_value_l2163_216307


namespace NUMINAMATH_CALUDE_sum_of_integers_minus15_to_5_l2163_216343

-- Define the range of integers
def lower_bound : Int := -15
def upper_bound : Int := 5

-- Define the sum of integers function
def sum_of_integers (a b : Int) : Int :=
  let n := b - a + 1
  let avg := (a + b) / 2
  n * avg

-- Theorem statement
theorem sum_of_integers_minus15_to_5 :
  sum_of_integers lower_bound upper_bound = -105 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_minus15_to_5_l2163_216343


namespace NUMINAMATH_CALUDE_line_parallel_to_parallel_planes_l2163_216336

-- Define the concept of a plane
structure Plane :=
  (p : Set (ℝ × ℝ × ℝ))

-- Define the concept of a line
structure Line :=
  (l : Set (ℝ × ℝ × ℝ))

-- Define parallel relationship between planes
def parallel_planes (p1 p2 : Plane) : Prop := sorry

-- Define parallel relationship between a line and a plane
def parallel_line_plane (l : Line) (p : Plane) : Prop := sorry

-- Define when a line is within a plane
def line_within_plane (l : Line) (p : Plane) : Prop := sorry

-- Theorem statement
theorem line_parallel_to_parallel_planes 
  (p1 p2 : Plane) (l : Line) 
  (h1 : parallel_planes p1 p2) 
  (h2 : parallel_line_plane l p1) :
  parallel_line_plane l p2 ∨ line_within_plane l p2 := 
sorry

end NUMINAMATH_CALUDE_line_parallel_to_parallel_planes_l2163_216336


namespace NUMINAMATH_CALUDE_expanded_parallelepiped_volume_l2163_216398

/-- The volume of a set of points inside or within one unit of a rectangular parallelepiped -/
def volume_expanded_parallelepiped (a b c : ℝ) : ℝ :=
  (a + 2) * (b + 2) * (c + 2) - (a * b * c)

/-- Represents the condition that two natural numbers are coprime -/
def coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

theorem expanded_parallelepiped_volume 
  (m n p : ℕ) 
  (h_positive : m > 0 ∧ n > 0 ∧ p > 0) 
  (h_coprime : coprime n p) 
  (h_volume : volume_expanded_parallelepiped 2 3 4 = (m + n * Real.pi) / p) :
  m + n + p = 262 := by
  sorry

end NUMINAMATH_CALUDE_expanded_parallelepiped_volume_l2163_216398


namespace NUMINAMATH_CALUDE_algebra_textbooks_count_l2163_216379

theorem algebra_textbooks_count : ∃ (x y n : ℕ), 
  x * n + y = 2015 ∧ 
  y * n + x = 1580 ∧ 
  n > 0 ∧ 
  y = 287 := by
  sorry

end NUMINAMATH_CALUDE_algebra_textbooks_count_l2163_216379


namespace NUMINAMATH_CALUDE_square_expression_l2163_216329

theorem square_expression (x : ℝ) (hx : x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1) :
  (1 / (1 / x - 1 / (x + 1)) - x = x^2) ∧
  (1 / (1 / (x - 1) - 1 / x) + x = x^2) := by
  sorry

end NUMINAMATH_CALUDE_square_expression_l2163_216329


namespace NUMINAMATH_CALUDE_two_roots_implies_a_greater_than_e_l2163_216363

-- Define the function f(x) = x / ln(x)
noncomputable def f (x : ℝ) : ℝ := x / Real.log x

-- State the theorem
theorem two_roots_implies_a_greater_than_e (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ a * Real.log x = x ∧ a * Real.log y = y) → a > Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_two_roots_implies_a_greater_than_e_l2163_216363


namespace NUMINAMATH_CALUDE_captain_birth_year_is_1938_l2163_216341

/-- Represents the ages of the crew members -/
structure CrewAges where
  sailor : ℕ
  cook : ℕ
  engineer : ℕ

/-- The conditions given in the problem -/
def satisfiesConditions (ages : CrewAges) : Prop :=
  Odd ages.cook ∧
  ¬Odd ages.sailor ∧
  ¬Odd ages.engineer ∧
  ages.engineer = ages.sailor + 4 ∧
  ages.cook = 3 * (ages.sailor / 2) ∧
  ages.sailor = 2 * (ages.sailor / 2)

/-- The captain's birth year is the LCM of the crew's ages -/
def captainBirthYear (ages : CrewAges) : ℕ :=
  Nat.lcm ages.sailor (Nat.lcm ages.cook ages.engineer)

/-- The main theorem stating that the captain's birth year is 1938 -/
theorem captain_birth_year_is_1938 :
  ∃ ages : CrewAges, satisfiesConditions ages ∧ captainBirthYear ages = 1938 :=
sorry

end NUMINAMATH_CALUDE_captain_birth_year_is_1938_l2163_216341


namespace NUMINAMATH_CALUDE_brownie_problem_l2163_216357

theorem brownie_problem (initial_brownies : ℕ) 
  (h1 : initial_brownies = 16) 
  (children_ate_percent : ℚ) 
  (h2 : children_ate_percent = 1/4) 
  (family_ate_percent : ℚ) 
  (h3 : family_ate_percent = 1/2) 
  (lorraine_ate : ℕ) 
  (h4 : lorraine_ate = 1) : 
  initial_brownies - 
  (initial_brownies * children_ate_percent).floor - 
  ((initial_brownies - (initial_brownies * children_ate_percent).floor) * family_ate_percent).floor - 
  lorraine_ate = 5 := by
sorry


end NUMINAMATH_CALUDE_brownie_problem_l2163_216357


namespace NUMINAMATH_CALUDE_floor_sqrt_30_squared_l2163_216362

theorem floor_sqrt_30_squared : ⌊Real.sqrt 30⌋^2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_30_squared_l2163_216362


namespace NUMINAMATH_CALUDE_exists_m_n_for_any_d_l2163_216388

theorem exists_m_n_for_any_d (d : ℤ) : ∃ (m n : ℤ), d = (n - 2*m + 1) / (m^2 - n) := by
  sorry

end NUMINAMATH_CALUDE_exists_m_n_for_any_d_l2163_216388


namespace NUMINAMATH_CALUDE_remainder_problem_l2163_216356

theorem remainder_problem (N : ℤ) (h : N % 221 = 43) : N % 17 = 9 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2163_216356


namespace NUMINAMATH_CALUDE_polynomial_real_root_iff_b_in_set_l2163_216306

/-- The polynomial in question -/
def polynomial (b x : ℝ) : ℝ := x^5 + b*x^4 - x^3 + b*x^2 - x + b

/-- The set of b values for which the polynomial has at least one real root -/
def valid_b_set : Set ℝ := Set.Iic (-1) ∪ Set.Ici 1

theorem polynomial_real_root_iff_b_in_set (b : ℝ) :
  (∃ x : ℝ, polynomial b x = 0) ↔ b ∈ valid_b_set := by sorry

end NUMINAMATH_CALUDE_polynomial_real_root_iff_b_in_set_l2163_216306


namespace NUMINAMATH_CALUDE_orange_bags_l2163_216360

def total_weight : ℝ := 45.0
def bag_capacity : ℝ := 23.0

theorem orange_bags : ⌊total_weight / bag_capacity⌋ = 1 := by sorry

end NUMINAMATH_CALUDE_orange_bags_l2163_216360


namespace NUMINAMATH_CALUDE_choose_leaders_count_l2163_216311

/-- A club with members divided by gender and class -/
structure Club where
  total_members : ℕ
  boys : ℕ
  girls : ℕ
  classes : ℕ
  boys_per_class : ℕ
  girls_per_class : ℕ

/-- The number of ways to choose a president and vice-president -/
def choose_leaders (c : Club) : ℕ := sorry

/-- The specific club configuration -/
def my_club : Club := {
  total_members := 24,
  boys := 12,
  girls := 12,
  classes := 2,
  boys_per_class := 6,
  girls_per_class := 6
}

/-- Theorem stating the number of ways to choose leaders for the given club -/
theorem choose_leaders_count : choose_leaders my_club = 144 := by sorry

end NUMINAMATH_CALUDE_choose_leaders_count_l2163_216311


namespace NUMINAMATH_CALUDE_inverse_proportion_example_l2163_216335

/-- Two real numbers are inversely proportional if their product is constant. -/
def InverselyProportional (x y : ℝ → ℝ) : Prop :=
  ∃ c : ℝ, ∀ t : ℝ, x t * y t = c

theorem inverse_proportion_example :
  ∀ x y : ℝ → ℝ,
  InverselyProportional x y →
  x 5 = 40 →
  x 10 = 20 := by
sorry

end NUMINAMATH_CALUDE_inverse_proportion_example_l2163_216335


namespace NUMINAMATH_CALUDE_cricket_players_count_l2163_216337

/-- The number of cricket players in a games hour -/
def cricket_players (total_players hockey_players football_players softball_players : ℕ) : ℕ :=
  total_players - (hockey_players + football_players + softball_players)

/-- Theorem: There are 12 cricket players present in the ground -/
theorem cricket_players_count :
  cricket_players 50 17 11 10 = 12 := by
  sorry

end NUMINAMATH_CALUDE_cricket_players_count_l2163_216337


namespace NUMINAMATH_CALUDE_tourist_distribution_count_l2163_216309

def num_guides : ℕ := 3
def num_tourists : ℕ := 8

theorem tourist_distribution_count :
  (3^8 : ℕ) - num_guides * (2^8 : ℕ) + (num_guides.choose 2) * (1^8 : ℕ) = 5796 :=
sorry

end NUMINAMATH_CALUDE_tourist_distribution_count_l2163_216309


namespace NUMINAMATH_CALUDE_prob_sum_greater_than_seven_l2163_216373

/-- The number of faces on each die -/
def numFaces : ℕ := 6

/-- The total number of possible outcomes when rolling two dice -/
def totalOutcomes : ℕ := numFaces * numFaces

/-- The number of ways to roll a sum of 7 or less with two dice -/
def waysToRollSevenOrLess : ℕ := 21

/-- The probability of rolling a sum greater than 7 with two dice -/
def probSumGreaterThanSeven : ℚ := 5 / 12

/-- Theorem stating that the probability of rolling a sum greater than 7 with two fair six-sided dice is 5/12 -/
theorem prob_sum_greater_than_seven :
  probSumGreaterThanSeven = 1 - (waysToRollSevenOrLess : ℚ) / totalOutcomes := by
  sorry

end NUMINAMATH_CALUDE_prob_sum_greater_than_seven_l2163_216373


namespace NUMINAMATH_CALUDE_simplify_expression_l2163_216340

theorem simplify_expression : 0.3 * 0.8 + 0.1 * 0.5 = 0.29 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2163_216340


namespace NUMINAMATH_CALUDE_equation_solution_l2163_216308

theorem equation_solution (m n : ℚ) : 
  (m * 1 + n * 1 = 6) → 
  (m * 2 + n * (-1) = 6) → 
  (m = 4 ∧ n = 2) := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l2163_216308


namespace NUMINAMATH_CALUDE_dice_probability_l2163_216389

/-- The probability of all dice showing the same number -/
def probability : ℝ := 0.0007716049382716049

/-- The number of faces on each die -/
def faces : ℕ := 6

/-- The number of dice thrown -/
def num_dice : ℕ := 5

theorem dice_probability :
  (1 / faces : ℝ) ^ (num_dice - 1) = probability := by sorry

end NUMINAMATH_CALUDE_dice_probability_l2163_216389


namespace NUMINAMATH_CALUDE_cyclist_rate_problem_l2163_216367

/-- Prove that given two cyclists A and B traveling between Newton and Kingston,
    with the given conditions, the rate of cyclist A is 10 mph. -/
theorem cyclist_rate_problem (rate_A rate_B : ℝ) : 
  rate_B = rate_A + 5 →                   -- B travels 5 mph faster than A
  50 / rate_A = (50 + 10) / rate_B →      -- Time for A to travel 40 miles equals time for B to travel 60 miles
  rate_A = 10 := by
sorry

end NUMINAMATH_CALUDE_cyclist_rate_problem_l2163_216367


namespace NUMINAMATH_CALUDE_cos_315_degrees_l2163_216313

theorem cos_315_degrees : Real.cos (315 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_315_degrees_l2163_216313


namespace NUMINAMATH_CALUDE_markup_calculation_l2163_216350

/-- The markup required for an article with given purchase price, overhead percentage, and desired net profit. -/
def required_markup (purchase_price : ℝ) (overhead_percent : ℝ) (net_profit : ℝ) : ℝ :=
  purchase_price * overhead_percent + net_profit

/-- Theorem stating that the required markup for the given conditions is $34.80 -/
theorem markup_calculation :
  required_markup 48 0.35 18 = 34.80 := by
  sorry

end NUMINAMATH_CALUDE_markup_calculation_l2163_216350


namespace NUMINAMATH_CALUDE_sqrt_expression_sum_l2163_216394

theorem sqrt_expression_sum (a b c : ℤ) : 
  (64 + 24 * Real.sqrt 3 : ℝ) = (a + b * Real.sqrt c)^2 →
  c > 0 →
  (∀ n : ℕ, n > 1 → ¬(∃ m : ℕ, c = n^2 * m)) →
  a + b + c = 11 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_sum_l2163_216394


namespace NUMINAMATH_CALUDE_rationalize_denominator_l2163_216387

theorem rationalize_denominator : (1 : ℝ) / (Real.sqrt 3 - 1) = (Real.sqrt 3 + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l2163_216387


namespace NUMINAMATH_CALUDE_telescope_purchase_sum_l2163_216364

/-- The sum of Joan and Karl's telescope purchases -/
def sum_of_purchases (joan_price karl_price : ℕ) : ℕ :=
  joan_price + karl_price

/-- Theorem stating the sum of Joan and Karl's telescope purchases -/
theorem telescope_purchase_sum :
  ∀ (joan_price karl_price : ℕ),
    joan_price = 158 →
    2 * joan_price = karl_price + 74 →
    sum_of_purchases joan_price karl_price = 400 := by
  sorry

end NUMINAMATH_CALUDE_telescope_purchase_sum_l2163_216364


namespace NUMINAMATH_CALUDE_max_students_per_dentist_l2163_216320

theorem max_students_per_dentist (num_dentists num_students min_students_per_dentist : ℕ) 
  (h1 : num_dentists = 12)
  (h2 : num_students = 29)
  (h3 : min_students_per_dentist = 2)
  (h4 : num_dentists * min_students_per_dentist ≤ num_students) :
  ∃ (max_students : ℕ), max_students = 7 ∧ 
  (∀ (d : ℕ), d ≤ num_dentists → ∃ (s : ℕ), s ≤ num_students ∧ s ≤ max_students) ∧
  (∃ (d : ℕ), d ≤ num_dentists ∧ ∃ (s : ℕ), s = max_students) :=
by sorry

end NUMINAMATH_CALUDE_max_students_per_dentist_l2163_216320
