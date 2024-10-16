import Mathlib

namespace NUMINAMATH_CALUDE_expression_factorization_l3798_379827

theorem expression_factorization (x : ℝ) : 
  (16 * x^7 + 81 * x^4 - 9) - (4 * x^7 - 18 * x^4 + 3) = 3 * (4 * x^7 + 33 * x^4 - 4) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l3798_379827


namespace NUMINAMATH_CALUDE_buckingham_palace_visitor_difference_l3798_379815

/-- The number of visitors to Buckingham Palace on the current day -/
def current_day_visitors : ℕ := 317

/-- The number of visitors to Buckingham Palace on the previous day -/
def previous_day_visitors : ℕ := 295

/-- The difference in visitors between the current day and the previous day -/
def visitor_difference : ℕ := current_day_visitors - previous_day_visitors

theorem buckingham_palace_visitor_difference :
  visitor_difference = 22 :=
by sorry

end NUMINAMATH_CALUDE_buckingham_palace_visitor_difference_l3798_379815


namespace NUMINAMATH_CALUDE_square_garden_area_perimeter_relation_l3798_379897

theorem square_garden_area_perimeter_relation :
  ∀ (s : ℝ), 
    s > 0 →
    4 * s = 40 →
    s^2 - 2 * (4 * s) = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_square_garden_area_perimeter_relation_l3798_379897


namespace NUMINAMATH_CALUDE_zinc_copper_mixture_weight_l3798_379804

/-- Proves that given a mixture of zinc and copper in the ratio 9:11, 
    where 27 kg of zinc is used, the total weight of the mixture is 60 kg. -/
theorem zinc_copper_mixture_weight (zinc_weight : ℝ) (copper_weight : ℝ) :
  zinc_weight = 27 →
  zinc_weight / copper_weight = 9 / 11 →
  zinc_weight + copper_weight = 60 := by
sorry

end NUMINAMATH_CALUDE_zinc_copper_mixture_weight_l3798_379804


namespace NUMINAMATH_CALUDE_equation_solution_l3798_379862

theorem equation_solution : 
  ∃! x : ℚ, (x + 7) / (x - 4) = (x - 5) / (x + 3) ∧ x = -1/19 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l3798_379862


namespace NUMINAMATH_CALUDE_simplify_expression_l3798_379821

theorem simplify_expression (y : ℝ) : 5*y - 3*y + 7*y - 2*y + 6*y = 13*y := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3798_379821


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3798_379809

/-- A geometric sequence with first term 1 and fourth term 1/64 has common ratio 1/4 -/
theorem geometric_sequence_common_ratio (a : ℕ → ℚ) :
  (∀ n, a (n + 1) = a n * (a 2 / a 1)) →  -- Geometric sequence condition
  a 1 = 1 →                               -- First term is 1
  a 4 = 1 / 64 →                          -- Fourth term is 1/64
  a 2 / a 1 = 1 / 4 :=                    -- Common ratio is 1/4
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3798_379809


namespace NUMINAMATH_CALUDE_focal_lengths_equal_l3798_379887

/-- Focal length of a hyperbola with equation 15y^2 - x^2 = 15 -/
def hyperbola_focal_length : ℝ := 4

/-- Focal length of an ellipse with equation x^2/25 + y^2/9 = 1 -/
def ellipse_focal_length : ℝ := 4

/-- The focal lengths of the given hyperbola and ellipse are equal -/
theorem focal_lengths_equal : hyperbola_focal_length = ellipse_focal_length := by sorry

end NUMINAMATH_CALUDE_focal_lengths_equal_l3798_379887


namespace NUMINAMATH_CALUDE_total_spent_equals_621_l3798_379883

/-- The total amount spent by Tate and Peyton on their remaining tickets -/
def total_spent (tate_initial_tickets : ℕ) (tate_initial_price : ℕ) 
  (tate_additional_tickets : ℕ) (tate_additional_price : ℕ)
  (peyton_price : ℕ) : ℕ :=
  let tate_total := tate_initial_tickets * tate_initial_price + 
                    tate_additional_tickets * tate_additional_price
  let peyton_initial_tickets := tate_initial_tickets / 2
  let peyton_remaining_tickets := peyton_initial_tickets - 
                                  (peyton_initial_tickets / 3)
  let peyton_total := peyton_remaining_tickets * peyton_price
  tate_total + peyton_total

/-- Theorem stating the total amount spent by Tate and Peyton -/
theorem total_spent_equals_621 : 
  total_spent 32 14 2 15 13 = 621 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_equals_621_l3798_379883


namespace NUMINAMATH_CALUDE_exists_multiple_with_sum_of_digits_equal_to_n_l3798_379808

def sumOfDigits (m : ℕ) : ℕ :=
  if m < 10 then m else m % 10 + sumOfDigits (m / 10)

theorem exists_multiple_with_sum_of_digits_equal_to_n (n : ℕ) (hn : n > 0) :
  ∃ m : ℕ, m % n = 0 ∧ sumOfDigits m = n := by
  sorry

end NUMINAMATH_CALUDE_exists_multiple_with_sum_of_digits_equal_to_n_l3798_379808


namespace NUMINAMATH_CALUDE_last_digit_of_A_l3798_379888

theorem last_digit_of_A (A : ℕ) : 
  A = (2+1)*(2^2+1)*(2^4+1)*(2^8+1)+1 → 
  A % 10 = (2^16) % 10 := by
sorry

end NUMINAMATH_CALUDE_last_digit_of_A_l3798_379888


namespace NUMINAMATH_CALUDE_spiral_stripe_length_l3798_379816

/-- The length of a spiral stripe on a cylindrical water tower -/
theorem spiral_stripe_length 
  (circumference height : ℝ) 
  (h_circumference : circumference = 18) 
  (h_height : height = 24) :
  Real.sqrt (circumference^2 + height^2) = 30 := by sorry

end NUMINAMATH_CALUDE_spiral_stripe_length_l3798_379816


namespace NUMINAMATH_CALUDE_largest_solution_equation_inverse_x_12_value_l3798_379853

noncomputable def largest_x : ℝ :=
  Real.exp (- (7 / 12) * Real.log 10)

theorem largest_solution_equation (x : ℝ) (h : x = largest_x) :
  (Real.log 10) / (Real.log (10 * x^2)) + (Real.log 10) / (Real.log (100 * x^3)) = -2 :=
sorry

theorem inverse_x_12_value :
  (1 : ℝ) / largest_x^12 = 10000000 :=
sorry

end NUMINAMATH_CALUDE_largest_solution_equation_inverse_x_12_value_l3798_379853


namespace NUMINAMATH_CALUDE_distance_to_focus_l3798_379844

/-- The distance from a point on a parabola to its focus -/
theorem distance_to_focus (y : ℝ) : 
  y^2 = 8 * 2 →  -- Point M(2, y) is on the parabola y^2 = 8x
  4 = (2 - (-2)) -- Distance from M to the directrix (x = -2)
    + (2 - 0)    -- Distance from M to the x-coordinate of the focus (which is at x = 0 for this parabola)
  := by sorry

end NUMINAMATH_CALUDE_distance_to_focus_l3798_379844


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l3798_379859

theorem repeating_decimal_sum : 
  (1/3 : ℚ) + (4/999 : ℚ) + (5/9999 : ℚ) = (3378/9999 : ℚ) := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l3798_379859


namespace NUMINAMATH_CALUDE_triangle_area_product_l3798_379854

theorem triangle_area_product (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ a * x + b * y = 12) →
  (1/2 * (12/a) * (12/b) = 12) →
  a * b = 6 := by sorry

end NUMINAMATH_CALUDE_triangle_area_product_l3798_379854


namespace NUMINAMATH_CALUDE_k_range_unique_triangle_l3798_379898

/-- Represents an acute triangle ABC with specific properties -/
structure AcuteTriangle where
  /-- Side length AB -/
  k : ℝ
  /-- Angle C in radians -/
  angleC : ℝ
  /-- Angle A is 60 degrees (π/3 radians) -/
  angleA_eq : angleA = π/3
  /-- Side length BC is 6 -/
  bc_eq : bc = 6
  /-- Triangle is acute -/
  acute : 0 < angleC ∧ angleC < π/2
  /-- Sine rule holds -/
  sine_rule : k = 4 * Real.sqrt 3 * Real.sin angleC

/-- The range of k for the specific acute triangle -/
theorem k_range (t : AcuteTriangle) : 2 * Real.sqrt 3 < t.k ∧ t.k < 4 * Real.sqrt 3 := by
  sorry

/-- There exists only one such triangle -/
theorem unique_triangle : ∃! t : AcuteTriangle, True := by
  sorry

end NUMINAMATH_CALUDE_k_range_unique_triangle_l3798_379898


namespace NUMINAMATH_CALUDE_solve_linear_equation_l3798_379824

theorem solve_linear_equation :
  ∀ x : ℚ, -3 * x - 8 = 5 * x + 4 → x = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l3798_379824


namespace NUMINAMATH_CALUDE_original_bottle_caps_l3798_379870

theorem original_bottle_caps (removed : ℕ) (left : ℕ) (original : ℕ) : 
  removed = 47 → left = 40 → original = removed + left → original = 87 := by
  sorry

end NUMINAMATH_CALUDE_original_bottle_caps_l3798_379870


namespace NUMINAMATH_CALUDE_f_max_min_implies_m_range_l3798_379851

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^2 - 4*x + 5

-- State the theorem
theorem f_max_min_implies_m_range (m : ℝ) :
  (∀ x ∈ Set.Icc 0 m, f x ≤ 5) ∧  -- Maximum value is 5
  (∃ x ∈ Set.Icc 0 m, f x = 5) ∧  -- Maximum value is attained
  (∀ x ∈ Set.Icc 0 m, f x ≥ 1) ∧  -- Minimum value is 1
  (∃ x ∈ Set.Icc 0 m, f x = 1) →  -- Minimum value is attained
  m ∈ Set.Icc 2 4 :=
by sorry

end NUMINAMATH_CALUDE_f_max_min_implies_m_range_l3798_379851


namespace NUMINAMATH_CALUDE_arithmetic_progression_rth_term_l3798_379812

/-- The sum of the first n terms of an arithmetic progression -/
def S (n : ℕ) : ℚ := 3 * n + 5 * n^2

/-- The rth term of the arithmetic progression -/
def a (r : ℕ) : ℚ := S r - S (r - 1)

theorem arithmetic_progression_rth_term (r : ℕ) (hr : r > 0) : a r = 10 * r - 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_rth_term_l3798_379812


namespace NUMINAMATH_CALUDE_christine_needs_32_tablespoons_l3798_379818

/-- Represents the number of tablespoons of aquafaba equivalent to one egg white -/
def aquafaba_per_egg : ℕ := 2

/-- Represents the number of cakes Christine is making -/
def num_cakes : ℕ := 2

/-- Represents the number of egg whites required for each cake -/
def egg_whites_per_cake : ℕ := 8

/-- Calculates the total number of tablespoons of aquafaba needed -/
def aquafaba_needed : ℕ := aquafaba_per_egg * num_cakes * egg_whites_per_cake

/-- Proves that Christine needs 32 tablespoons of aquafaba -/
theorem christine_needs_32_tablespoons : aquafaba_needed = 32 := by
  sorry

end NUMINAMATH_CALUDE_christine_needs_32_tablespoons_l3798_379818


namespace NUMINAMATH_CALUDE_shirt_cost_l3798_379868

theorem shirt_cost (jeans_cost shirt_cost : ℚ) : 
  (3 * jeans_cost + 2 * shirt_cost = 69) →
  (2 * jeans_cost + 3 * shirt_cost = 61) →
  shirt_cost = 9 := by
sorry

end NUMINAMATH_CALUDE_shirt_cost_l3798_379868


namespace NUMINAMATH_CALUDE_moon_speed_conversion_l3798_379866

/-- The speed of the moon in kilometers per second -/
def moon_speed_km_per_sec : ℝ := 1.04

/-- The number of seconds in an hour -/
def seconds_per_hour : ℕ := 3600

/-- The speed of the moon in kilometers per hour -/
def moon_speed_km_per_hour : ℝ := moon_speed_km_per_sec * seconds_per_hour

theorem moon_speed_conversion : 
  moon_speed_km_per_hour = 3744 := by sorry

end NUMINAMATH_CALUDE_moon_speed_conversion_l3798_379866


namespace NUMINAMATH_CALUDE_fill_time_two_pumps_trough_fill_time_l3798_379861

/-- Calculates the time to fill a trough with two pumps working simultaneously -/
theorem fill_time_two_pumps (t1 t2 : ℝ) (h1 : t1 > 0) (h2 : t2 > 0) : 
  1 / (1 / t1 + 1 / t2) = (t1 * t2) / (t1 + t2) := by sorry

/-- Proves that two pumps with fill times of 600s and 200s will fill a trough in 150s when working simultaneously -/
theorem trough_fill_time : 
  1 / (1 / 600 + 1 / 200) = 150 := by sorry

end NUMINAMATH_CALUDE_fill_time_two_pumps_trough_fill_time_l3798_379861


namespace NUMINAMATH_CALUDE_tunneled_cube_surface_area_l3798_379817

/-- Represents a cube with its dimensions and composition -/
structure Cube where
  side_length : ℕ
  sub_cube_side : ℕ
  sub_cube_count : ℕ

/-- Represents the tunneling operation on the cube -/
structure TunneledCube extends Cube where
  removed_layers : ℕ
  removed_edge_units : ℕ

/-- Calculates the surface area of a tunneled cube -/
def surface_area (tc : TunneledCube) : ℕ :=
  sorry

/-- The main theorem stating the surface area of the specific tunneled cube -/
theorem tunneled_cube_surface_area :
  let original_cube : Cube := {
    side_length := 12,
    sub_cube_side := 3,
    sub_cube_count := 64
  }
  let tunneled_cube : TunneledCube := {
    side_length := original_cube.side_length,
    sub_cube_side := original_cube.sub_cube_side,
    sub_cube_count := original_cube.sub_cube_count,
    removed_layers := 2,
    removed_edge_units := 1
  }
  surface_area tunneled_cube = 2496 := by
  sorry

end NUMINAMATH_CALUDE_tunneled_cube_surface_area_l3798_379817


namespace NUMINAMATH_CALUDE_population_net_increase_l3798_379813

/-- Represents the birth rate in people per two seconds -/
def birth_rate : ℚ := 7

/-- Represents the death rate in people per two seconds -/
def death_rate : ℚ := 2

/-- Represents the number of seconds in a day -/
def seconds_per_day : ℕ := 24 * 60 * 60

/-- Theorem stating the net increase in population size in one day -/
theorem population_net_increase : 
  (birth_rate - death_rate) / 2 * seconds_per_day = 216000 := by sorry

end NUMINAMATH_CALUDE_population_net_increase_l3798_379813


namespace NUMINAMATH_CALUDE_triangle_properties_l3798_379826

theorem triangle_properties (A B C : Real) (a b c : Real) (D : Real) :
  (0 < A) ∧ (A < π) ∧ (0 < B) ∧ (B < π) ∧ (0 < C) ∧ (C < π) →
  (2 * a * (Real.sin (2 * B) - Real.sin A * Real.cos C) = c * Real.sin (2 * A)) →
  (3 : Real) = 3 →
  (Real.sin (π / 3 : Real) = Real.sin (Real.pi / 3)) →
  ((1 / 2 : Real) * a * c * Real.sin B = 3 * Real.sqrt 3) →
  (B = π / 3) ∧
  (a + b + c = 2 * Real.sqrt 13 + 4) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l3798_379826


namespace NUMINAMATH_CALUDE_units_digit_product_l3798_379829

theorem units_digit_product (n : ℕ) : n = 3^401 * 7^402 * 23^403 → n % 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_product_l3798_379829


namespace NUMINAMATH_CALUDE_find_number_l3798_379893

theorem find_number (x : ℝ) (h : 0.46 * x = 165.6) : x = 360 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l3798_379893


namespace NUMINAMATH_CALUDE_ginger_water_usage_l3798_379832

/-- The amount of water Ginger drank and used in her garden --/
def water_used (hours_worked : ℕ) (cups_per_bottle : ℕ) (extra_bottles : ℕ) : ℕ :=
  (hours_worked * cups_per_bottle) + (extra_bottles * cups_per_bottle)

/-- Theorem stating the total amount of water Ginger used --/
theorem ginger_water_usage :
  water_used 8 2 5 = 26 := by
  sorry

end NUMINAMATH_CALUDE_ginger_water_usage_l3798_379832


namespace NUMINAMATH_CALUDE_inequality_proof_l3798_379865

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a / (b + 2*c + 3*d)) + (b / (c + 2*d + 3*a)) + (c / (d + 2*a + 3*b)) + (d / (a + 2*b + 3*c)) ≥ 2/3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3798_379865


namespace NUMINAMATH_CALUDE_count_satisfying_integers_l3798_379886

-- Define the function f(n)
def f (n : ℤ) : ℤ := ⌈(99 * n : ℚ) / 100⌉ - ⌊(100 * n : ℚ) / 101⌋

-- State the theorem
theorem count_satisfying_integers :
  (∃ (S : Finset ℤ), (∀ n ∈ S, f n = 1) ∧ S.card = 10100 ∧
    (∀ n : ℤ, f n = 1 → n ∈ S)) :=
sorry

end NUMINAMATH_CALUDE_count_satisfying_integers_l3798_379886


namespace NUMINAMATH_CALUDE_cannot_determine_percentage_increase_l3798_379838

/-- Represents a manufacturing machine -/
structure Machine where
  name : String
  production_rate : ℝ

/-- The problem setup -/
def sprocket_problem (time_q : ℝ) : Prop :=
  let machine_a : Machine := ⟨"A", 4⟩
  let machine_q : Machine := ⟨"Q", 440 / time_q⟩
  let machine_p : Machine := ⟨"P", 440 / (time_q + 10)⟩
  let percentage_increase := (machine_q.production_rate - machine_a.production_rate) / machine_a.production_rate * 100

  -- Conditions
  440 > 0 ∧
  time_q > 0 ∧
  machine_p.production_rate < machine_q.production_rate ∧
  -- Question: Can we determine the percentage increase?
  ∃ (x : ℝ), percentage_increase = x

/-- The theorem stating that we cannot determine the percentage increase without knowing time_q -/
theorem cannot_determine_percentage_increase :
  ¬∃ (x : ℝ), ∀ (time_q : ℝ), sprocket_problem time_q → 
    (440 / time_q - 4) / 4 * 100 = x :=
sorry

end NUMINAMATH_CALUDE_cannot_determine_percentage_increase_l3798_379838


namespace NUMINAMATH_CALUDE_complex_modulus_power_eight_l3798_379850

theorem complex_modulus_power_eight : 
  Complex.abs ((1/2 : ℂ) + (Complex.I * (Real.sqrt 3 / 2)))^8 = 1 := by sorry

end NUMINAMATH_CALUDE_complex_modulus_power_eight_l3798_379850


namespace NUMINAMATH_CALUDE_negative_operation_l3798_379825

theorem negative_operation (a b c d : ℤ) : a = (-7) * (-6) ∧ b = (-7) - (-15) ∧ c = 0 * (-2) * (-3) ∧ d = (-6) + (-4) → d < 0 := by
  sorry

end NUMINAMATH_CALUDE_negative_operation_l3798_379825


namespace NUMINAMATH_CALUDE_stating_anoop_join_time_l3798_379814

/-- Represents the number of months in a year -/
def monthsInYear : ℕ := 12

/-- Represents Arjun's investment in rupees -/
def arjunInvestment : ℕ := 20000

/-- Represents Anoop's investment in rupees -/
def anoopInvestment : ℕ := 4000

/-- 
Theorem stating that if Arjun invests for 12 months and Anoop invests for (12 - x) months,
and their profits are divided equally, then Anoop must have joined after 7 months.
-/
theorem anoop_join_time (x : ℕ) : 
  (arjunInvestment * monthsInYear) / (anoopInvestment * (monthsInYear - x)) = 1 → x = 7 := by
  sorry


end NUMINAMATH_CALUDE_stating_anoop_join_time_l3798_379814


namespace NUMINAMATH_CALUDE_tetrachloromethane_formation_l3798_379867

-- Define the chemical species
structure ChemicalSpecies where
  name : String
  moles : ℝ

-- Define the reaction equation
structure ReactionEquation where
  reactants : List ChemicalSpecies
  products : List ChemicalSpecies

-- Define the problem parameters
def methane : ChemicalSpecies := ⟨"CH4", 1⟩
def chlorine : ChemicalSpecies := ⟨"Cl2", 4⟩
def tetrachloromethane : ChemicalSpecies := ⟨"CCl4", 0⟩ -- Initial amount is 0
def hydrogenChloride : ChemicalSpecies := ⟨"HCl", 0⟩ -- Initial amount is 0

-- Define the balanced reaction equation
def balancedEquation : ReactionEquation :=
  ⟨[methane, chlorine], [tetrachloromethane, hydrogenChloride]⟩

-- Theorem statement
theorem tetrachloromethane_formation
  (reactionEq : ReactionEquation)
  (h1 : reactionEq = balancedEquation)
  (h2 : methane.moles = 1)
  (h3 : chlorine.moles = 4) :
  tetrachloromethane.moles = 1 :=
sorry

end NUMINAMATH_CALUDE_tetrachloromethane_formation_l3798_379867


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l3798_379845

theorem simplify_sqrt_expression (x : ℝ) (hx : x ≠ 0) :
  Real.sqrt (4 + ((x^3 - 2) / x)^2) = (Real.sqrt (x^6 - 4*x^3 + 4*x^2 + 4)) / x :=
by sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l3798_379845


namespace NUMINAMATH_CALUDE_problem_solution_l3798_379858

theorem problem_solution (a b c : ℝ) : 
  b = 15 → 
  c = 3 → 
  a * b * c = (Real.sqrt ((a + 2) * (b + 3))) / (c + 1) → 
  a * b * c = 3 → 
  a = 6 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3798_379858


namespace NUMINAMATH_CALUDE_petes_flag_shapes_l3798_379879

/-- Given a flag with circles and squares, calculate the total number of shapes --/
def total_shapes (stars : ℕ) (stripes : ℕ) : ℕ :=
  let circles := stars / 2 - 3
  let squares := stripes * 2 + 6
  circles + squares

/-- Theorem: The total number of shapes on Pete's flag is 54 --/
theorem petes_flag_shapes :
  total_shapes 50 13 = 54 := by
  sorry

end NUMINAMATH_CALUDE_petes_flag_shapes_l3798_379879


namespace NUMINAMATH_CALUDE_sequence_formula_correct_l3798_379802

/-- Given a sequence {a_n} where S_n is the sum of the first n terms -/
def S (n : ℕ) : ℤ := n^2 - 10*n

/-- The nth term of the sequence -/
def a (n : ℕ) : ℤ := 2*n - 11

/-- Theorem stating that the given formula for a_n is correct -/
theorem sequence_formula_correct (n : ℕ) : 
  n ≥ 1 → S n - S (n-1) = a n := by sorry

end NUMINAMATH_CALUDE_sequence_formula_correct_l3798_379802


namespace NUMINAMATH_CALUDE_evaluate_expression_l3798_379847

theorem evaluate_expression (c x y z : ℚ) :
  c = -2 →
  x = 2/5 →
  y = 3/5 →
  z = -3 →
  c * x^3 * y^4 * z^2 = -11664/78125 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3798_379847


namespace NUMINAMATH_CALUDE_candy_sharing_l3798_379846

theorem candy_sharing (hugh tommy melany : ℕ) (h1 : hugh = 8) (h2 : tommy = 6) (h3 : melany = 7) :
  (hugh + tommy + melany) / 3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_candy_sharing_l3798_379846


namespace NUMINAMATH_CALUDE_problem_statement_l3798_379871

theorem problem_statement (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (hab : a^2 + b^2 = a*b + 1) (hcd : c*d > 1) :
  (a + b ≤ 2) ∧ (Real.sqrt (a*c) + Real.sqrt (b*d) < c + d) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3798_379871


namespace NUMINAMATH_CALUDE_bd_squared_equals_four_l3798_379876

theorem bd_squared_equals_four (a b c d : ℤ) 
  (eq1 : a - b - c + d = 13) 
  (eq2 : a + b - c - d = 9) : 
  (b - d)^2 = 4 := by sorry

end NUMINAMATH_CALUDE_bd_squared_equals_four_l3798_379876


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_l3798_379807

-- Define the propositions p and q
def p (a : ℝ) : Prop := 1/a > 1/4

def q (a : ℝ) : Prop := ∀ x : ℝ, a*x^2 + a*x + 1 > 0

-- Theorem stating that p is sufficient but not necessary for q
theorem p_sufficient_not_necessary :
  (∃ a : ℝ, p a ∧ q a) ∧ (∃ a : ℝ, ¬p a ∧ q a) ∧ (∀ a : ℝ, p a → q a) :=
sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_l3798_379807


namespace NUMINAMATH_CALUDE_point_transformation_l3798_379855

def rotate_180 (x y : ℝ) : ℝ × ℝ :=
  (4 - x, 6 - y)

def reflect_y_eq_x (x y : ℝ) : ℝ × ℝ :=
  (y, x)

theorem point_transformation (a b : ℝ) :
  (reflect_y_eq_x (rotate_180 a b).1 (rotate_180 a b).2) = (2, -5) →
  a + b = 13 := by
sorry

end NUMINAMATH_CALUDE_point_transformation_l3798_379855


namespace NUMINAMATH_CALUDE_average_calls_proof_l3798_379801

def average_calls (mon tue wed thu fri : ℕ) : ℚ :=
  (mon + tue + wed + thu + fri : ℚ) / 5

theorem average_calls_proof (mon tue wed thu fri : ℕ) :
  average_calls mon tue wed thu fri = (mon + tue + wed + thu + fri : ℚ) / 5 := by
  sorry

end NUMINAMATH_CALUDE_average_calls_proof_l3798_379801


namespace NUMINAMATH_CALUDE_xy_length_is_30_l3798_379875

/-- A right triangle XYZ with specific angle and side length properties -/
structure RightTriangleXYZ where
  /-- The length of side XZ -/
  xz : ℝ
  /-- The measure of angle Y in radians -/
  angle_y : ℝ
  /-- XZ equals 15 -/
  xz_eq : xz = 15
  /-- Angle Y equals 30 degrees (π/6 radians) -/
  angle_y_eq : angle_y = π / 6
  /-- The triangle is a right triangle (angle X is 90 degrees) -/
  right_angle : True

/-- The length of side XY in the right triangle XYZ -/
def length_xy (t : RightTriangleXYZ) : ℝ := 2 * t.xz

/-- Theorem stating that the length of XY is 30 in the given right triangle -/
theorem xy_length_is_30 (t : RightTriangleXYZ) : length_xy t = 30 := by
  sorry

end NUMINAMATH_CALUDE_xy_length_is_30_l3798_379875


namespace NUMINAMATH_CALUDE_height_ratio_l3798_379820

def sara_height : ℝ := 120 - 82
def joe_height : ℝ := 82

axiom combined_height : sara_height + joe_height = 120
axiom joe_height_relation : ∃ k : ℝ, joe_height = k * sara_height + 6

theorem height_ratio : (joe_height / sara_height) = 41 / 19 := by
  sorry

end NUMINAMATH_CALUDE_height_ratio_l3798_379820


namespace NUMINAMATH_CALUDE_fifty_billion_scientific_notation_l3798_379843

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h_coeff : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem fifty_billion_scientific_notation :
  toScientificNotation 50000000000 = ScientificNotation.mk 5 10 (by sorry) :=
sorry

end NUMINAMATH_CALUDE_fifty_billion_scientific_notation_l3798_379843


namespace NUMINAMATH_CALUDE_smaller_circle_radius_l3798_379828

/-- Given two circles where one encloses the other, this theorem proves
    the radius of the smaller circle given specific conditions. -/
theorem smaller_circle_radius
  (R : ℝ) -- Radius of the larger circle
  (r : ℝ) -- Radius of the smaller circle
  (A₁ : ℝ) -- Area of the smaller circle
  (A₂ : ℝ) -- Area difference between the two circles
  (h1 : R = 5) -- The larger circle has a radius of 5 units
  (h2 : A₁ = π * r^2) -- Area formula for the smaller circle
  (h3 : A₂ = π * R^2 - A₁) -- Area difference
  (h4 : ∃ (d : ℝ), A₁ + d = A₂ ∧ A₂ + d = A₁ + A₂) -- Arithmetic progression condition
  : r = 5 * Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_smaller_circle_radius_l3798_379828


namespace NUMINAMATH_CALUDE_tournament_games_l3798_379806

theorem tournament_games (total_teams : Nat) (preliminary_teams : Nat) (preliminary_matches : Nat) :
  total_teams = 24 →
  preliminary_teams = 16 →
  preliminary_matches = 8 →
  preliminary_teams = 2 * preliminary_matches →
  (total_games : Nat) = preliminary_matches + (total_teams - preliminary_matches) - 1 →
  total_games = 23 := by
  sorry

end NUMINAMATH_CALUDE_tournament_games_l3798_379806


namespace NUMINAMATH_CALUDE_roundness_of_1728000_l3798_379873

/-- Roundness of a positive integer is the sum of the exponents in its prime factorization -/
def roundness (n : ℕ+) : ℕ := sorry

/-- The roundness of 1,728,000 is 19 -/
theorem roundness_of_1728000 : roundness 1728000 = 19 := by sorry

end NUMINAMATH_CALUDE_roundness_of_1728000_l3798_379873


namespace NUMINAMATH_CALUDE_brand_preference_l3798_379819

theorem brand_preference (total : ℕ) (ratio : ℚ) (brand_x : ℕ) : 
  total = 180 →
  ratio = 5 / 1 →
  brand_x * (1 + 1 / ratio) = total →
  brand_x = 150 :=
by sorry

end NUMINAMATH_CALUDE_brand_preference_l3798_379819


namespace NUMINAMATH_CALUDE_additional_track_length_l3798_379890

/-- Calculate the additional track length required when reducing grade -/
theorem additional_track_length
  (rise : ℝ)
  (initial_grade : ℝ)
  (reduced_grade : ℝ)
  (h1 : rise = 800)
  (h2 : initial_grade = 0.04)
  (h3 : reduced_grade = 0.025) :
  (rise / reduced_grade) - (rise / initial_grade) = 12000 :=
by sorry

end NUMINAMATH_CALUDE_additional_track_length_l3798_379890


namespace NUMINAMATH_CALUDE_monthly_spending_fraction_l3798_379857

/-- If a person saves a constant fraction of their unchanging monthly salary,
    and their yearly savings are 6 times their monthly spending,
    then they spend 2/3 of their salary each month. -/
theorem monthly_spending_fraction
  (salary : ℝ)
  (savings_fraction : ℝ)
  (h_salary_positive : 0 < salary)
  (h_savings_fraction : 0 ≤ savings_fraction ∧ savings_fraction ≤ 1)
  (h_yearly_savings : 12 * savings_fraction * salary = 6 * (1 - savings_fraction) * salary) :
  1 - savings_fraction = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_monthly_spending_fraction_l3798_379857


namespace NUMINAMATH_CALUDE_unique_modular_solution_l3798_379884

theorem unique_modular_solution : ∃! n : ℤ, n ≡ -5678 [ZMOD 10] ∧ 0 ≤ n ∧ n ≤ 9 ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_modular_solution_l3798_379884


namespace NUMINAMATH_CALUDE_remainder_theorem_polynomial_division_remainder_l3798_379881

def P (x : ℝ) : ℝ := 8*x^5 - 10*x^4 + 6*x^3 - 2*x^2 + 3*x - 35

theorem remainder_theorem (P : ℝ → ℝ) (a : ℝ) :
  ∃ Q : ℝ → ℝ, ∀ x, P x = (x - a) * Q x + P a :=
sorry

theorem polynomial_division_remainder :
  ∃ Q : ℝ → ℝ, ∀ x, P x = (2*x - 8) * Q x + 5961 :=
sorry

end NUMINAMATH_CALUDE_remainder_theorem_polynomial_division_remainder_l3798_379881


namespace NUMINAMATH_CALUDE_jacksons_grade_l3798_379872

/-- Calculates Jackson's grade based on his study time and point increase rate. -/
def calculate_grade (gaming_hours : ℝ) (study_ratio : ℝ) (points_per_hour : ℝ) : ℝ :=
  gaming_hours * study_ratio * points_per_hour

/-- Theorem stating that Jackson's grade is 45 points given the problem conditions. -/
theorem jacksons_grade :
  let gaming_hours : ℝ := 9
  let study_ratio : ℝ := 1/3
  let points_per_hour : ℝ := 15
  calculate_grade gaming_hours study_ratio points_per_hour = 45 := by
  sorry


end NUMINAMATH_CALUDE_jacksons_grade_l3798_379872


namespace NUMINAMATH_CALUDE_transform_equation_l3798_379848

theorem transform_equation (m n x y : ℚ) :
  m + x = n + y → m = n → x = y := by
  sorry

end NUMINAMATH_CALUDE_transform_equation_l3798_379848


namespace NUMINAMATH_CALUDE_x_squared_plus_reciprocal_l3798_379899

theorem x_squared_plus_reciprocal (x : ℝ) (h : 47 = x^4 + 1/x^4) : x^2 + 1/x^2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_plus_reciprocal_l3798_379899


namespace NUMINAMATH_CALUDE_distance_circle_center_to_line_l3798_379860

theorem distance_circle_center_to_line :
  let line_eq : ℝ → ℝ → Prop := λ x y => x + y = 6
  let circle_eq : ℝ → ℝ → Prop := λ x y => x^2 + (y - 2)^2 = 4
  let circle_center : ℝ × ℝ := (0, 2)
  ∃ d : ℝ, d = 2 * Real.sqrt 2 ∧
    d = (|0 + 2 - 6|) / Real.sqrt ((1:ℝ)^2 + 1^2) :=
by sorry

end NUMINAMATH_CALUDE_distance_circle_center_to_line_l3798_379860


namespace NUMINAMATH_CALUDE_max_value_product_l3798_379880

theorem max_value_product (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) 
  (hsum : a + b + c = 3) : 
  (a^2 - a*b + b^2) * (a^2 - a*c + c^2) * (b^2 - b*c + c^2) ≤ 729/432 ∧ 
  ∃ a b c, a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ a + b + c = 3 ∧ 
    (a^2 - a*b + b^2) * (a^2 - a*c + c^2) * (b^2 - b*c + c^2) = 729/432 :=
by sorry

end NUMINAMATH_CALUDE_max_value_product_l3798_379880


namespace NUMINAMATH_CALUDE_area_of_inner_rectangle_l3798_379895

theorem area_of_inner_rectangle (s : ℝ) (h : s > 0) : 
  let larger_square_area := s^2
  let half_larger_square_area := larger_square_area / 2
  let inner_rectangle_side := s / 2
  let inner_rectangle_area := inner_rectangle_side^2
  half_larger_square_area = 80 → inner_rectangle_area = 40 := by
  sorry

end NUMINAMATH_CALUDE_area_of_inner_rectangle_l3798_379895


namespace NUMINAMATH_CALUDE_initial_stock_proof_l3798_379836

/-- The number of coloring books initially in stock at a store -/
def initial_stock : ℕ := 86

/-- The number of coloring books sold -/
def books_sold : ℕ := 37

/-- The number of shelves used for remaining books -/
def shelves_used : ℕ := 7

/-- The number of books on each shelf -/
def books_per_shelf : ℕ := 7

/-- Theorem stating that the initial stock equals 86 -/
theorem initial_stock_proof : 
  initial_stock = books_sold + (shelves_used * books_per_shelf) :=
by sorry

end NUMINAMATH_CALUDE_initial_stock_proof_l3798_379836


namespace NUMINAMATH_CALUDE_product_inequality_l3798_379800

theorem product_inequality (a b m : ℕ) : 
  (a + b = 40 → a * b ≤ 20^2) ∧ 
  (a + b = m → a * b ≤ (m / 2)^2) :=
by sorry

end NUMINAMATH_CALUDE_product_inequality_l3798_379800


namespace NUMINAMATH_CALUDE_inequality_theorem_l3798_379837

theorem inequality_theorem (x : ℝ) : x^2 + 1 + (x^2 + 1)⁻¹ ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_theorem_l3798_379837


namespace NUMINAMATH_CALUDE_polygon_rotation_theorem_l3798_379805

theorem polygon_rotation_theorem (n : ℕ) (h : n ≥ 3) 
  (a : Fin n → Fin n) (h_perm : Function.Bijective a) 
  (h_initial : ∀ i : Fin n, a i ≠ i) :
  ∃ (r : ℕ) (i j : Fin n), i ≠ j ∧ 
    (a i).val - i.val ≡ r [MOD n] ∧
    (a j).val - j.val ≡ r [MOD n] :=
sorry

end NUMINAMATH_CALUDE_polygon_rotation_theorem_l3798_379805


namespace NUMINAMATH_CALUDE_definite_integral_sin_plus_one_l3798_379840

theorem definite_integral_sin_plus_one : ∫ x in (-1)..(1), (Real.sin x + 1) = 2 := by sorry

end NUMINAMATH_CALUDE_definite_integral_sin_plus_one_l3798_379840


namespace NUMINAMATH_CALUDE_commodity_trade_fair_companies_l3798_379810

theorem commodity_trade_fair_companies : ∃ (n : ℕ), n > 0 ∧ n * (n - 1) / 2 = 45 := by
  sorry

end NUMINAMATH_CALUDE_commodity_trade_fair_companies_l3798_379810


namespace NUMINAMATH_CALUDE_coefficient_abc_in_expansion_coefficient_of_ab2c3_l3798_379892

theorem coefficient_abc_in_expansion : ℕ → Prop :=
  fun n => (1 + 1 + 1)^6 = n + sorry

theorem coefficient_of_ab2c3 : coefficient_abc_in_expansion 60 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_abc_in_expansion_coefficient_of_ab2c3_l3798_379892


namespace NUMINAMATH_CALUDE_interest_equality_theorem_l3798_379831

theorem interest_equality_theorem (total : ℝ) (x : ℝ) : 
  total = 2665 →
  (x * 3 * 8) / 100 = ((total - x) * 5 * 3) / 100 →
  total - x = 1640 := by
  sorry

end NUMINAMATH_CALUDE_interest_equality_theorem_l3798_379831


namespace NUMINAMATH_CALUDE_imaginary_part_of_reciprocal_l3798_379889

-- Define the complex number z
def z : ℂ := 1 - 2 * Complex.I

-- Theorem statement
theorem imaginary_part_of_reciprocal (z : ℂ) (h : z = 1 - 2 * Complex.I) :
  Complex.im (z⁻¹) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_reciprocal_l3798_379889


namespace NUMINAMATH_CALUDE_correlation_strength_increases_l3798_379877

-- Define the correlation coefficient as a real number between -1 and 1
def correlation_coefficient : Type := {r : ℝ // -1 ≤ r ∧ r ≤ 1}

-- Define a measure of linear correlation strength
def linear_correlation_strength (r : correlation_coefficient) : ℝ := |r.val|

-- Define a notion of "closer to 1"
def closer_to_one (r1 r2 : correlation_coefficient) : Prop :=
  |r1.val - 1| < |r2.val - 1|

-- Statement: As |r| approaches 1, the linear correlation becomes stronger
theorem correlation_strength_increases (r1 r2 : correlation_coefficient) :
  closer_to_one r1 r2 → linear_correlation_strength r1 > linear_correlation_strength r2 :=
sorry

end NUMINAMATH_CALUDE_correlation_strength_increases_l3798_379877


namespace NUMINAMATH_CALUDE_baseball_card_pages_l3798_379830

theorem baseball_card_pages (cards_per_page : ℕ) (new_cards : ℕ) (old_cards : ℕ) :
  cards_per_page = 3 →
  new_cards = 2 →
  old_cards = 10 →
  (new_cards + old_cards) / cards_per_page = 4 :=
by sorry

end NUMINAMATH_CALUDE_baseball_card_pages_l3798_379830


namespace NUMINAMATH_CALUDE_jersey_profit_calculation_l3798_379885

/-- The amount of money made from each jersey -/
def jersey_profit : ℝ := 165

/-- The number of jerseys sold -/
def jerseys_sold : ℕ := 156

/-- The total money made from selling jerseys -/
def total_jersey_profit : ℝ := jersey_profit * (jerseys_sold : ℝ)

theorem jersey_profit_calculation : total_jersey_profit = 25740 := by
  sorry

end NUMINAMATH_CALUDE_jersey_profit_calculation_l3798_379885


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l3798_379869

theorem polynomial_division_remainder : ∃ (q r : Polynomial ℝ),
  X^4 + 1 = (X^2 - 4*X + 6) * q + r ∧
  r.degree < (X^2 - 4*X + 6).degree ∧
  r = 16*X - 59 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l3798_379869


namespace NUMINAMATH_CALUDE_average_of_four_numbers_l3798_379841

theorem average_of_four_numbers (n : ℝ) :
  (3 + 16 + 33 + (n + 1)) / 4 = 20 → n = 27 := by
  sorry

end NUMINAMATH_CALUDE_average_of_four_numbers_l3798_379841


namespace NUMINAMATH_CALUDE_jenny_walking_distance_l3798_379864

theorem jenny_walking_distance (ran_distance : Real) (extra_ran_distance : Real) :
  ran_distance = 0.6 →
  extra_ran_distance = 0.2 →
  ∃ walked_distance : Real,
    walked_distance + extra_ran_distance = ran_distance ∧
    walked_distance = 0.4 :=
by sorry

end NUMINAMATH_CALUDE_jenny_walking_distance_l3798_379864


namespace NUMINAMATH_CALUDE_utensils_packs_john_utensil_purchase_l3798_379894

theorem utensils_packs (total_utensils : Nat) (spoons_wanted : Nat) : Nat :=
  let spoons_per_pack := total_utensils / 3
  let packs_needed := spoons_wanted / spoons_per_pack
  packs_needed

theorem john_utensil_purchase : utensils_packs 30 50 = 5 := by
  sorry

end NUMINAMATH_CALUDE_utensils_packs_john_utensil_purchase_l3798_379894


namespace NUMINAMATH_CALUDE_triangle_area_l3798_379891

/-- Given a triangle ABC with sides a, b, c and circumradius R, 
    prove that its area is 2√3 / 3 under specific conditions -/
theorem triangle_area (a b c R : ℝ) (h1 : (a^2 - c^2) / (2*R) = (a - b) * Real.sin b)
                                    (h2 : Real.sin b = 2 * Real.sin a)
                                    (h3 : c = 2) :
  (1/2) * a * b * Real.sin ((1/3) * Real.pi) = 2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l3798_379891


namespace NUMINAMATH_CALUDE_select_at_most_one_ab_l3798_379803

def students : ℕ := 5
def selected : ℕ := 3
def competitions : ℕ := 3

def ways_to_select (n k : ℕ) : ℕ := Nat.choose n k

def ways_to_assign (n : ℕ) : ℕ := Nat.factorial n

def select_with_at_most_one_specific (total specific selected : ℕ) : ℕ :=
  -- Case 1: One specific student selected
  2 * (ways_to_select (total - 2) (selected - 1) * ways_to_assign competitions) +
  -- Case 2: Neither specific student selected
  (ways_to_select (total - 2) selected * ways_to_assign competitions)

theorem select_at_most_one_ab :
  select_with_at_most_one_specific students 2 selected = 42 := by
  sorry

end NUMINAMATH_CALUDE_select_at_most_one_ab_l3798_379803


namespace NUMINAMATH_CALUDE_ticket_cost_l3798_379874

/-- Given the total amount collected and average daily ticket sales over three days,
    prove that the cost of one ticket is $4. -/
theorem ticket_cost (total_amount : ℚ) (avg_daily_sales : ℚ) 
  (h1 : total_amount = 960)
  (h2 : avg_daily_sales = 80) : 
  total_amount / (avg_daily_sales * 3) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ticket_cost_l3798_379874


namespace NUMINAMATH_CALUDE_increasing_function_a_range_l3798_379822

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then (a - 1) * x + 3 * a - 4 else a^x

-- State the theorem
theorem increasing_function_a_range (a : ℝ) 
  (h1 : a > 0) (h2 : a ≠ 1) 
  (h3 : ∀ x y : ℝ, x < y → f a x < f a y) : 
  a > 1 ∧ a ≤ 5/3 := by
  sorry

end NUMINAMATH_CALUDE_increasing_function_a_range_l3798_379822


namespace NUMINAMATH_CALUDE_ratio_sum_squares_implies_sum_l3798_379896

theorem ratio_sum_squares_implies_sum (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  b = 2 * a →
  c = 3 * a →
  a^2 + b^2 + c^2 = 2016 →
  a + b + c = 72 := by
sorry

end NUMINAMATH_CALUDE_ratio_sum_squares_implies_sum_l3798_379896


namespace NUMINAMATH_CALUDE_division_theorem_l3798_379835

theorem division_theorem (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) :
  dividend = 132 →
  divisor = 16 →
  quotient = 8 →
  dividend = divisor * quotient + remainder →
  remainder = 4 := by
sorry

end NUMINAMATH_CALUDE_division_theorem_l3798_379835


namespace NUMINAMATH_CALUDE_cards_per_box_l3798_379878

theorem cards_per_box (total_cards : ℕ) (unboxed_cards : ℕ) (boxes_given : ℕ) (boxes_left : ℕ) :
  total_cards = 75 →
  unboxed_cards = 5 →
  boxes_given = 2 →
  boxes_left = 5 →
  (total_cards - unboxed_cards) % (boxes_given + boxes_left) = 0 →
  (total_cards - unboxed_cards) / (boxes_given + boxes_left) = 10 := by
  sorry

end NUMINAMATH_CALUDE_cards_per_box_l3798_379878


namespace NUMINAMATH_CALUDE_part_one_part_two_l3798_379811

/-- The quadratic function y in terms of x and a -/
def y (x a : ℝ) : ℝ := x^2 - (a + 2) * x + 4

/-- Part 1 of the theorem -/
theorem part_one (a b : ℝ) (h1 : b > 1) 
  (h2 : ∀ x, y x a < 0 ↔ 1 < x ∧ x < b) : 
  a = 3 ∧ b = 4 := by sorry

/-- Part 2 of the theorem -/
theorem part_two (a : ℝ) 
  (h : ∀ x, 1 ≤ x → x ≤ 4 → y x a ≥ -a - 1) : 
  a ≤ 4 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3798_379811


namespace NUMINAMATH_CALUDE_average_weight_whole_class_l3798_379863

theorem average_weight_whole_class 
  (students_a : ℕ) (students_b : ℕ) 
  (avg_weight_a : ℚ) (avg_weight_b : ℚ) :
  students_a = 40 →
  students_b = 20 →
  avg_weight_a = 50 →
  avg_weight_b = 40 →
  (students_a * avg_weight_a + students_b * avg_weight_b) / (students_a + students_b) = 140 / 3 :=
by sorry

end NUMINAMATH_CALUDE_average_weight_whole_class_l3798_379863


namespace NUMINAMATH_CALUDE_complex_equation_sum_l3798_379823

theorem complex_equation_sum (a b : ℝ) (h : (a : ℂ) + b * Complex.I = (1 - Complex.I) * (2 + Complex.I)) : 
  a + b = 2 := by sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l3798_379823


namespace NUMINAMATH_CALUDE_replaced_lettuce_cost_is_1_75_l3798_379882

/-- Represents the grocery order with its components -/
structure GroceryOrder where
  originalTotal : ℝ
  tomatoesOld : ℝ
  tomatoesNew : ℝ
  lettuceOld : ℝ
  celeryOld : ℝ
  celeryNew : ℝ
  deliveryAndTip : ℝ
  newTotal : ℝ

/-- The cost of the replaced lettuce given the grocery order details -/
def replacedLettuceCost (order : GroceryOrder) : ℝ :=
  order.lettuceOld + (order.newTotal - order.originalTotal - order.deliveryAndTip) -
  ((order.tomatoesNew - order.tomatoesOld) + (order.celeryNew - order.celeryOld))

/-- Theorem stating that the cost of the replaced lettuce is $1.75 -/
theorem replaced_lettuce_cost_is_1_75 (order : GroceryOrder)
  (h1 : order.originalTotal = 25)
  (h2 : order.tomatoesOld = 0.99)
  (h3 : order.tomatoesNew = 2.20)
  (h4 : order.lettuceOld = 1.00)
  (h5 : order.celeryOld = 1.96)
  (h6 : order.celeryNew = 2.00)
  (h7 : order.deliveryAndTip = 8.00)
  (h8 : order.newTotal = 35) :
  replacedLettuceCost order = 1.75 := by
  sorry

end NUMINAMATH_CALUDE_replaced_lettuce_cost_is_1_75_l3798_379882


namespace NUMINAMATH_CALUDE_cookie_boxes_theorem_l3798_379834

theorem cookie_boxes_theorem (n : ℕ) : 
  (∃ (m a : ℕ), 
    m = n - 11 ∧ 
    a = n - 2 ∧ 
    m ≥ 1 ∧ 
    a ≥ 1 ∧ 
    m + a < n) → 
  n = 12 := by
  sorry

end NUMINAMATH_CALUDE_cookie_boxes_theorem_l3798_379834


namespace NUMINAMATH_CALUDE_geometric_series_second_term_l3798_379852

theorem geometric_series_second_term :
  ∀ (a : ℝ) (r : ℝ) (S : ℝ),
    r = (1 : ℝ) / 4 →
    S = 40 →
    S = a / (1 - r) →
    a * r = (15 : ℝ) / 2 :=
by sorry

end NUMINAMATH_CALUDE_geometric_series_second_term_l3798_379852


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3798_379842

/-- Given a quadratic inequality a*x^2 + b*x + 1 > 0 with solution set {x | -1 < x < 1/3},
    prove that the product ab equals -6. -/
theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x, a*x^2 + b*x + 1 > 0 ↔ -1 < x ∧ x < 1/3) → 
  a * b = -6 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3798_379842


namespace NUMINAMATH_CALUDE_det_scaled_matrices_l3798_379839

-- Define a 2x2 matrix type
def Matrix2x2 := Fin 2 → Fin 2 → ℝ

-- Define the determinant function for 2x2 matrices
def det (A : Matrix2x2) : ℝ :=
  A 0 0 * A 1 1 - A 0 1 * A 1 0

-- Define a function to scale all elements of a matrix by a factor
def scaleMatrix (A : Matrix2x2) (k : ℝ) : Matrix2x2 :=
  λ i j ↦ k * A i j

-- Define a function to scale columns of a matrix by different factors
def scaleColumns (A : Matrix2x2) (k1 k2 : ℝ) : Matrix2x2 :=
  λ i j ↦ if j = 0 then k1 * A i j else k2 * A i j

-- State the theorem
theorem det_scaled_matrices (A : Matrix2x2) (h : det A = 3) :
  det (scaleMatrix A 3) = 27 ∧ det (scaleColumns A 4 2) = 24 := by
  sorry

end NUMINAMATH_CALUDE_det_scaled_matrices_l3798_379839


namespace NUMINAMATH_CALUDE_intersection_point_unique_l3798_379856

/-- The intersection point of two lines in 2D space -/
def intersection_point : ℚ × ℚ := (20/7, -11/7)

/-- First line equation: 5x - 3y = 19 -/
def line1 (x y : ℚ) : Prop := 5 * x - 3 * y = 19

/-- Second line equation: 6x + 2y = 14 -/
def line2 (x y : ℚ) : Prop := 6 * x + 2 * y = 14

theorem intersection_point_unique :
  let (x, y) := intersection_point
  (line1 x y ∧ line2 x y) ∧
  ∀ x' y', line1 x' y' ∧ line2 x' y' → x' = x ∧ y' = y :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_unique_l3798_379856


namespace NUMINAMATH_CALUDE_total_pets_is_45_l3798_379833

/-- The total number of pets given the specified conditions -/
def total_pets : ℕ :=
  let taylor_cats := 4
  let friends_with_double_pets := 3
  let friend1_dogs := 3
  let friend1_birds := 1
  let friend2_dogs := 5
  let friend2_cats := 2
  let friend3_reptiles := 2
  let friend3_birds := 3
  let friend3_cats := 1

  let total_cats := taylor_cats + friends_with_double_pets * (2 * taylor_cats) + friend2_cats + friend3_cats
  let total_dogs := friend1_dogs + friend2_dogs
  let total_birds := friend1_birds + friend3_birds
  let total_reptiles := friend3_reptiles

  total_cats + total_dogs + total_birds + total_reptiles

theorem total_pets_is_45 : total_pets = 45 := by
  sorry

end NUMINAMATH_CALUDE_total_pets_is_45_l3798_379833


namespace NUMINAMATH_CALUDE_min_value_x_plus_4y_l3798_379849

theorem min_value_x_plus_4y (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) 
  (h : 1/x + 1/(2*y) = 2) : 
  x + 4*y ≥ 3/2 + Real.sqrt 2 ∧ 
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 
    1/x₀ + 1/(2*y₀) = 2 ∧ 
    x₀ + 4*y₀ = 3/2 + Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_4y_l3798_379849
