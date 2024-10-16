import Mathlib

namespace NUMINAMATH_CALUDE_fraction_not_lowest_terms_count_l555_55534

theorem fraction_not_lowest_terms_count : 
  ∃ (S : Finset ℕ), 
    S.card = 102 ∧ 
    (∀ N ∈ S, 1 ≤ N ∧ N ≤ 1000 ∧ Nat.gcd (N^2 + 11) (N + 5) > 1) ∧
    (∀ N, 1 ≤ N ∧ N ≤ 1000 ∧ Nat.gcd (N^2 + 11) (N + 5) > 1 → N ∈ S) :=
by sorry

end NUMINAMATH_CALUDE_fraction_not_lowest_terms_count_l555_55534


namespace NUMINAMATH_CALUDE_age_ratio_l555_55562

def tom_age : ℝ := 40.5
def total_age : ℝ := 54

theorem age_ratio : 
  let antonette_age := total_age - tom_age
  tom_age / antonette_age = 3 := by sorry

end NUMINAMATH_CALUDE_age_ratio_l555_55562


namespace NUMINAMATH_CALUDE_difference_calculation_l555_55566

theorem difference_calculation (x y : ℝ) (hx : x = 497) (hy : y = 325) :
  2/5 * (3*x + 7*y) - 3/5 * (x * y) = -95408.6 := by
  sorry

end NUMINAMATH_CALUDE_difference_calculation_l555_55566


namespace NUMINAMATH_CALUDE_hexagon_extension_length_l555_55551

-- Define the regular hexagon
def RegularHexagon (C D E F G H : ℝ × ℝ) : Prop :=
  -- Add conditions for a regular hexagon with side length 4
  sorry

-- Define the extension of CD to Y
def ExtendCD (C D Y : ℝ × ℝ) : Prop :=
  dist C Y = 2 * dist C D

-- Main theorem
theorem hexagon_extension_length 
  (C D E F G H Y : ℝ × ℝ) 
  (hex : RegularHexagon C D E F G H) 
  (ext : ExtendCD C D Y) : 
  dist H Y = 6 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_extension_length_l555_55551


namespace NUMINAMATH_CALUDE_F_propagation_l555_55514

-- Define F as a proposition on natural numbers
variable (F : ℕ → Prop)

-- State the theorem
theorem F_propagation (h1 : ∀ k : ℕ, k > 0 → (F k → F (k + 1)))
                      (h2 : ¬ F 7) :
  ¬ F 6 ∧ ¬ F 5 := by
  sorry

end NUMINAMATH_CALUDE_F_propagation_l555_55514


namespace NUMINAMATH_CALUDE_train_speed_proof_l555_55516

theorem train_speed_proof (train_length bridge_length crossing_time : Real) 
  (h1 : train_length = 110)
  (h2 : bridge_length = 170)
  (h3 : crossing_time = 16.7986561075114) : 
  let total_distance := train_length + bridge_length
  let speed_ms := total_distance / crossing_time
  let speed_kmph := speed_ms * 3.6
  ⌊speed_kmph⌋ = 60 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_proof_l555_55516


namespace NUMINAMATH_CALUDE_quadratic_vertex_l555_55561

/-- The vertex of a quadratic function -/
theorem quadratic_vertex
  (a k c d : ℝ)
  (ha : a > 0)
  (hk : k ≠ b)  -- Note: 'b' is not defined, but kept as per the original problem
  (f : ℝ → ℝ)
  (hf : f = fun x ↦ a * x^2 + k * x + c + d) :
  let x₀ := -k / (2 * a)
  ∃ y₀, (x₀, y₀) = (-k / (2 * a), -k^2 / (4 * a) + c + d) ∧ 
       ∀ x, f x ≥ f x₀ :=
by sorry

end NUMINAMATH_CALUDE_quadratic_vertex_l555_55561


namespace NUMINAMATH_CALUDE_min_value_implies_a_bound_l555_55535

/-- The piecewise function f(x) --/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2 - 1 else a*x^2 - x + 2

/-- Theorem stating that if the minimum value of f(x) is -1, then a ≥ 1/12 --/
theorem min_value_implies_a_bound (a : ℝ) :
  (∀ x, f a x ≥ -1) ∧ (∃ x, f a x = -1) → a ≥ 1/12 :=
by sorry

end NUMINAMATH_CALUDE_min_value_implies_a_bound_l555_55535


namespace NUMINAMATH_CALUDE_gunny_bag_capacity_is_13_tons_l555_55591

/-- Represents the weight of a packet in pounds -/
def packet_weight : ℚ := 16 + 4 / 16

/-- Represents the number of packets -/
def num_packets : ℕ := 1840

/-- Represents the number of pounds in a ton -/
def pounds_per_ton : ℕ := 2300

/-- Represents the capacity of the gunny bag in tons -/
def gunny_bag_capacity : ℚ := (packet_weight * num_packets) / pounds_per_ton

theorem gunny_bag_capacity_is_13_tons : gunny_bag_capacity = 13 := by
  sorry

end NUMINAMATH_CALUDE_gunny_bag_capacity_is_13_tons_l555_55591


namespace NUMINAMATH_CALUDE_min_x_plus_y_l555_55533

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - (2*a + 1) * x + 2

-- State the theorem
theorem min_x_plus_y (a : ℝ) (h : a > 0) :
  (∀ x y : ℝ, y ≥ f a (|x|)) →
  (∃ x₀ y₀ : ℝ, y₀ ≥ f a (|x₀|) ∧ x₀ + y₀ = -a - 1/a) ∧
  (∀ x y : ℝ, y ≥ f a (|x|) → x + y ≥ -a - 1/a) :=
by sorry

end NUMINAMATH_CALUDE_min_x_plus_y_l555_55533


namespace NUMINAMATH_CALUDE_probability_three_white_balls_l555_55540

/-- The probability of drawing three white balls from a box containing 7 white balls and 8 black balls is 1/13. -/
theorem probability_three_white_balls (white_balls black_balls : ℕ) 
  (h1 : white_balls = 7) (h2 : black_balls = 8) : 
  (Nat.choose white_balls 3 : ℚ) / (Nat.choose (white_balls + black_balls) 3) = 1 / 13 := by
  sorry

#eval Nat.choose 7 3
#eval Nat.choose 15 3
#eval (35 : ℚ) / 455

end NUMINAMATH_CALUDE_probability_three_white_balls_l555_55540


namespace NUMINAMATH_CALUDE_last_remaining_number_l555_55538

def josephus_variant (n : ℕ) : ℕ :=
  let rec aux (k m : ℕ) : ℕ :=
    if k ≤ 1 then m
    else
      let m' := (m + 1) % k
      aux (k - 1) (2 * m' + 1)
  aux n 0

theorem last_remaining_number :
  josephus_variant 150 = 73 := by sorry

end NUMINAMATH_CALUDE_last_remaining_number_l555_55538


namespace NUMINAMATH_CALUDE_cubic_increasing_implies_positive_a_l555_55507

/-- A cubic function f(x) = ax^3 + x -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + x

/-- The property of f being increasing on all real numbers -/
def increasing_on_reals (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

/-- Theorem: If f(x) = ax^3 + x is increasing on all real numbers, then a > 0 -/
theorem cubic_increasing_implies_positive_a (a : ℝ) :
  increasing_on_reals (f a) → a > 0 :=
by sorry

end NUMINAMATH_CALUDE_cubic_increasing_implies_positive_a_l555_55507


namespace NUMINAMATH_CALUDE_rectangular_plot_poles_l555_55525

/-- The number of poles needed to enclose a rectangular plot -/
def poles_needed (length width pole_distance : ℕ) : ℕ :=
  ((2 * (length + width) + pole_distance - 1) / pole_distance : ℕ)

/-- Theorem: A 135m by 80m plot with poles 7m apart needs 62 poles -/
theorem rectangular_plot_poles :
  poles_needed 135 80 7 = 62 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_poles_l555_55525


namespace NUMINAMATH_CALUDE_sqrt_735_simplification_l555_55536

theorem sqrt_735_simplification : Real.sqrt 735 = 7 * Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_735_simplification_l555_55536


namespace NUMINAMATH_CALUDE_coin_toss_probability_l555_55570

theorem coin_toss_probability (n : ℕ) : (∀ k : ℕ, k < n → 1 - (1/2)^k < 15/16) ∧ 1 - (1/2)^n ≥ 15/16 → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_coin_toss_probability_l555_55570


namespace NUMINAMATH_CALUDE_max_value_theorem_l555_55522

theorem max_value_theorem (a b c d : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0)
  (h1 : a^2 + b^2 - c^2 - d^2 = 0)
  (h2 : a^2 - b^2 - c^2 + d^2 = 56/53 * (b*c + a*d)) :
  (∀ x y z w : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0 ∧ 
    x^2 + y^2 - z^2 - w^2 = 0 ∧
    x^2 - y^2 - z^2 + w^2 = 56/53 * (y*z + x*w) →
    (x*y + z*w) / (y*z + x*w) ≤ (a*b + c*d) / (b*c + a*d)) ∧
  (a*b + c*d) / (b*c + a*d) = 45/53 :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l555_55522


namespace NUMINAMATH_CALUDE_cone_rolling_ratio_l555_55589

/-- Represents a right circular cone. -/
structure RightCircularCone where
  r : ℝ  -- base radius
  h : ℝ  -- height

/-- Represents the rolling properties of the cone. -/
structure RollingCone extends RightCircularCone where
  rotations : ℕ
  no_slip : Bool

theorem cone_rolling_ratio (c : RollingCone) 
  (h_positive : c.h > 0)
  (r_positive : c.r > 0)
  (twenty_rotations : c.rotations = 20)
  (no_slip : c.no_slip = true) :
  c.h / c.r = Real.sqrt 399 :=
sorry

end NUMINAMATH_CALUDE_cone_rolling_ratio_l555_55589


namespace NUMINAMATH_CALUDE_min_value_trig_expression_l555_55517

theorem min_value_trig_expression (α β : ℝ) :
  (3 * Real.cos α + 6 * Real.sin β - 10)^2 + (3 * Real.sin α + 6 * Real.cos β - 18)^2 ≥ 121 ∧
  ∃ α₀ β₀ : ℝ, (3 * Real.cos α₀ + 6 * Real.sin β₀ - 10)^2 + (3 * Real.sin α₀ + 6 * Real.cos β₀ - 18)^2 = 121 :=
by sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_l555_55517


namespace NUMINAMATH_CALUDE_ball_probability_l555_55571

theorem ball_probability (x : ℕ) : 
  (4 : ℝ) / (4 + x) = (2 : ℝ) / 5 → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_ball_probability_l555_55571


namespace NUMINAMATH_CALUDE_negation_of_existential_proposition_negation_of_specific_proposition_l555_55552

theorem negation_of_existential_proposition (p : ℝ → Prop) :
  (¬ ∃ x : ℝ, p x) ↔ (∀ x : ℝ, ¬ p x) := by sorry

theorem negation_of_specific_proposition :
  (¬ ∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) ↔ (∀ x : ℝ, x^2 + 2*x + 2 > 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existential_proposition_negation_of_specific_proposition_l555_55552


namespace NUMINAMATH_CALUDE_max_value_of_sine_plus_one_l555_55576

theorem max_value_of_sine_plus_one :
  ∀ x : ℝ, 1 + Real.sin x ≤ 2 ∧ ∃ x : ℝ, 1 + Real.sin x = 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_sine_plus_one_l555_55576


namespace NUMINAMATH_CALUDE_shoe_shirt_earnings_l555_55575

theorem shoe_shirt_earnings : 
  let shoe_pairs : ℕ := 6
  let shoe_price : ℕ := 3
  let shirt_count : ℕ := 18
  let shirt_price : ℕ := 2
  let total_earnings := shoe_pairs * shoe_price + shirt_count * shirt_price
  let people_count : ℕ := 2
  (total_earnings / people_count : ℕ) = 27 := by sorry

end NUMINAMATH_CALUDE_shoe_shirt_earnings_l555_55575


namespace NUMINAMATH_CALUDE_molecular_weight_calculation_l555_55598

theorem molecular_weight_calculation (total_weight : ℝ) (number_of_moles : ℝ) 
  (h1 : total_weight = 2376)
  (h2 : number_of_moles = 8) : 
  total_weight / number_of_moles = 297 := by
sorry

end NUMINAMATH_CALUDE_molecular_weight_calculation_l555_55598


namespace NUMINAMATH_CALUDE_calculator_sale_loss_l555_55545

theorem calculator_sale_loss :
  ∀ (x y : ℝ),
    x * (1 + 0.2) = 60 →
    y * (1 - 0.2) = 60 →
    60 + 60 - (x + y) = -5 :=
by
  sorry

end NUMINAMATH_CALUDE_calculator_sale_loss_l555_55545


namespace NUMINAMATH_CALUDE_bumper_cars_cost_l555_55547

/-- The cost of bumper cars given the costs of other attractions and ticket information -/
theorem bumper_cars_cost 
  (total_cost : ℕ → ℕ → ℕ)  -- Function to calculate total cost
  (current_tickets : ℕ)     -- Current number of tickets
  (additional_tickets : ℕ)  -- Additional tickets needed
  (ferris_wheel_cost : ℕ)   -- Cost of Ferris wheel
  (roller_coaster_cost : ℕ) -- Cost of roller coaster
  (h1 : current_tickets = 5)
  (h2 : additional_tickets = 8)
  (h3 : ferris_wheel_cost = 5)
  (h4 : roller_coaster_cost = 4)
  (h5 : ∀ x y, total_cost x y = x + y) -- Definition of total cost function
  : ∃ (bumper_cars_cost : ℕ), 
    total_cost current_tickets additional_tickets = 
    ferris_wheel_cost + roller_coaster_cost + bumper_cars_cost ∧ 
    bumper_cars_cost = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_bumper_cars_cost_l555_55547


namespace NUMINAMATH_CALUDE_coefficient_sum_l555_55594

theorem coefficient_sum (b₅ b₄ b₃ b₂ b₁ b₀ : ℝ) :
  (∀ x, (2*x + 3)^5 = b₅*x^5 + b₄*x^4 + b₃*x^3 + b₂*x^2 + b₁*x + b₀) →
  b₅ + b₄ + b₃ + b₂ + b₁ + b₀ = 3125 := by
sorry

end NUMINAMATH_CALUDE_coefficient_sum_l555_55594


namespace NUMINAMATH_CALUDE_right_triangle_squares_area_l555_55588

theorem right_triangle_squares_area (a b : ℝ) (ha : a = 3) (hb : b = 9) :
  let c := Real.sqrt (a^2 + b^2)
  a^2 + b^2 + c^2 + (1/2 * a * b) = 193.5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_squares_area_l555_55588


namespace NUMINAMATH_CALUDE_intersection_A_B_l555_55587

def A : Set ℕ := {x | 0 < x ∧ x < 6}
def B : Set ℕ := {2, 4, 6, 8}

theorem intersection_A_B : A ∩ B = {2, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l555_55587


namespace NUMINAMATH_CALUDE_equation_solution_l555_55556

theorem equation_solution (x : ℝ) : 5 * x^2 + 4 = 3 * x + 9 → (10 * x - 3)^2 = 109 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l555_55556


namespace NUMINAMATH_CALUDE_joshua_friends_count_l555_55586

def total_skittles : ℕ := 40
def skittles_per_friend : ℕ := 8

theorem joshua_friends_count : 
  total_skittles / skittles_per_friend = 5 := by sorry

end NUMINAMATH_CALUDE_joshua_friends_count_l555_55586


namespace NUMINAMATH_CALUDE_final_short_oak_count_l555_55513

/-- The number of short oak trees in the park after planting -/
def short_oak_trees_after_planting (current : ℕ) (to_plant : ℕ) : ℕ :=
  current + to_plant

/-- Theorem stating the number of short oak trees after planting -/
theorem final_short_oak_count :
  short_oak_trees_after_planting 3 9 = 12 := by
  sorry

end NUMINAMATH_CALUDE_final_short_oak_count_l555_55513


namespace NUMINAMATH_CALUDE_base_8_5624_equals_2964_l555_55581

def base_8_to_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

theorem base_8_5624_equals_2964 : 
  base_8_to_10 [4, 2, 6, 5] = 2964 := by
  sorry

end NUMINAMATH_CALUDE_base_8_5624_equals_2964_l555_55581


namespace NUMINAMATH_CALUDE_sum_of_common_divisors_l555_55568

def number_list : List Int := [24, 48, -18, 108, 72]

def is_common_divisor (d : Nat) : Bool :=
  number_list.all (fun n => n % d == 0)

def common_divisors : List Nat :=
  (List.range 108).filter is_common_divisor

theorem sum_of_common_divisors : (common_divisors.sum) = 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_common_divisors_l555_55568


namespace NUMINAMATH_CALUDE_smoking_chronic_bronchitis_relationship_l555_55580

-- Define the confidence level
def confidence_level : Real := 0.99

-- Define the relationship between smoking and chronic bronchitis
def smoking_related_to_chronic_bronchitis : Prop := True

-- Define a sample of smokers
def sample_size : Nat := 100

-- Define the possibility of no chronic bronchitis cases in the sample
def possible_no_cases : Prop := True

-- Theorem statement
theorem smoking_chronic_bronchitis_relationship 
  (h1 : confidence_level > 0.99)
  (h2 : smoking_related_to_chronic_bronchitis) :
  possible_no_cases := by
  sorry

end NUMINAMATH_CALUDE_smoking_chronic_bronchitis_relationship_l555_55580


namespace NUMINAMATH_CALUDE_theater_ticket_pricing_l555_55554

theorem theater_ticket_pricing (total_tickets : ℕ) (total_revenue : ℕ) 
  (balcony_price : ℕ) (balcony_orchestra_diff : ℕ) :
  total_tickets = 360 →
  total_revenue = 3320 →
  balcony_price = 8 →
  balcony_orchestra_diff = 140 →
  ∃ (orchestra_price : ℕ), 
    orchestra_price = 12 ∧
    orchestra_price * (total_tickets - balcony_orchestra_diff) / 2 + 
    balcony_price * (total_tickets + balcony_orchestra_diff) / 2 = total_revenue :=
by sorry

end NUMINAMATH_CALUDE_theater_ticket_pricing_l555_55554


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_median_relation_l555_55529

/-- In a right triangle, the square of the hypotenuse is equal to four-fifths of the sum of squares of the medians to the other two sides. -/
theorem right_triangle_hypotenuse_median_relation (a b c k_a k_b : ℝ) :
  a > 0 → b > 0 → c > 0 → k_a > 0 → k_b > 0 →
  a^2 + b^2 = c^2 →  -- Right triangle condition
  k_a^2 = (2*b^2 + 2*c^2 - a^2) / 4 →  -- Definition of k_a
  k_b^2 = (2*a^2 + 2*c^2 - b^2) / 4 →  -- Definition of k_b
  c^2 = (4/5) * (k_a^2 + k_b^2) := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_median_relation_l555_55529


namespace NUMINAMATH_CALUDE_f_theorem_l555_55585

def f_properties (f : ℝ → ℝ) : Prop :=
  Continuous f ∧
  (∀ x, f (-x) = f x) ∧
  (∀ x₁ x₂, x₁ > 0 → x₂ > 0 → x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) < 0) ∧
  f (-1) = 0

theorem f_theorem (f : ℝ → ℝ) (h : f_properties f) :
  f 3 > f 4 ∧
  (∀ m, f (m - 1) < f 2 → m < -1 ∨ m > 3) ∧
  ∃ M, ∀ x, f x ≤ M :=
sorry

end NUMINAMATH_CALUDE_f_theorem_l555_55585


namespace NUMINAMATH_CALUDE_evaluate_expression_l555_55526

theorem evaluate_expression : 2003^3 - 2002 * 2003^2 - 2002^2 * 2003 + 2002^3 = 4005 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l555_55526


namespace NUMINAMATH_CALUDE_line_tangent_to_parabola_l555_55592

/-- A parabola defined by y = 2x^2 -/
def parabola (x y : ℝ) : Prop := y = 2 * x^2

/-- The point A on the parabola -/
def point_A : ℝ × ℝ := (-1, 2)

/-- The equation of line l -/
def line_l (x y : ℝ) : Prop := 4 * x + y + 2 = 0

/-- Theorem stating that line l is tangent to the parabola at point A -/
theorem line_tangent_to_parabola :
  parabola (point_A.1) (point_A.2) ∧
  line_l (point_A.1) (point_A.2) ∧
  ∀ x y : ℝ, parabola x y ∧ line_l x y → (x, y) = point_A :=
sorry

end NUMINAMATH_CALUDE_line_tangent_to_parabola_l555_55592


namespace NUMINAMATH_CALUDE_cube_surface_area_difference_l555_55577

theorem cube_surface_area_difference (large_cube_volume : ℕ) (num_small_cubes : ℕ) (small_cube_volume : ℕ) : 
  large_cube_volume = 6859 →
  num_small_cubes = 6859 →
  small_cube_volume = 1 →
  (num_small_cubes * 6 * small_cube_volume^(2/3) : ℕ) - (6 * large_cube_volume^(2/3) : ℕ) = 38988 := by
  sorry

#eval (6859 * 6 * 1^(2/3) : ℕ) - (6 * 6859^(2/3) : ℕ)

end NUMINAMATH_CALUDE_cube_surface_area_difference_l555_55577


namespace NUMINAMATH_CALUDE_binomial_20_10_l555_55542

theorem binomial_20_10 (h1 : Nat.choose 18 8 = 43758) 
                       (h2 : Nat.choose 18 9 = 48620) 
                       (h3 : Nat.choose 18 10 = 43758) : 
  Nat.choose 20 10 = 184756 := by
  sorry

end NUMINAMATH_CALUDE_binomial_20_10_l555_55542


namespace NUMINAMATH_CALUDE_square_root_of_sixteen_l555_55584

theorem square_root_of_sixteen (x : ℝ) : (x + 3) ^ 2 = 16 → x = 1 ∨ x = -7 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_sixteen_l555_55584


namespace NUMINAMATH_CALUDE_complex_number_problem_l555_55590

-- Define the complex number z
variable (z : ℂ)

-- Define the conditions
variable (h1 : ∃ (r : ℝ), z + 2*I = r)
variable (h2 : ∃ (t : ℝ), z - 4 = t*I)

-- Define m as a real number
variable (m : ℝ)

-- Define the fourth quadrant condition
def in_fourth_quadrant (w : ℂ) : Prop :=
  w.re > 0 ∧ w.im < 0

-- Theorem statement
theorem complex_number_problem :
  z = 4 - 2*I ∧
  (in_fourth_quadrant ((z + m*I)^2) ↔ -2 < m ∧ m < 2) :=
sorry

end NUMINAMATH_CALUDE_complex_number_problem_l555_55590


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l555_55512

theorem line_segment_endpoint (x : ℝ) : x > 0 ∧ 
  Real.sqrt ((x - 2)^2 + (5 - 2)^2) = 8 → x = 2 + Real.sqrt 55 := by
  sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l555_55512


namespace NUMINAMATH_CALUDE_rectangle_composition_l555_55531

/-- Given a rectangle ABCD composed of six identical smaller rectangles,
    prove that the length y is 20 -/
theorem rectangle_composition (x y : ℝ) : 
  (3 * y) * (2 * x) = 2400 →  -- Area of ABCD
  y = 20 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_composition_l555_55531


namespace NUMINAMATH_CALUDE_hanging_spheres_mass_ratio_l555_55543

/-- Given two hanging spheres with masses m₁ and m₂, where the tension in the upper string
    is twice the tension in the lower string, prove that the ratio of masses m₁/m₂ = 1 -/
theorem hanging_spheres_mass_ratio (m₁ m₂ : ℝ) (g : ℝ) (h : g > 0) : 
  (m₁ * g + m₂ * g = 2 * (m₂ * g)) → m₁ / m₂ = 1 := by
  sorry

end NUMINAMATH_CALUDE_hanging_spheres_mass_ratio_l555_55543


namespace NUMINAMATH_CALUDE_bruce_son_age_l555_55541

/-- Bruce's current age -/
def bruce_age : ℕ := 36

/-- Number of years in the future -/
def years_future : ℕ := 6

/-- Bruce's son's current age -/
def son_age : ℕ := 8

theorem bruce_son_age :
  (bruce_age + years_future) = 3 * (son_age + years_future) :=
sorry

end NUMINAMATH_CALUDE_bruce_son_age_l555_55541


namespace NUMINAMATH_CALUDE_teairra_closet_count_l555_55524

/-- The number of shirts and pants that are neither plaid nor purple -/
def non_plaid_purple_count (total_shirts : ℕ) (total_pants : ℕ) (plaid_shirts : ℕ) (purple_pants : ℕ) : ℕ :=
  (total_shirts - plaid_shirts) + (total_pants - purple_pants)

theorem teairra_closet_count :
  non_plaid_purple_count 5 24 3 5 = 21 := by
  sorry

end NUMINAMATH_CALUDE_teairra_closet_count_l555_55524


namespace NUMINAMATH_CALUDE_tangent_line_y_intercept_l555_55527

/-- A circle with a given center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line tangent to two circles -/
structure TangentLine where
  circle1 : Circle
  circle2 : Circle
  tangentPoint1 : ℝ × ℝ
  tangentPoint2 : ℝ × ℝ

/-- The y-intercept of the tangent line to two circles -/
def yIntercept (line : TangentLine) : ℝ := sorry

theorem tangent_line_y_intercept :
  let c1 : Circle := { center := (3, 0), radius := 3 }
  let c2 : Circle := { center := (7, 0), radius := 2 }
  let line : TangentLine := {
    circle1 := c1,
    circle2 := c2,
    tangentPoint1 := sorry,  -- Exact point not given, but in first quadrant
    tangentPoint2 := sorry   -- Exact point not given, but in first quadrant
  }
  yIntercept line = 4.5 := by sorry

end NUMINAMATH_CALUDE_tangent_line_y_intercept_l555_55527


namespace NUMINAMATH_CALUDE_midpoint_octagon_area_ratio_l555_55579

/-- A regular octagon -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ
  is_regular : sorry  -- Additional properties to ensure the octagon is regular

/-- The octagon formed by joining the midpoints of a regular octagon's sides -/
def midpointOctagon (o : RegularOctagon) : RegularOctagon :=
  sorry

/-- The area of a regular octagon -/
def area (o : RegularOctagon) : ℝ :=
  sorry

/-- The theorem stating that the area of the midpoint octagon is 3/4 of the original octagon -/
theorem midpoint_octagon_area_ratio (o : RegularOctagon) :
  area (midpointOctagon o) = (3/4) * area o :=
sorry

end NUMINAMATH_CALUDE_midpoint_octagon_area_ratio_l555_55579


namespace NUMINAMATH_CALUDE_largest_n_for_trig_inequality_l555_55564

theorem largest_n_for_trig_inequality : 
  (∃ (n : ℕ), n > 0 ∧ ∀ (x : ℝ), (Real.sin x)^n + (Real.cos x)^n ≥ 2/n) ∧ 
  (∀ (n : ℕ), n > 2 → ∃ (x : ℝ), (Real.sin x)^n + (Real.cos x)^n < 2/n) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_for_trig_inequality_l555_55564


namespace NUMINAMATH_CALUDE_yellow_candles_count_l555_55597

/-- The number of yellow candles on a birthday cake --/
def yellow_candles (total_candles red_candles blue_candles : ℕ) : ℕ :=
  total_candles - (red_candles + blue_candles)

/-- Theorem: The number of yellow candles is 27 --/
theorem yellow_candles_count :
  yellow_candles 79 14 38 = 27 := by
  sorry

end NUMINAMATH_CALUDE_yellow_candles_count_l555_55597


namespace NUMINAMATH_CALUDE_investment_proportion_l555_55583

/-- Given two investors X and Y, where X invested 5000 and their profit is divided in the ratio 2:6,
    prove that Y's investment is 15000. -/
theorem investment_proportion (x_investment y_investment : ℕ) (profit_ratio_x profit_ratio_y : ℕ) :
  x_investment = 5000 →
  profit_ratio_x = 2 →
  profit_ratio_y = 6 →
  y_investment = 15000 :=
by
  sorry

end NUMINAMATH_CALUDE_investment_proportion_l555_55583


namespace NUMINAMATH_CALUDE_total_cookie_time_l555_55548

/-- The time it takes to make black & white cookies -/
def cookie_making_time (batter_time baking_time cooling_time white_icing_time chocolate_icing_time : ℕ) : ℕ :=
  batter_time + baking_time + cooling_time + white_icing_time + chocolate_icing_time

/-- Theorem stating that the total time to make black & white cookies is 100 minutes -/
theorem total_cookie_time :
  ∃ (batter_time cooling_time : ℕ),
    batter_time = 10 ∧
    cooling_time = 15 ∧
    cookie_making_time batter_time 15 cooling_time 30 30 = 100 := by
  sorry

end NUMINAMATH_CALUDE_total_cookie_time_l555_55548


namespace NUMINAMATH_CALUDE_roper_lawn_cutting_l555_55509

/-- The number of times Mr. Roper cuts his lawn per month from April to September -/
def summer_cuts : ℕ := 15

/-- The number of times Mr. Roper cuts his lawn per month from October to March -/
def winter_cuts : ℕ := 3

/-- The number of months in each season (summer and winter) -/
def months_per_season : ℕ := 6

/-- The total number of months in a year -/
def months_in_year : ℕ := 12

/-- The average number of times Mr. Roper cuts his lawn per month -/
def average_cuts : ℚ := (summer_cuts * months_per_season + winter_cuts * months_per_season) / months_in_year

theorem roper_lawn_cutting :
  average_cuts = 9 := by sorry

end NUMINAMATH_CALUDE_roper_lawn_cutting_l555_55509


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l555_55582

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x ≥ 1}
def B : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 1 ≤ x ∧ x < 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l555_55582


namespace NUMINAMATH_CALUDE_complex_sum_sixth_power_l555_55573

theorem complex_sum_sixth_power : 
  let z₁ : ℂ := (-1 + Complex.I * Real.sqrt 3) / 2
  let z₂ : ℂ := (-1 - Complex.I * Real.sqrt 3) / 2
  z₁^6 + z₂^6 = 2 := by sorry

end NUMINAMATH_CALUDE_complex_sum_sixth_power_l555_55573


namespace NUMINAMATH_CALUDE_x_plus_inv_x_eight_l555_55500

theorem x_plus_inv_x_eight (x : ℝ) (h : x + 1/x = 8) : x^4 + 1/x^4 = 3842 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_inv_x_eight_l555_55500


namespace NUMINAMATH_CALUDE_estimate_fish_population_l555_55502

/-- Estimates the number of fish in a pond using the mark-recapture method. -/
theorem estimate_fish_population
  (initially_tagged : ℕ)
  (second_catch : ℕ)
  (tagged_in_second_catch : ℕ)
  (h1 : initially_tagged = 100)
  (h2 : second_catch = 300)
  (h3 : tagged_in_second_catch = 15) :
  (initially_tagged * second_catch) / tagged_in_second_catch = 2000 := by
  sorry

#check estimate_fish_population

end NUMINAMATH_CALUDE_estimate_fish_population_l555_55502


namespace NUMINAMATH_CALUDE_a_4_equals_8_l555_55559

def a (n : ℕ) : ℤ := (-1)^n * (2 * n)

theorem a_4_equals_8 : a 4 = 8 := by sorry

end NUMINAMATH_CALUDE_a_4_equals_8_l555_55559


namespace NUMINAMATH_CALUDE_min_value_of_linear_function_l555_55569

theorem min_value_of_linear_function :
  ∃ (m : ℝ), ∀ (x y : ℝ), 2*x + 3*y ≥ m ∧ ∃ (x₀ y₀ : ℝ), 2*x₀ + 3*y₀ = m :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_linear_function_l555_55569


namespace NUMINAMATH_CALUDE_vector_sum_example_l555_55574

theorem vector_sum_example :
  let v1 : Fin 3 → ℝ := ![3, -2, 7]
  let v2 : Fin 3 → ℝ := ![-1, 5, -3]
  v1 + v2 = ![2, 3, 4] := by
sorry

end NUMINAMATH_CALUDE_vector_sum_example_l555_55574


namespace NUMINAMATH_CALUDE_inequality_reversal_l555_55572

theorem inequality_reversal (a b : ℝ) (h : a > b) : a / (-2) < b / (-2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_reversal_l555_55572


namespace NUMINAMATH_CALUDE_q4_value_l555_55511

def sequence_a : ℕ → ℝ
| 0 => 1  -- We define a₁ = 1 based on the solution
| n + 1 => 2 * sequence_a n + 4

def sequence_q : ℕ → ℝ
| 0 => 17  -- We define q₁ = 17 to satisfy q₄ = 76
| n + 1 => 4 * sequence_q n + 8

theorem q4_value :
  sequence_a 4 = sequence_q 3 ∧ 
  sequence_a 6 = 316 → 
  sequence_q 3 = 76 := by
sorry

end NUMINAMATH_CALUDE_q4_value_l555_55511


namespace NUMINAMATH_CALUDE_max_profit_is_32500_l555_55567

/-- Profit function given price increase x -/
def profit (x : ℝ) : ℝ := -5 * x^2 + 500 * x + 20000

/-- The initial purchase price per item -/
def initial_price : ℝ := 80

/-- The total number of items -/
def total_items : ℝ := 1000

/-- The base selling price at which all items would be sold -/
def base_selling_price : ℝ := 100

/-- The decrease in sales volume for each yuan increase in price -/
def sales_decrease_rate : ℝ := 5

/-- The optimal price increase that maximizes profit -/
def optimal_price_increase : ℝ := 50

theorem max_profit_is_32500 :
  ∃ (x : ℝ), profit x = 32500 ∧ ∀ (y : ℝ), profit y ≤ profit x :=
sorry

end NUMINAMATH_CALUDE_max_profit_is_32500_l555_55567


namespace NUMINAMATH_CALUDE_max_value_of_g_l555_55550

def g (x : ℝ) : ℝ := 5 * x^2 - 2 * x^4

theorem max_value_of_g :
  ∃ (x : ℝ), 0 ≤ x ∧ x ≤ Real.sqrt 2 ∧
  g x = 25/8 ∧
  ∀ (y : ℝ), 0 ≤ y ∧ y ≤ Real.sqrt 2 → g y ≤ g x :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_g_l555_55550


namespace NUMINAMATH_CALUDE_quadratic_root_implies_c_value_l555_55593

theorem quadratic_root_implies_c_value (c : ℝ) :
  (∀ x : ℝ, (3/2) * x^2 + 11*x + c = 0 ↔ x = (-11 + Real.sqrt 7) / 3 ∨ x = (-11 - Real.sqrt 7) / 3) →
  c = 19 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_c_value_l555_55593


namespace NUMINAMATH_CALUDE_log_relationship_l555_55503

theorem log_relationship (a b : ℝ) : 
  a = Real.log 256 / Real.log 8 → b = Real.log 16 / Real.log 2 → a = (2 * b) / 3 := by
sorry

end NUMINAMATH_CALUDE_log_relationship_l555_55503


namespace NUMINAMATH_CALUDE_cards_given_to_jeff_l555_55518

/-- Proves that the number of cards Nell gave to Jeff is equal to the difference between her initial number of cards and the number of cards she has left. -/
theorem cards_given_to_jeff (initial_cards : ℕ) (remaining_cards : ℕ) 
  (h1 : initial_cards = 304)
  (h2 : remaining_cards = 276)
  (h3 : initial_cards ≥ remaining_cards) :
  initial_cards - remaining_cards = 28 := by
  sorry

end NUMINAMATH_CALUDE_cards_given_to_jeff_l555_55518


namespace NUMINAMATH_CALUDE_last_two_digits_of_7_power_last_two_digits_of_7_2017_l555_55508

theorem last_two_digits_of_7_power (n : ℕ) :
  (7^n) % 100 = (7^(n % 4 + 4)) % 100 :=
sorry

theorem last_two_digits_of_7_2017 :
  (7^2017) % 100 = 49 :=
sorry

end NUMINAMATH_CALUDE_last_two_digits_of_7_power_last_two_digits_of_7_2017_l555_55508


namespace NUMINAMATH_CALUDE_cylinder_height_l555_55560

/-- The height of a cylinder given its lateral surface area and volume -/
theorem cylinder_height (r h : ℝ) (h1 : 2 * π * r * h = 12 * π) (h2 : π * r^2 * h = 12 * π) : h = 3 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_height_l555_55560


namespace NUMINAMATH_CALUDE_orthocenter_from_circumcenter_l555_55599

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a tetrahedron -/
structure Tetrahedron where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Represents a sphere -/
structure Sphere where
  center : Point3D
  radius : ℝ

/-- Checks if a point is the circumcenter of a triangle -/
def isCircumcenter (O : Point3D) (A B C : Point3D) : Prop := sorry

/-- Checks if a point is the orthocenter of a triangle -/
def isOrthocenter (H : Point3D) (A B C : Point3D) : Prop := sorry

/-- Checks if a sphere is inscribed in a tetrahedron -/
def isInscribed (s : Sphere) (t : Tetrahedron) : Prop := sorry

/-- Checks if a sphere touches a plane at a point -/
def touchesPlaneAt (s : Sphere) (p : Point3D) : Prop := sorry

/-- Checks if a sphere touches the planes of the other faces of a tetrahedron externally -/
def touchesOtherFacesExternally (s : Sphere) (t : Tetrahedron) : Prop := sorry

theorem orthocenter_from_circumcenter 
  (t : Tetrahedron) 
  (s1 s2 : Sphere) 
  (H O : Point3D) :
  isInscribed s1 t →
  touchesPlaneAt s1 H →
  touchesPlaneAt s2 O →
  touchesOtherFacesExternally s2 t →
  isCircumcenter O t.A t.B t.C →
  isOrthocenter H t.A t.B t.C := by
  sorry

end NUMINAMATH_CALUDE_orthocenter_from_circumcenter_l555_55599


namespace NUMINAMATH_CALUDE_normal_block_volume_l555_55521

-- Define the volume of a normal block
def normal_volume (w d l : ℝ) : ℝ := w * d * l

-- Define the volume of a large block
def large_volume (w d l : ℝ) : ℝ := (2*w) * (2*d) * (2*l)

-- Theorem statement
theorem normal_block_volume :
  ∀ w d l : ℝ, w > 0 → d > 0 → l > 0 →
  large_volume w d l = 32 →
  normal_volume w d l = 4 := by
  sorry

end NUMINAMATH_CALUDE_normal_block_volume_l555_55521


namespace NUMINAMATH_CALUDE_floor_ceil_sum_l555_55505

theorem floor_ceil_sum : ⌊(-1.001 : ℝ)⌋ + ⌈(3.999 : ℝ)⌉ + ⌊(0.998 : ℝ)⌋ = 2 := by
  sorry

end NUMINAMATH_CALUDE_floor_ceil_sum_l555_55505


namespace NUMINAMATH_CALUDE_expression_value_l555_55506

theorem expression_value (x y z : ℝ) 
  (hxy : x ≠ y) (hxz : x ≠ z) (hyz : y ≠ z)
  (hx : x ≠ 2) (hy : y ≠ 3) (hz : z ≠ 5) :
  ((x - 2) / (3 - z) * (y - 3) / (5 - x) * (z - 5) / (2 - y))^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l555_55506


namespace NUMINAMATH_CALUDE_specific_hexagon_area_l555_55557

/-- A hexagon formed by cutting a triangular corner from a square -/
structure CornerCutHexagon where
  sides : Fin 6 → ℕ
  is_valid_sides : (sides 0) + (sides 1) + (sides 2) + (sides 3) + (sides 4) + (sides 5) = 11 + 17 + 14 + 23 + 17 + 20

/-- The area of the hexagon -/
def hexagon_area (h : CornerCutHexagon) : ℕ :=
  sorry

/-- Theorem stating that the area of the specific hexagon is 1096 -/
theorem specific_hexagon_area : ∃ h : CornerCutHexagon, hexagon_area h = 1096 := by
  sorry

end NUMINAMATH_CALUDE_specific_hexagon_area_l555_55557


namespace NUMINAMATH_CALUDE_smallest_n_with_seven_in_squares_l555_55555

/-- A function that checks if a natural number contains the digit 7 -/
def contains_seven (n : ℕ) : Prop :=
  ∃ k m : ℕ, n = 10 * k + 7 + 10 * m

/-- The smallest natural number n such that both n^2 and (n+1)^2 contain the digit 7 -/
theorem smallest_n_with_seven_in_squares :
  ∃ n : ℕ, n = 26 ∧
    contains_seven (n^2) ∧
    contains_seven ((n+1)^2) ∧
    ∀ m : ℕ, m < n → ¬(contains_seven (m^2) ∧ contains_seven ((m+1)^2)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_with_seven_in_squares_l555_55555


namespace NUMINAMATH_CALUDE_intersection_point_Q_l555_55537

-- Define the circles
def circle1 (x y r : ℝ) : Prop := (x + 1)^2 + (y - 1)^2 = r^2
def circle2 (x y R : ℝ) : Prop := (x - 2)^2 + (y + 2)^2 = R^2

-- Define the intersection points
def P : ℝ × ℝ := (1, 2)
def Q : ℝ × ℝ := (-2, -1)

-- Theorem statement
theorem intersection_point_Q :
  ∀ (r R : ℝ),
  (∃ (x y : ℝ), circle1 x y r ∧ circle2 x y R) →  -- Circles intersect
  circle1 P.1 P.2 r →                            -- P is on circle1
  circle2 P.1 P.2 R →                            -- P is on circle2
  circle1 Q.1 Q.2 r ∧ circle2 Q.1 Q.2 R          -- Q is on both circles
  := by sorry

end NUMINAMATH_CALUDE_intersection_point_Q_l555_55537


namespace NUMINAMATH_CALUDE_geometric_sequence_20th_term_l555_55558

/-- A geometric sequence is defined by its first term and common ratio -/
def geometric_sequence (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * r^(n - 1)

/-- Theorem: In a geometric sequence where the 5th term is 5 and the 12th term is 1280, the 20th term is 2621440 -/
theorem geometric_sequence_20th_term 
  (a : ℝ) (r : ℝ) 
  (h1 : geometric_sequence a r 5 = 5)
  (h2 : geometric_sequence a r 12 = 1280) :
  geometric_sequence a r 20 = 2621440 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_20th_term_l555_55558


namespace NUMINAMATH_CALUDE_subset_iff_range_l555_55578

def A : Set ℝ := {x | x^2 - 4*x + 3 ≤ 0}
def B (a : ℝ) : Set ℝ := {x | x^2 - a*x < x - a}

theorem subset_iff_range (a : ℝ) : A ⊇ B a ↔ 1 ≤ a ∧ a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_subset_iff_range_l555_55578


namespace NUMINAMATH_CALUDE_smallest_BD_is_five_l555_55596

/-- Represents a quadrilateral with side lengths and an angle -/
structure Quadrilateral :=
  (AB BC CD DA : ℝ)
  (angleBDA : ℝ)

/-- The smallest possible integer value of BD in the given quadrilateral -/
def smallest_integer_BD (q : Quadrilateral) : ℕ :=
  sorry

/-- Theorem stating the smallest possible integer value of BD -/
theorem smallest_BD_is_five (q : Quadrilateral) 
  (h1 : q.AB = 7)
  (h2 : q.BC = 15)
  (h3 : q.CD = 7)
  (h4 : q.DA = 11)
  (h5 : q.angleBDA = 90) :
  smallest_integer_BD q = 5 :=
sorry

end NUMINAMATH_CALUDE_smallest_BD_is_five_l555_55596


namespace NUMINAMATH_CALUDE_exam_fail_percentage_l555_55544

theorem exam_fail_percentage 
  (total_candidates : ℕ) 
  (girls : ℕ) 
  (pass_rate : ℚ) 
  (h1 : total_candidates = 2000)
  (h2 : girls = 900)
  (h3 : pass_rate = 32/100) :
  let boys := total_candidates - girls
  let passed_candidates := (boys * pass_rate).floor + (girls * pass_rate).floor
  let failed_candidates := total_candidates - passed_candidates
  let fail_percentage := (failed_candidates : ℚ) / total_candidates * 100
  fail_percentage = 68 := by sorry

end NUMINAMATH_CALUDE_exam_fail_percentage_l555_55544


namespace NUMINAMATH_CALUDE_complex_equation_solution_l555_55565

theorem complex_equation_solution (z : ℂ) :
  z * (1 - Complex.I) = 2 * Complex.I → z = -1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l555_55565


namespace NUMINAMATH_CALUDE_number_manipulation_l555_55504

theorem number_manipulation (x : ℝ) : (x - 5) / 7 = 7 → (x - 14) / 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_number_manipulation_l555_55504


namespace NUMINAMATH_CALUDE_corrected_mean_l555_55520

def number_of_observations : ℕ := 100
def original_mean : ℝ := 125.6
def incorrect_observation1 : ℝ := 95.3
def incorrect_observation2 : ℝ := -15.9
def correct_observation1 : ℝ := 48.2
def correct_observation2 : ℝ := -35.7

theorem corrected_mean (n : ℕ) (om : ℝ) (io1 io2 co1 co2 : ℝ) :
  n = number_of_observations ∧
  om = original_mean ∧
  io1 = incorrect_observation1 ∧
  io2 = incorrect_observation2 ∧
  co1 = correct_observation1 ∧
  co2 = correct_observation2 →
  (n : ℝ) * om - (io1 + io2) + (co1 + co2) / n = 124.931 := by
  sorry

end NUMINAMATH_CALUDE_corrected_mean_l555_55520


namespace NUMINAMATH_CALUDE_light_bulb_replacement_l555_55523

def month_number (m : String) : Nat :=
  match m with
  | "January" => 1
  | "February" => 2
  | "March" => 3
  | "April" => 4
  | "May" => 5
  | "June" => 6
  | "July" => 7
  | "August" => 8
  | "September" => 9
  | "October" => 10
  | "November" => 11
  | "December" => 12
  | _ => 0

def cycle_length : Nat := 7
def start_month : String := "January"
def replacement_count : Nat := 12

theorem light_bulb_replacement :
  (cycle_length * (replacement_count - 1)) % 12 + month_number start_month = month_number "June" :=
by sorry

end NUMINAMATH_CALUDE_light_bulb_replacement_l555_55523


namespace NUMINAMATH_CALUDE_multiply_divide_example_l555_55501

theorem multiply_divide_example : (3.242 * 15) / 100 = 0.4863 := by
  sorry

end NUMINAMATH_CALUDE_multiply_divide_example_l555_55501


namespace NUMINAMATH_CALUDE_toothpick_15th_stage_l555_55595

def toothpick_sequence (n : ℕ) : ℕ :=
  5 + 3 * (n - 1)

theorem toothpick_15th_stage :
  toothpick_sequence 15 = 47 := by
  sorry

end NUMINAMATH_CALUDE_toothpick_15th_stage_l555_55595


namespace NUMINAMATH_CALUDE_sum_odd_integers_13_to_41_l555_55510

/-- The sum of odd integers from 13 to 41, inclusive -/
def sumOddIntegers : ℕ :=
  let first := 13
  let last := 41
  let n := (last - first) / 2 + 1
  n * (first + last) / 2

theorem sum_odd_integers_13_to_41 :
  sumOddIntegers = 405 := by
  sorry

end NUMINAMATH_CALUDE_sum_odd_integers_13_to_41_l555_55510


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l555_55532

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_property 
  (a : ℕ → ℝ) 
  (h_geometric : is_geometric_sequence a)
  (h_eq1 : a 2 * a 4 * a 5 = a 3 * a 6)
  (h_eq2 : a 9 * a 10 = -8) :
  a 7 = -2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l555_55532


namespace NUMINAMATH_CALUDE_no_divisible_seven_digit_numbers_l555_55539

/-- A function that checks if a number uses each of the digits 1-7 exactly once. -/
def usesDigits1To7Once (n : ℕ) : Prop :=
  ∃ (a b c d e f g : ℕ),
    n = a * 1000000 + b * 100000 + c * 10000 + d * 1000 + e * 100 + f * 10 + g ∧
    ({a, b, c, d, e, f, g} : Finset ℕ) = {1, 2, 3, 4, 5, 6, 7}

/-- Theorem stating that there are no two seven-digit numbers formed using
    digits 1-7 once each where one divides the other. -/
theorem no_divisible_seven_digit_numbers :
  ¬∃ (m n : ℕ), m ≠ n ∧ 
    usesDigits1To7Once m ∧ 
    usesDigits1To7Once n ∧ 
    m ∣ n :=
by sorry

end NUMINAMATH_CALUDE_no_divisible_seven_digit_numbers_l555_55539


namespace NUMINAMATH_CALUDE_marcia_wardrobe_cost_l555_55530

/-- Calculates the total cost of Marcia's wardrobe --/
def wardrobeCost (skirtPrice blousePrice pantPrice : ℚ) 
                 (numSkirts numBlouses numPants : ℕ) : ℚ :=
  let skirtCost := skirtPrice * numSkirts
  let blouseCost := blousePrice * numBlouses
  let pantCost := pantPrice * (numPants - 1) + (pantPrice / 2)
  skirtCost + blouseCost + pantCost

/-- Proves that the total cost of Marcia's wardrobe is $180.00 --/
theorem marcia_wardrobe_cost :
  wardrobeCost 20 15 30 3 5 2 = 180 := by
  sorry

end NUMINAMATH_CALUDE_marcia_wardrobe_cost_l555_55530


namespace NUMINAMATH_CALUDE_workshop_day_probability_l555_55546

/-- The probability of a student being absent on a normal day -/
def normal_absence_rate : ℚ := 1/20

/-- The probability of a student being absent on the workshop day -/
def workshop_absence_rate : ℚ := min (2 * normal_absence_rate) 1

/-- The probability of a student being present on the workshop day -/
def workshop_presence_rate : ℚ := 1 - workshop_absence_rate

/-- The probability of one student being absent and one being present on the workshop day -/
def one_absent_one_present : ℚ := 
  workshop_absence_rate * workshop_presence_rate * 2

theorem workshop_day_probability : one_absent_one_present = 18/100 := by
  sorry

end NUMINAMATH_CALUDE_workshop_day_probability_l555_55546


namespace NUMINAMATH_CALUDE_max_subway_commuters_l555_55553

theorem max_subway_commuters (total_employees : ℕ) 
  (h_total : total_employees = 48) 
  (part_time full_time : ℕ) 
  (h_sum : part_time + full_time = total_employees) 
  (h_both_exist : part_time > 0 ∧ full_time > 0) :
  ∃ (subway_commuters : ℕ), 
    subway_commuters = ⌊(1 / 3 : ℚ) * part_time⌋ + ⌊(1 / 4 : ℚ) * full_time⌋ ∧
    subway_commuters ≤ 15 ∧
    (∀ (pt ft : ℕ), 
      pt + ft = total_employees → 
      pt > 0 → 
      ft > 0 → 
      ⌊(1 / 3 : ℚ) * pt⌋ + ⌊(1 / 4 : ℚ) * ft⌋ ≤ subway_commuters) :=
by sorry

end NUMINAMATH_CALUDE_max_subway_commuters_l555_55553


namespace NUMINAMATH_CALUDE_sum_of_triangle_ops_equals_21_l555_55549

-- Define the triangle operation
def triangle_op (a b c : ℕ) : ℕ := a + b + c

-- Define the two triangles
def triangle1 : (ℕ × ℕ × ℕ) := (2, 4, 3)
def triangle2 : (ℕ × ℕ × ℕ) := (1, 6, 5)

-- Theorem statement
theorem sum_of_triangle_ops_equals_21 :
  triangle_op triangle1.1 triangle1.2.1 triangle1.2.2 +
  triangle_op triangle2.1 triangle2.2.1 triangle2.2.2 = 21 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_triangle_ops_equals_21_l555_55549


namespace NUMINAMATH_CALUDE_wire_length_ratio_l555_55519

/-- The ratio of wire lengths in cube frame construction -/
theorem wire_length_ratio (bonnie_wire_pieces : ℕ) (bonnie_wire_length : ℝ) 
  (roark_wire_length : ℝ) : 
  bonnie_wire_pieces = 12 →
  bonnie_wire_length = 8 →
  roark_wire_length = 0.5 →
  (bonnie_wire_length ^ 3) * (roark_wire_length ^ 3)⁻¹ * 
    (12 * roark_wire_length) * (bonnie_wire_pieces * bonnie_wire_length)⁻¹ = 256 →
  (bonnie_wire_pieces * bonnie_wire_length) * 
    ((bonnie_wire_length ^ 3) * (roark_wire_length ^ 3)⁻¹ * (12 * roark_wire_length))⁻¹ = 1 / 256 := by
  sorry

end NUMINAMATH_CALUDE_wire_length_ratio_l555_55519


namespace NUMINAMATH_CALUDE_hyperbola_point_k_l555_55563

/-- Given a point P(-3, 1) on the hyperbola y = k/x where k ≠ 0, prove that k = -3 -/
theorem hyperbola_point_k (k : ℝ) (h1 : k ≠ 0) (h2 : (1 : ℝ) = k / (-3)) : k = -3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_point_k_l555_55563


namespace NUMINAMATH_CALUDE_pencil_distribution_solution_l555_55515

/-- Represents the pencil distribution problem --/
def PencilDistribution (initial_pencils : ℕ) (initial_containers : ℕ) (first_addition : ℕ) (second_addition : ℕ) (final_containers : ℕ) : Prop :=
  let total_pencils := initial_pencils + first_addition + second_addition
  ∃ (distributed_pencils : ℕ), 
    distributed_pencils ≤ total_pencils ∧
    distributed_pencils % final_containers = 0 ∧
    ∀ (n : ℕ), n > distributed_pencils → n % final_containers ≠ 0 ∨ n > total_pencils

/-- Theorem stating the solution to the pencil distribution problem --/
theorem pencil_distribution_solution :
  PencilDistribution 150 5 30 47 6 → 
  ∃ (distributed_pencils : ℕ), distributed_pencils = 222 ∧ distributed_pencils % 6 = 0 :=
sorry

end NUMINAMATH_CALUDE_pencil_distribution_solution_l555_55515


namespace NUMINAMATH_CALUDE_common_divisors_9240_10800_l555_55528

theorem common_divisors_9240_10800 : Nat.card {d : ℕ | d ∣ 9240 ∧ d ∣ 10800} = 16 := by
  sorry

end NUMINAMATH_CALUDE_common_divisors_9240_10800_l555_55528
