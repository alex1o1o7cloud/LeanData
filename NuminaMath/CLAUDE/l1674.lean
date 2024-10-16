import Mathlib

namespace NUMINAMATH_CALUDE_unoccupied_seats_l1674_167495

theorem unoccupied_seats (seats_per_row : ℕ) (num_rows : ℕ) (occupancy_ratio : ℚ) : 
  seats_per_row = 8 →
  num_rows = 12 →
  occupancy_ratio = 3/4 →
  seats_per_row * num_rows - (seats_per_row * num_rows * occupancy_ratio).floor = 24 := by
  sorry

end NUMINAMATH_CALUDE_unoccupied_seats_l1674_167495


namespace NUMINAMATH_CALUDE_speed_equivalence_l1674_167472

/-- Proves that a speed of 0.8 km/h is equivalent to 8/36 m/s -/
theorem speed_equivalence : ∃ (speed : ℚ), 
  (speed = 8 / 36) ∧ 
  (speed * 3600 / 1000 = 0.8) := by
  sorry

end NUMINAMATH_CALUDE_speed_equivalence_l1674_167472


namespace NUMINAMATH_CALUDE_projected_revenue_increase_l1674_167449

theorem projected_revenue_increase (actual_decrease : Real) (actual_to_projected_ratio : Real) 
  (h1 : actual_decrease = 0.3)
  (h2 : actual_to_projected_ratio = 0.5) :
  ∃ (projected_increase : Real), 
    (1 - actual_decrease) = actual_to_projected_ratio * (1 + projected_increase) ∧ 
    projected_increase = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_projected_revenue_increase_l1674_167449


namespace NUMINAMATH_CALUDE_lighting_power_increase_l1674_167401

/-- Proves that the increase in lighting power is 60 BT given the initial and final power values. -/
theorem lighting_power_increase (N_before N_after : ℝ) 
  (h1 : N_before = 240)
  (h2 : N_after = 300) :
  N_after - N_before = 60 := by
  sorry

end NUMINAMATH_CALUDE_lighting_power_increase_l1674_167401


namespace NUMINAMATH_CALUDE_max_value_xy_difference_l1674_167448

theorem max_value_xy_difference (x y : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) :
  x^2 * y - y^2 * x ≤ 1/4 := by
  sorry

end NUMINAMATH_CALUDE_max_value_xy_difference_l1674_167448


namespace NUMINAMATH_CALUDE_roots_reciprocal_sum_l1674_167465

theorem roots_reciprocal_sum (x₁ x₂ : ℝ) : 
  (5 * x₁^2 - 3 * x₁ - 2 = 0) → 
  (5 * x₂^2 - 3 * x₂ - 2 = 0) → 
  (x₁ ≠ x₂) →
  (1 / x₁ + 1 / x₂ = -3/2) := by
  sorry

end NUMINAMATH_CALUDE_roots_reciprocal_sum_l1674_167465


namespace NUMINAMATH_CALUDE_jess_walks_to_store_l1674_167450

/-- The number of blocks Jess walks to the store -/
def blocks_to_store : ℕ := sorry

/-- The total number of blocks Jess walks -/
def total_blocks : ℕ := 25

/-- Theorem stating that Jess walks 11 blocks to the store -/
theorem jess_walks_to_store : 
  blocks_to_store = 11 :=
by
  have h1 : blocks_to_store + 6 + 8 = total_blocks := sorry
  sorry


end NUMINAMATH_CALUDE_jess_walks_to_store_l1674_167450


namespace NUMINAMATH_CALUDE_prob_3a_minus_1_gt_0_prob_3a_minus_1_gt_0_is_two_thirds_l1674_167412

/-- The probability that 3a - 1 > 0, where a is a uniform random number between 0 and 1 -/
theorem prob_3a_minus_1_gt_0 : ℝ :=
  2/3

/-- Proof that the probability is indeed 2/3 -/
theorem prob_3a_minus_1_gt_0_is_two_thirds :
  prob_3a_minus_1_gt_0 = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_prob_3a_minus_1_gt_0_prob_3a_minus_1_gt_0_is_two_thirds_l1674_167412


namespace NUMINAMATH_CALUDE_fifth_term_of_arithmetic_sequence_l1674_167463

def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ := a + d * (n - 1)

theorem fifth_term_of_arithmetic_sequence
  (a d : ℝ)
  (h1 : arithmetic_sequence a d 2 + arithmetic_sequence a d 4 = 10)
  (h2 : arithmetic_sequence a d 1 + arithmetic_sequence a d 3 = 8) :
  arithmetic_sequence a d 5 = 7 := by
sorry

end NUMINAMATH_CALUDE_fifth_term_of_arithmetic_sequence_l1674_167463


namespace NUMINAMATH_CALUDE_problem_solution_l1674_167423

def f (x : ℝ) := abs (2*x - 1) - abs (2*x - 2)

theorem problem_solution :
  (∃ k : ℝ, ∀ x : ℝ, f x ≤ k) ∧
  ({x : ℝ | f x ≥ x} = {x : ℝ | x ≤ -1 ∨ x = 1}) ∧
  (∀ x : ℝ, f x ≤ 1) ∧
  (¬∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a + 2*b = 1 ∧ 2/a + 1/b = 4 - 1/(a*b)) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1674_167423


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l1674_167481

theorem arithmetic_calculation : 3^2 * 4 + 5 * (6 + 3) - 15 / 3 = 76 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l1674_167481


namespace NUMINAMATH_CALUDE_shopping_money_l1674_167456

/-- Proves that if a person spends $20 and is left with $4 more than half of their original amount, then their original amount was $48. -/
theorem shopping_money (original : ℕ) : 
  (original / 2 + 4 = original - 20) → original = 48 :=
by sorry

end NUMINAMATH_CALUDE_shopping_money_l1674_167456


namespace NUMINAMATH_CALUDE_budget_utilities_percentage_l1674_167410

theorem budget_utilities_percentage (transportation : ℝ) (research_development : ℝ) 
  (equipment : ℝ) (supplies : ℝ) (salaries_degrees : ℝ) :
  transportation = 15 →
  research_development = 9 →
  equipment = 4 →
  supplies = 2 →
  salaries_degrees = 234 →
  (salaries_degrees / 360) * 100 + transportation + research_development + equipment + supplies + 
    (100 - ((salaries_degrees / 360) * 100 + transportation + research_development + equipment + supplies)) = 100 →
  100 - ((salaries_degrees / 360) * 100 + transportation + research_development + equipment + supplies) = 5 := by
sorry

end NUMINAMATH_CALUDE_budget_utilities_percentage_l1674_167410


namespace NUMINAMATH_CALUDE_average_of_numbers_l1674_167435

def numbers : List ℝ := [13, 14, 510, 520, 530, 1115, 1120, 1, 1252140, 2345]

theorem average_of_numbers : 
  (numbers.sum / numbers.length : ℝ) = 125830.8 := by
  sorry

end NUMINAMATH_CALUDE_average_of_numbers_l1674_167435


namespace NUMINAMATH_CALUDE_abc_magnitude_order_l1674_167464

/-- Given the definitions of a, b, and c, prove that b > c > a -/
theorem abc_magnitude_order :
  let a := (1/2) * Real.cos (16 * π / 180) - (Real.sqrt 3 / 2) * Real.sin (16 * π / 180)
  let b := (2 * Real.tan (14 * π / 180)) / (1 + Real.tan (14 * π / 180) ^ 2)
  let c := Real.sqrt ((1 - Real.cos (50 * π / 180)) / 2)
  b > c ∧ c > a := by sorry

end NUMINAMATH_CALUDE_abc_magnitude_order_l1674_167464


namespace NUMINAMATH_CALUDE_tangent_circles_ratio_l1674_167441

/-- Two circles are tangent if the distance between their centers is equal to the sum of their radii -/
def are_tangent (center1 center2 : ℝ × ℝ) (radius : ℝ) : Prop :=
  (center1.1 - center2.1)^2 + (center1.2 - center2.2)^2 = (2 * radius)^2

/-- Definition of circle C₁ -/
def circle_C1 (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = a^2}

/-- Definition of circle C₂ -/
def circle_C2 (a b c : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - b)^2 + (p.2 - c)^2 = a^2}

theorem tangent_circles_ratio (a b c : ℝ) (ha : a > 0) :
  are_tangent (0, 0) (b, c) a →
  (b^2 + c^2) / a^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_tangent_circles_ratio_l1674_167441


namespace NUMINAMATH_CALUDE_parabola_axis_of_symmetry_l1674_167445

/-- The equation of a parabola -/
def parabola_equation (x y : ℝ) : Prop := y = -2 * x^2

/-- The equation of the axis of symmetry -/
def axis_of_symmetry (y : ℝ) : Prop := y = 1/8

/-- Theorem: The axis of symmetry for the parabola y = -2x^2 is y = 1/8 -/
theorem parabola_axis_of_symmetry :
  ∀ x y : ℝ, parabola_equation x y → axis_of_symmetry y := by
  sorry

end NUMINAMATH_CALUDE_parabola_axis_of_symmetry_l1674_167445


namespace NUMINAMATH_CALUDE_dabbie_turkey_cost_l1674_167498

/-- The cost of Dabbie's turkeys -/
def turkey_cost : ℕ → ℕ
| 0 => 6  -- weight of first turkey
| 1 => 9  -- weight of second turkey
| 2 => 2 * turkey_cost 1  -- weight of third turkey
| _ => 0  -- for completeness

/-- The total weight of all turkeys -/
def total_weight : ℕ := turkey_cost 0 + turkey_cost 1 + turkey_cost 2

/-- The cost per kilogram of turkey -/
def cost_per_kg : ℕ := 2

/-- The theorem stating the total cost of Dabbie's turkeys -/
theorem dabbie_turkey_cost : total_weight * cost_per_kg = 66 := by
  sorry

end NUMINAMATH_CALUDE_dabbie_turkey_cost_l1674_167498


namespace NUMINAMATH_CALUDE_inverse_function_range_l1674_167482

-- Define the function
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + a

-- State the theorem
theorem inverse_function_range (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 3, Function.Injective (fun x => f a x)) ∧ 
  (|a - 1| + |a - 3| ≤ 4) →
  a ∈ Set.Icc 0 1 ∪ Set.Icc 3 4 :=
sorry

end NUMINAMATH_CALUDE_inverse_function_range_l1674_167482


namespace NUMINAMATH_CALUDE_second_to_last_digit_is_five_l1674_167446

def is_power_of_prime (n : ℕ) : Prop :=
  ∃ p k, Prime p ∧ n = p ^ k

theorem second_to_last_digit_is_five (N : ℕ) 
  (h1 : N % 10 = 0) 
  (h2 : ∃ d : ℕ, d < N ∧ d ∣ N ∧ is_power_of_prime d ∧ ∀ m : ℕ, m < N → m ∣ N → m ≤ d)
  (h3 : N > 10) :
  (N / 10) % 10 = 5 :=
sorry

end NUMINAMATH_CALUDE_second_to_last_digit_is_five_l1674_167446


namespace NUMINAMATH_CALUDE_vacation_cost_balance_l1674_167431

/-- Proves that the difference between what Tom and Dorothy owe Sammy is -50 --/
theorem vacation_cost_balance (tom_paid dorothy_paid sammy_paid t d : ℚ) : 
  tom_paid = 140 →
  dorothy_paid = 90 →
  sammy_paid = 220 →
  (tom_paid + t) = (dorothy_paid + d) →
  (tom_paid + t) = (sammy_paid - t - d) →
  t - d = -50 := by
sorry

end NUMINAMATH_CALUDE_vacation_cost_balance_l1674_167431


namespace NUMINAMATH_CALUDE_paint_combinations_l1674_167454

theorem paint_combinations (num_colors num_methods : ℕ) :
  num_colors = 5 → num_methods = 4 → num_colors * num_methods = 20 := by
  sorry

end NUMINAMATH_CALUDE_paint_combinations_l1674_167454


namespace NUMINAMATH_CALUDE_inequalities_proof_l1674_167462

theorem inequalities_proof (k : ℕ) (x : Fin k → ℝ) 
  (h_pos : ∀ i, x i > 0) (h_diff : ∀ i j, i ≠ j → x i ≠ x j) : 
  (Real.sqrt ((Finset.univ.sum (λ i => (x i)^2)) / k) > 
   (Finset.univ.sum (λ i => x i)) / k) ∧
  ((Finset.univ.sum (λ i => x i)) / k > 
   k / (Finset.univ.sum (λ i => 1 / (x i)))) := by
  sorry


end NUMINAMATH_CALUDE_inequalities_proof_l1674_167462


namespace NUMINAMATH_CALUDE_derivative_of_power_function_l1674_167407

open Real

/-- Given differentiable functions u and v, where u is positive,
    f(x) = u(x)^(v(x)) is differentiable and its derivative is as stated. -/
theorem derivative_of_power_function (u v : ℝ → ℝ) (hu : Differentiable ℝ u)
    (hv : Differentiable ℝ v) (hup : ∀ x, u x > 0) :
  let f := λ x => (u x) ^ (v x)
  Differentiable ℝ f ∧ 
  ∀ x, deriv f x = (u x)^(v x) * (deriv v x * log (u x) + v x * deriv u x / u x) :=
by sorry

end NUMINAMATH_CALUDE_derivative_of_power_function_l1674_167407


namespace NUMINAMATH_CALUDE_modulus_of_complex_expression_l1674_167480

theorem modulus_of_complex_expression :
  let z : ℂ := (1 : ℂ) / (1 + Complex.I) + Complex.I
  Complex.abs z = Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_modulus_of_complex_expression_l1674_167480


namespace NUMINAMATH_CALUDE_litter_patrol_collection_l1674_167432

/-- The number of glass bottles picked up by the Litter Patrol -/
def glass_bottles : ℕ := 10

/-- The number of aluminum cans picked up by the Litter Patrol -/
def aluminum_cans : ℕ := 8

/-- The total number of pieces of litter is the sum of glass bottles and aluminum cans -/
def total_litter : ℕ := glass_bottles + aluminum_cans

/-- Theorem stating that the total number of pieces of litter is 18 -/
theorem litter_patrol_collection : total_litter = 18 := by
  sorry

end NUMINAMATH_CALUDE_litter_patrol_collection_l1674_167432


namespace NUMINAMATH_CALUDE_at_least_one_real_root_l1674_167467

theorem at_least_one_real_root (a b c : ℝ) : 
  (a - b)^2 - 4*(b - c) ≥ 0 ∨ 
  (b - c)^2 - 4*(c - a) ≥ 0 ∨ 
  (c - a)^2 - 4*(a - b) ≥ 0 := by
sorry

end NUMINAMATH_CALUDE_at_least_one_real_root_l1674_167467


namespace NUMINAMATH_CALUDE_exam_maximum_marks_l1674_167437

theorem exam_maximum_marks (ashley_marks : ℕ) (ashley_percentage : ℚ) :
  ashley_marks = 332 →
  ashley_percentage = 83 / 100 →
  (ashley_marks : ℚ) / ashley_percentage = 400 :=
by
  sorry

end NUMINAMATH_CALUDE_exam_maximum_marks_l1674_167437


namespace NUMINAMATH_CALUDE_exterior_angle_measure_l1674_167428

theorem exterior_angle_measure (a b : ℝ) (ha : a = 40) (hb : b = 30) : 
  180 - (180 - a - b) = 70 := by sorry

end NUMINAMATH_CALUDE_exterior_angle_measure_l1674_167428


namespace NUMINAMATH_CALUDE_fifth_term_is_16_l1674_167489

/-- A geometric sequence with first term 1 and common ratio 2 -/
def geometric_sequence (n : ℕ) : ℝ := 1 * 2^(n - 1)

/-- The fifth term of the geometric sequence is 16 -/
theorem fifth_term_is_16 : geometric_sequence 5 = 16 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_is_16_l1674_167489


namespace NUMINAMATH_CALUDE_negation_inequality_statement_l1674_167490

theorem negation_inequality_statement :
  ¬(∀ (x : ℝ), x^2 + 1 > 0) ≠ (∃ (x : ℝ), x^2 + 1 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_inequality_statement_l1674_167490


namespace NUMINAMATH_CALUDE_zinc_copper_mixture_weight_l1674_167427

/-- Proves that the total weight of a zinc-copper mixture is 70 kg,
    given a 9:11 ratio and 31.5 kg of zinc used. -/
theorem zinc_copper_mixture_weight
  (zinc_ratio : ℝ)
  (copper_ratio : ℝ)
  (zinc_weight : ℝ)
  (h_ratio : zinc_ratio / copper_ratio = 9 / 11)
  (h_zinc : zinc_weight = 31.5) :
  zinc_weight + (copper_ratio / zinc_ratio) * zinc_weight = 70 :=
by sorry

end NUMINAMATH_CALUDE_zinc_copper_mixture_weight_l1674_167427


namespace NUMINAMATH_CALUDE_range_of_a_l1674_167451

def A (a : ℝ) : Set ℝ := {x | (a * x - 1) * (a - x) > 0}

theorem range_of_a :
  ∀ a : ℝ, (2 ∈ A a ∧ 3 ∉ A a) ↔ a ∈ (Set.Ioo 2 3 ∪ Set.Ico (1/3) (1/2)) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1674_167451


namespace NUMINAMATH_CALUDE_cucumber_weight_after_evaporation_l1674_167458

theorem cucumber_weight_after_evaporation 
  (initial_weight : ℝ) 
  (initial_water_percentage : ℝ) 
  (final_water_percentage : ℝ) :
  initial_weight = 100 →
  initial_water_percentage = 0.99 →
  final_water_percentage = 0.95 →
  ∃ (final_weight : ℝ), 
    final_weight * (1 - final_water_percentage) = initial_weight * (1 - initial_water_percentage) ∧
    final_weight = 20 :=
by sorry

end NUMINAMATH_CALUDE_cucumber_weight_after_evaporation_l1674_167458


namespace NUMINAMATH_CALUDE_photo_arrangements_l1674_167488

def number_of_students : ℕ := 5

def arrangements (n : ℕ) : ℕ := sorry

theorem photo_arrangements :
  arrangements number_of_students = 36 := by sorry

end NUMINAMATH_CALUDE_photo_arrangements_l1674_167488


namespace NUMINAMATH_CALUDE_intersection_point_satisfies_equations_intersection_point_unique_l1674_167447

/-- The system of linear equations representing two lines -/
def line1 (x y : ℚ) : Prop := 12 * x - 5 * y = 40
def line2 (x y : ℚ) : Prop := 8 * x + 2 * y = 20

/-- The intersection point of the two lines -/
def intersection_point : ℚ × ℚ := (45/16, -5/4)

/-- Theorem stating that the intersection point satisfies both equations -/
theorem intersection_point_satisfies_equations :
  let (x, y) := intersection_point
  line1 x y ∧ line2 x y :=
sorry

/-- Theorem stating that the intersection point is unique -/
theorem intersection_point_unique (x y : ℚ) :
  line1 x y → line2 x y → (x, y) = intersection_point :=
sorry

end NUMINAMATH_CALUDE_intersection_point_satisfies_equations_intersection_point_unique_l1674_167447


namespace NUMINAMATH_CALUDE_lucia_outfits_l1674_167416

/-- Represents the number of different outfits Lucia can create -/
def outfits (shoes dresses hats : ℕ) : ℕ := shoes * dresses * hats

/-- Proves that Lucia can create 60 different outfits -/
theorem lucia_outfits :
  outfits 3 5 4 = 60 := by
  sorry

end NUMINAMATH_CALUDE_lucia_outfits_l1674_167416


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l1674_167477

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + 2

def geometric_sequence (a b c : ℤ) : Prop :=
  b * b = a * c

theorem arithmetic_geometric_sequence (a : ℕ → ℤ) :
  arithmetic_sequence a →
  geometric_sequence (a 1) (a 3) (a 4) →
  a 2 = -6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l1674_167477


namespace NUMINAMATH_CALUDE_no_solution_exists_l1674_167404

theorem no_solution_exists (w x y z : ℂ) (n : ℕ+) : 
  w ≠ 0 → x ≠ 0 → y ≠ 0 → z ≠ 0 →
  1 / w + 1 / x + 1 / y + 1 / z = 3 →
  w * x + w * y + w * z + x * y + x * z + y * z = 14 →
  (w + x)^3 + (w + y)^3 + (w + z)^3 + (x + y)^3 + (x + z)^3 + (y + z)^3 = 2160 →
  ∃ (r : ℝ), w + x + y + z + Complex.I * Real.sqrt n = r →
  False :=
by sorry

end NUMINAMATH_CALUDE_no_solution_exists_l1674_167404


namespace NUMINAMATH_CALUDE_chord_intersection_l1674_167409

/-- Given a circle and a line intersecting it, prove the value of the line's slope -/
theorem chord_intersection (x y : ℝ) (a : ℝ) : 
  (x^2 + y^2 - 2*x - 8*y + 13 = 0) →  -- Circle equation
  (a*x + y - 1 = 0) →                -- Line equation
  (∃ (x1 y1 x2 y2 : ℝ), 
    (x1^2 + y1^2 - 2*x1 - 8*y1 + 13 = 0) ∧ 
    (x2^2 + y2^2 - 2*x2 - 8*y2 + 13 = 0) ∧ 
    (a*x1 + y1 - 1 = 0) ∧ 
    (a*x2 + y2 - 1 = 0) ∧ 
    ((x1 - x2)^2 + (y1 - y2)^2 = 12)) →  -- Chord length condition
  (a = -4/3) :=
by sorry

end NUMINAMATH_CALUDE_chord_intersection_l1674_167409


namespace NUMINAMATH_CALUDE_cube_sum_of_roots_l1674_167485

theorem cube_sum_of_roots (r s t : ℝ) : 
  (r - (20 : ℝ)^(1/3)) * (r - (60 : ℝ)^(1/3)) * (r - (120 : ℝ)^(1/3)) = 1 →
  (s - (20 : ℝ)^(1/3)) * (s - (60 : ℝ)^(1/3)) * (s - (120 : ℝ)^(1/3)) = 1 →
  (t - (20 : ℝ)^(1/3)) * (t - (60 : ℝ)^(1/3)) * (t - (120 : ℝ)^(1/3)) = 1 →
  r ≠ s → r ≠ t → s ≠ t →
  r^3 + s^3 + t^3 = 203 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_of_roots_l1674_167485


namespace NUMINAMATH_CALUDE_sum_of_squares_l1674_167455

theorem sum_of_squares (x y : ℚ) (h1 : x + 2*y = 20) (h2 : 3*x + y = 19) : x^2 + y^2 = 401/5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l1674_167455


namespace NUMINAMATH_CALUDE_steve_pie_difference_l1674_167474

/-- The number of days Steve bakes apple pies in a week -/
def apple_pie_days : ℕ := 3

/-- The number of days Steve bakes cherry pies in a week -/
def cherry_pie_days : ℕ := 2

/-- The number of pies Steve bakes per day -/
def pies_per_day : ℕ := 12

/-- The number of apple pies Steve bakes in a week -/
def apple_pies_per_week : ℕ := apple_pie_days * pies_per_day

/-- The number of cherry pies Steve bakes in a week -/
def cherry_pies_per_week : ℕ := cherry_pie_days * pies_per_day

theorem steve_pie_difference : apple_pies_per_week - cherry_pies_per_week = 12 := by
  sorry

end NUMINAMATH_CALUDE_steve_pie_difference_l1674_167474


namespace NUMINAMATH_CALUDE_father_daughter_age_sum_l1674_167461

theorem father_daughter_age_sum :
  ∀ (father_age daughter_age : ℕ),
    father_age - daughter_age = 22 →
    daughter_age = 16 →
    father_age + daughter_age = 54 :=
by
  sorry

end NUMINAMATH_CALUDE_father_daughter_age_sum_l1674_167461


namespace NUMINAMATH_CALUDE_eight_thousand_eight_place_values_l1674_167403

/-- Represents the place value of a digit in a number -/
inductive PlaceValue
  | Ones
  | Tens
  | Hundreds
  | Thousands

/-- Returns the place value of a digit based on its position from the right -/
def getPlaceValue (position : Nat) : PlaceValue :=
  match position with
  | 1 => PlaceValue.Ones
  | 2 => PlaceValue.Tens
  | 3 => PlaceValue.Hundreds
  | 4 => PlaceValue.Thousands
  | _ => PlaceValue.Ones  -- Default to Ones for other positions

/-- Represents a digit in a specific position of a number -/
structure Digit where
  value : Nat
  position : Nat

/-- Theorem: In the number 8008, the 8 in the first position from the right represents 8 units of ones,
    and the 8 in the fourth position from the right represents 8 units of thousands -/
theorem eight_thousand_eight_place_values :
  let num := 8008
  let rightmost_eight : Digit := { value := 8, position := 1 }
  let leftmost_eight : Digit := { value := 8, position := 4 }
  (getPlaceValue rightmost_eight.position = PlaceValue.Ones) ∧
  (getPlaceValue leftmost_eight.position = PlaceValue.Thousands) :=
by sorry

end NUMINAMATH_CALUDE_eight_thousand_eight_place_values_l1674_167403


namespace NUMINAMATH_CALUDE_angel_score_is_11_l1674_167493

-- Define the scores for each player
def beth_score : ℕ := 12
def jan_score : ℕ := 10
def judy_score : ℕ := 8

-- Define the total score of the first team
def first_team_score : ℕ := beth_score + jan_score

-- Define the difference between the first and second team scores
def score_difference : ℕ := 3

-- Define Angel's score as a variable
def angel_score : ℕ := sorry

-- Theorem to prove
theorem angel_score_is_11 :
  angel_score = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_angel_score_is_11_l1674_167493


namespace NUMINAMATH_CALUDE_legos_lost_l1674_167418

theorem legos_lost (initial_legos current_legos : ℕ) 
  (h1 : initial_legos = 380) 
  (h2 : current_legos = 323) : 
  initial_legos - current_legos = 57 := by
  sorry

end NUMINAMATH_CALUDE_legos_lost_l1674_167418


namespace NUMINAMATH_CALUDE_encyclopedia_monthly_payment_l1674_167497

/-- Proves that the monthly payment for the encyclopedia purchase is $57 -/
theorem encyclopedia_monthly_payment
  (total_cost : ℝ)
  (down_payment : ℝ)
  (num_monthly_payments : ℕ)
  (final_payment : ℝ)
  (interest_rate : ℝ)
  (h_total_cost : total_cost = 750)
  (h_down_payment : down_payment = 300)
  (h_num_monthly_payments : num_monthly_payments = 9)
  (h_final_payment : final_payment = 21)
  (h_interest_rate : interest_rate = 0.18666666666666668)
  : ∃ (monthly_payment : ℝ),
    monthly_payment = 57 ∧
    total_cost - down_payment + (total_cost - down_payment) * interest_rate =
    monthly_payment * num_monthly_payments + final_payment := by
  sorry

end NUMINAMATH_CALUDE_encyclopedia_monthly_payment_l1674_167497


namespace NUMINAMATH_CALUDE_investment_problem_l1674_167487

/-- Proves that given the conditions of the investment problem, b's investment amount is 1000. -/
theorem investment_problem (a b c total_profit c_share : ℚ) : 
  a = 800 →
  c = 1200 →
  total_profit = 1000 →
  c_share = 400 →
  c_share / total_profit = c / (a + b + c) →
  b = 1000 := by
  sorry

end NUMINAMATH_CALUDE_investment_problem_l1674_167487


namespace NUMINAMATH_CALUDE_octal_to_decimal_l1674_167460

-- Define the octal number
def octal_number : ℕ := 724

-- Define the decimal number
def decimal_number : ℕ := 468

-- Theorem stating that the octal number 724 is equal to the decimal number 468
theorem octal_to_decimal :
  octal_number.digits 8 = [4, 2, 7] ∧ 
  decimal_number = 4 * 8^0 + 2 * 8^1 + 7 * 8^2 := by
  sorry

#check octal_to_decimal

end NUMINAMATH_CALUDE_octal_to_decimal_l1674_167460


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l1674_167469

-- Define the sets A and B
def A : Set ℝ := {x | |x - 1| > 2}
def B : Set ℝ := {x | x^2 - 6*x + 8 < 0}

-- Define the complement of A in the universal set ℝ
def C_UA : Set ℝ := (Set.univ : Set ℝ) \ A

-- State the theorem
theorem complement_A_intersect_B : (C_UA ∩ B) = Set.Ioc 2 3 := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l1674_167469


namespace NUMINAMATH_CALUDE_sqrt_eighteen_div_sqrt_two_equals_three_l1674_167420

theorem sqrt_eighteen_div_sqrt_two_equals_three : 
  Real.sqrt 18 / Real.sqrt 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_eighteen_div_sqrt_two_equals_three_l1674_167420


namespace NUMINAMATH_CALUDE_regular_hexagon_most_symmetry_l1674_167402

-- Define a type for shapes
inductive Shape
  | RegularPentagon
  | Kite
  | RegularHexagon
  | IsoscelesTriangle
  | ScaleneTriangle

-- Define a function to count lines of symmetry
def linesOfSymmetry (s : Shape) : ℕ :=
  match s with
  | Shape.RegularPentagon => 5
  | Shape.Kite => 1
  | Shape.RegularHexagon => 6
  | Shape.IsoscelesTriangle => 1
  | Shape.ScaleneTriangle => 0

-- Theorem statement
theorem regular_hexagon_most_symmetry :
  ∀ s : Shape, s ≠ Shape.RegularHexagon →
  linesOfSymmetry Shape.RegularHexagon > linesOfSymmetry s :=
by sorry

end NUMINAMATH_CALUDE_regular_hexagon_most_symmetry_l1674_167402


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1674_167443

/-- The function f(x) = x^2 - 2x + 6 -/
def f (x : ℝ) : ℝ := x^2 - 2*x + 6

/-- Theorem stating that f(m+3) > f(2m) is equivalent to -1/3 < m < 3 -/
theorem inequality_equivalence (m : ℝ) : 
  f (m + 3) > f (2 * m) ↔ -1/3 < m ∧ m < 3 := by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1674_167443


namespace NUMINAMATH_CALUDE_smallest_angle_is_76_l1674_167417

/-- A pentagon with angles in arithmetic sequence -/
structure ArithmeticPentagon where
  -- The common difference between consecutive angles
  d : ℝ
  -- The smallest angle
  a : ℝ
  -- The sum of all angles is 540°
  sum_constraint : a + (a + d) + (a + 2*d) + (a + 3*d) + (a + 4*d) = 540
  -- The largest angle is 140°
  max_angle : a + 4*d = 140

/-- 
If a pentagon has angles in arithmetic sequence and its largest angle is 140°,
then its smallest angle is 76°.
-/
theorem smallest_angle_is_76 (p : ArithmeticPentagon) : p.a = 76 := by
  sorry

end NUMINAMATH_CALUDE_smallest_angle_is_76_l1674_167417


namespace NUMINAMATH_CALUDE_bread_inventory_l1674_167486

def initial_loaves : ℕ := 2355
def sold_loaves : ℕ := 629
def delivered_loaves : ℕ := 489

theorem bread_inventory : 
  initial_loaves - sold_loaves + delivered_loaves = 2215 := by
  sorry

end NUMINAMATH_CALUDE_bread_inventory_l1674_167486


namespace NUMINAMATH_CALUDE_marie_erasers_l1674_167471

def initial_erasers : ℕ := 95
def lost_erasers : ℕ := 42

theorem marie_erasers : initial_erasers - lost_erasers = 53 := by
  sorry

end NUMINAMATH_CALUDE_marie_erasers_l1674_167471


namespace NUMINAMATH_CALUDE_express_y_in_terms_of_x_l1674_167476

theorem express_y_in_terms_of_x (x y : ℝ) (h : y - 3 * x = 5) : y = 3 * x + 5 := by
  sorry

end NUMINAMATH_CALUDE_express_y_in_terms_of_x_l1674_167476


namespace NUMINAMATH_CALUDE_somus_age_l1674_167496

theorem somus_age (s f : ℕ) (h1 : s = f / 3) (h2 : s - 9 = (f - 9) / 5) : s = 18 := by
  sorry

end NUMINAMATH_CALUDE_somus_age_l1674_167496


namespace NUMINAMATH_CALUDE_min_sum_absolute_values_l1674_167470

theorem min_sum_absolute_values :
  (∀ x : ℝ, |x - 3| + |x - 1| + |x + 6| ≥ 9) ∧
  (∃ x : ℝ, |x - 3| + |x - 1| + |x + 6| = 9) := by
  sorry

end NUMINAMATH_CALUDE_min_sum_absolute_values_l1674_167470


namespace NUMINAMATH_CALUDE_factor_expression_l1674_167436

theorem factor_expression (y : ℝ) : 3 * y * (y - 4) + 5 * (y - 4) = (3 * y + 5) * (y - 4) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l1674_167436


namespace NUMINAMATH_CALUDE_cost_of_dozen_pens_l1674_167415

/-- Given the cost of 3 pens and 5 pencils, and the cost ratio of pen to pencil,
    calculate the cost of one dozen pens -/
theorem cost_of_dozen_pens (total_cost : ℕ) (ratio_pen_pencil : ℕ) :
  total_cost = 260 →
  ratio_pen_pencil = 5 →
  ∃ (pen_cost : ℕ) (pencil_cost : ℕ),
    3 * pen_cost + 5 * pencil_cost = total_cost ∧
    pen_cost = ratio_pen_pencil * pencil_cost ∧
    12 * pen_cost = 780 :=
by sorry

end NUMINAMATH_CALUDE_cost_of_dozen_pens_l1674_167415


namespace NUMINAMATH_CALUDE_books_left_l1674_167405

def initial_books : ℕ := 75
def borrowed_books : ℕ := 18

theorem books_left : initial_books - borrowed_books = 57 := by
  sorry

end NUMINAMATH_CALUDE_books_left_l1674_167405


namespace NUMINAMATH_CALUDE_min_cubes_surface_area_52_l1674_167419

/-- The surface area of a rectangular prism -/
def surface_area (l w h : ℕ) : ℕ := 2 * (l * w + l * h + w * h)

/-- The volume of a rectangular prism -/
def volume (l w h : ℕ) : ℕ := l * w * h

/-- The minimum number of unit cubes needed to form a rectangular prism with surface area 52 -/
theorem min_cubes_surface_area_52 :
  (∃ l w h : ℕ, surface_area l w h = 52 ∧ 
    volume l w h = 16 ∧
    ∀ l' w' h' : ℕ, surface_area l' w' h' = 52 → volume l' w' h' ≥ 16) :=
by sorry

end NUMINAMATH_CALUDE_min_cubes_surface_area_52_l1674_167419


namespace NUMINAMATH_CALUDE_sine_equality_theorem_l1674_167499

theorem sine_equality_theorem (n : ℤ) :
  0 ≤ n ∧ n ≤ 180 →
  (Real.sin (n * π / 180) = Real.sin (192 * π / 180)) ↔ (n = 12 ∨ n = 168) := by
sorry

end NUMINAMATH_CALUDE_sine_equality_theorem_l1674_167499


namespace NUMINAMATH_CALUDE_apple_rate_is_70_l1674_167479

-- Define the given quantities
def apple_quantity : ℕ := 8
def mango_quantity : ℕ := 9
def mango_rate : ℕ := 45
def total_paid : ℕ := 965

-- Define the unknown apple rate
def apple_rate : ℕ := sorry

-- Theorem statement
theorem apple_rate_is_70 :
  apple_quantity * apple_rate + mango_quantity * mango_rate = total_paid →
  apple_rate = 70 := by
  sorry

end NUMINAMATH_CALUDE_apple_rate_is_70_l1674_167479


namespace NUMINAMATH_CALUDE_test_question_points_l1674_167439

theorem test_question_points (total_points total_questions two_point_questions : ℕ) 
  (h1 : total_points = 100)
  (h2 : total_questions = 40)
  (h3 : two_point_questions = 30) :
  (total_points - 2 * two_point_questions) / (total_questions - two_point_questions) = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_test_question_points_l1674_167439


namespace NUMINAMATH_CALUDE_is_projection_matrix_l1674_167413

def projection_matrix (P : Matrix (Fin 2) (Fin 2) ℚ) : Prop :=
  P * P = P

theorem is_projection_matrix : 
  let P : Matrix (Fin 2) (Fin 2) ℚ := !![2368/2401, 16/49; 33*2401/2240, 33/49]
  projection_matrix P := by sorry

end NUMINAMATH_CALUDE_is_projection_matrix_l1674_167413


namespace NUMINAMATH_CALUDE_first_month_sale_is_7435_l1674_167438

/-- Calculates the sale in the first month given the sales for months 2 to 6 and the average sale for 6 months -/
def first_month_sale (second_month : ℕ) (third_month : ℕ) (fourth_month : ℕ) (fifth_month : ℕ) (sixth_month : ℕ) (average_sale : ℕ) : ℕ :=
  6 * average_sale - (second_month + third_month + fourth_month + fifth_month + sixth_month)

/-- Theorem stating that the sale in the first month is 7435 given the specified conditions -/
theorem first_month_sale_is_7435 :
  first_month_sale 7927 7855 8230 7562 5991 7500 = 7435 := by
  sorry

end NUMINAMATH_CALUDE_first_month_sale_is_7435_l1674_167438


namespace NUMINAMATH_CALUDE_vector_angle_problem_l1674_167411

/-- The angle between two vectors a and c -/
def angle_between (a c : ℝ × ℝ) : ℝ := sorry

/-- The dot product of two vectors -/
def dot_product (u v : ℝ × ℝ) : ℝ := sorry

/-- The magnitude (length) of a vector -/
def magnitude (v : ℝ × ℝ) : ℝ := sorry

theorem vector_angle_problem (c : ℝ × ℝ) :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (-2, -4)
  magnitude c = Real.sqrt 5 →
  dot_product (a.1 + b.1, a.2 + b.2) c = 5 / 2 →
  angle_between a c = 2 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_vector_angle_problem_l1674_167411


namespace NUMINAMATH_CALUDE_total_students_is_thirteen_l1674_167442

/-- The number of students in a presentation order, where Eunjeong's position is known. -/
def total_students (students_before_eunjeong : ℕ) (eunjeong_position_from_last : ℕ) : ℕ :=
  students_before_eunjeong + 1 + (eunjeong_position_from_last - 1)

/-- Theorem stating that the total number of students is 13 given the problem conditions. -/
theorem total_students_is_thirteen :
  total_students 7 6 = 13 := by sorry

end NUMINAMATH_CALUDE_total_students_is_thirteen_l1674_167442


namespace NUMINAMATH_CALUDE_square_root_sum_implies_product_l1674_167492

theorem square_root_sum_implies_product (x : ℝ) :
  Real.sqrt (10 + x) + Real.sqrt (40 - x) = 10 →
  (10 + x) * (40 - x) = 625 := by
  sorry

end NUMINAMATH_CALUDE_square_root_sum_implies_product_l1674_167492


namespace NUMINAMATH_CALUDE_total_profit_calculation_l1674_167475

/-- Prove that the total profit is 60000 given the investment ratios and C's profit share -/
theorem total_profit_calculation (a b c : ℕ) (total_profit : ℕ) : 
  a * 2 = c * 3 →  -- A and C invested in ratio 3:2
  a = b * 3 →      -- A and B invested in ratio 3:1
  c * total_profit = 20000 * (a + b + c) →  -- C's profit share
  total_profit = 60000 := by
  sorry

#check total_profit_calculation

end NUMINAMATH_CALUDE_total_profit_calculation_l1674_167475


namespace NUMINAMATH_CALUDE_tony_remaining_money_l1674_167434

def initial_amount : ℕ := 20
def ticket_cost : ℕ := 8
def hotdog_cost : ℕ := 3

theorem tony_remaining_money :
  initial_amount - ticket_cost - hotdog_cost = 9 :=
by sorry

end NUMINAMATH_CALUDE_tony_remaining_money_l1674_167434


namespace NUMINAMATH_CALUDE_triangle_trig_max_value_l1674_167491

theorem triangle_trig_max_value (A C : ℝ) (h1 : 0 ≤ A ∧ A ≤ π) (h2 : 0 ≤ C ∧ C ≤ π) 
  (h3 : Real.sin A + Real.sin C = 3/2) :
  let t := 2 * Real.sin A * Real.sin C
  (∃ (x : ℝ), x = t * Real.sqrt ((9/4 - t) * (t - 1/4))) ∧
  (∀ (y : ℝ), y = t * Real.sqrt ((9/4 - t) * (t - 1/4)) → y ≤ 27 * Real.sqrt 7 / 64) ∧
  (∃ (z : ℝ), z = t * Real.sqrt ((9/4 - t) * (t - 1/4)) ∧ z = 27 * Real.sqrt 7 / 64) :=
by sorry

end NUMINAMATH_CALUDE_triangle_trig_max_value_l1674_167491


namespace NUMINAMATH_CALUDE_shoes_to_sandals_ratio_l1674_167457

def shoes_sold : ℕ := 72
def sandals_sold : ℕ := 40

theorem shoes_to_sandals_ratio :
  (shoes_sold / sandals_sold : ℚ) = 9 / 5 := by
  sorry

end NUMINAMATH_CALUDE_shoes_to_sandals_ratio_l1674_167457


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1674_167429

/-- Given a hyperbola and an ellipse with the same foci, where the eccentricity of the hyperbola is twice that of the ellipse, prove that the equation of the hyperbola is x²/4 - y²/3 = 1 -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1) →
  (∀ x y : ℝ, x^2/16 + y^2/9 = 1) →
  (∃ c : ℝ, c^2 = a^2 + b^2 ∧ c^2 = 16 - 9) →
  (∃ e_e e_h : ℝ, e_h = 2*e_e ∧ e_e = Real.sqrt 7 / 4 ∧ e_h = Real.sqrt 7 / a) →
  (∀ x y : ℝ, x^2/4 - y^2/3 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1674_167429


namespace NUMINAMATH_CALUDE_function_decreasing_interval_l1674_167466

/-- The function f(x) = xe^x + 1 is decreasing on the interval (-∞, -1) -/
theorem function_decreasing_interval (x : ℝ) : 
  x < -1 → (fun x => x * Real.exp x + 1) '' Set.Ioi x ⊆ Set.Iio ((x * Real.exp x + 1) : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_function_decreasing_interval_l1674_167466


namespace NUMINAMATH_CALUDE_trajectory_of_midpoint_l1674_167478

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 = 1

/-- Point P is on the hyperbola -/
def P_on_hyperbola (px py : ℝ) : Prop := hyperbola px py

/-- M is the midpoint of OP -/
def M_is_midpoint (mx my px py : ℝ) : Prop := mx = px / 2 ∧ my = py / 2

/-- The trajectory equation for point M -/
def trajectory (x y : ℝ) : Prop := x^2 - 4*y^2 = 1

theorem trajectory_of_midpoint (mx my px py : ℝ) :
  P_on_hyperbola px py → M_is_midpoint mx my px py → trajectory mx my :=
by sorry

end NUMINAMATH_CALUDE_trajectory_of_midpoint_l1674_167478


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_l1674_167408

theorem right_triangle_perimeter (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a * b / 2 = 150 →
  a = 30 →
  a^2 + b^2 = c^2 →
  a + b + c = 40 + 10 * Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_l1674_167408


namespace NUMINAMATH_CALUDE_reciprocal_of_x_l1674_167433

theorem reciprocal_of_x (x : ℝ) (h1 : x^3 - 2*x^2 = 0) (h2 : x ≠ 0) : 1/x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_x_l1674_167433


namespace NUMINAMATH_CALUDE_last_day_is_monday_l1674_167473

/-- 
Given a year with 365 days, if the 15th day falls on a Monday,
then the 365th day also falls on a Monday.
-/
theorem last_day_is_monday (year : ℕ) : 
  year % 7 = 1 → -- Assuming Monday is represented by 1
  (365 % 7 = year % 7) → -- The last day falls on the same day as the first
  (15 % 7 = 1) → -- The 15th day is a Monday
  (365 % 7 = 1) -- The 365th day is also a Monday
:= by sorry

end NUMINAMATH_CALUDE_last_day_is_monday_l1674_167473


namespace NUMINAMATH_CALUDE_geometric_to_arithmetic_sequence_l1674_167425

theorem geometric_to_arithmetic_sequence (a₁ a₂ a₃ a₄ q : ℝ) : 
  q > 0 ∧ q ≠ 1 ∧
  a₂ = a₁ * q ∧ a₃ = a₂ * q ∧ a₄ = a₃ * q ∧
  ((a₁ + a₃ = 2 * a₂) ∨ (a₁ + a₄ = 2 * a₃)) →
  q = ((-1 + Real.sqrt 5) / 2) ∨ q = ((1 + Real.sqrt 5) / 2) := by
sorry

end NUMINAMATH_CALUDE_geometric_to_arithmetic_sequence_l1674_167425


namespace NUMINAMATH_CALUDE_sphere_surface_volume_relation_l1674_167422

theorem sphere_surface_volume_relation :
  ∀ (r r' : ℝ) (A A' V V' : ℝ),
  (A = 4 * Real.pi * r^2) →
  (A' = 4 * A) →
  (V = (4/3) * Real.pi * r^3) →
  (V' = (4/3) * Real.pi * r'^3) →
  (A' = 4 * Real.pi * r'^2) →
  (V' = 8 * V) :=
by sorry

end NUMINAMATH_CALUDE_sphere_surface_volume_relation_l1674_167422


namespace NUMINAMATH_CALUDE_sum_of_extremal_x_values_l1674_167459

theorem sum_of_extremal_x_values (x y z : ℝ) 
  (sum_condition : x + y + z = 5)
  (square_sum_condition : x^2 + y^2 + z^2 = 11) :
  ∃ (m M : ℝ), 
    (∀ x', (∃ y' z', x' + y' + z' = 5 ∧ x'^2 + y'^2 + z'^2 = 11) → m ≤ x' ∧ x' ≤ M) ∧
    m + M = 10/3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_extremal_x_values_l1674_167459


namespace NUMINAMATH_CALUDE_leap_year_53_sundays_probability_l1674_167468

/-- The number of days in a leap year -/
def leap_year_days : ℕ := 366

/-- The number of possible combinations for the two extra days in a leap year -/
def extra_day_combinations : ℕ := 7

/-- The number of combinations that result in 53 Sundays -/
def favorable_combinations : ℕ := 2

/-- The probability of a leap year having 53 Sundays -/
def prob_53_sundays : ℚ := favorable_combinations / extra_day_combinations

theorem leap_year_53_sundays_probability :
  prob_53_sundays = 2 / 7 := by sorry

end NUMINAMATH_CALUDE_leap_year_53_sundays_probability_l1674_167468


namespace NUMINAMATH_CALUDE_theater_ticket_sales_l1674_167421

theorem theater_ticket_sales (total_tickets : ℕ) (adult_price senior_price : ℚ) (total_receipts : ℚ) :
  total_tickets = 510 →
  adult_price = 21 →
  senior_price = 15 →
  total_receipts = 8748 →
  ∃ (adult_tickets senior_tickets : ℕ),
    adult_tickets + senior_tickets = total_tickets ∧
    adult_price * adult_tickets + senior_price * senior_tickets = total_receipts ∧
    senior_tickets = 327 :=
by sorry

end NUMINAMATH_CALUDE_theater_ticket_sales_l1674_167421


namespace NUMINAMATH_CALUDE_solve_sqrt_equation_l1674_167430

theorem solve_sqrt_equation (x : ℝ) :
  Real.sqrt ((3 / x) + 3) = 5/3 → x = -27/2 := by
  sorry

end NUMINAMATH_CALUDE_solve_sqrt_equation_l1674_167430


namespace NUMINAMATH_CALUDE_harmonious_triplet_from_intersections_l1674_167440

/-- Definition of a harmonious triplet -/
def is_harmonious_triplet (x y z : ℝ) : Prop :=
  x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧
  (1/x = 1/y + 1/z ∨ 1/y = 1/x + 1/z ∨ 1/z = 1/x + 1/y)

/-- Theorem about harmonious triplets formed by intersections -/
theorem harmonious_triplet_from_intersections
  (a b c : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : c ≠ 0) :
  let x₁ := -c / b
  let x₂ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₃ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  is_harmonious_triplet x₁ x₂ x₃ :=
by sorry

end NUMINAMATH_CALUDE_harmonious_triplet_from_intersections_l1674_167440


namespace NUMINAMATH_CALUDE_N_cannot_be_2_7_l1674_167424

def M : Set ℕ := {1, 4, 7}

theorem N_cannot_be_2_7 (N : Set ℕ) (h : M ∪ N = M) : N ≠ {2, 7} := by
  sorry

end NUMINAMATH_CALUDE_N_cannot_be_2_7_l1674_167424


namespace NUMINAMATH_CALUDE_equation_solution_l1674_167444

theorem equation_solution : ∃! x : ℝ, (x^2 + 3*x + 5) / (x^2 + 5*x + 6) = x + 3 ∧ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1674_167444


namespace NUMINAMATH_CALUDE_binomial_expansion_largest_coeff_l1674_167483

theorem binomial_expansion_largest_coeff (n : ℕ+) :
  (∀ k : ℕ, k ≠ 5 → Nat.choose n 5 ≥ Nat.choose n k) → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_largest_coeff_l1674_167483


namespace NUMINAMATH_CALUDE_josanna_minimum_score_l1674_167484

def josanna_scores : List ℕ := [82, 76, 91, 65, 87, 78]

def current_average : ℚ := (josanna_scores.sum : ℚ) / josanna_scores.length

def target_average : ℚ := current_average + 5

def minimum_score : ℕ := 116

theorem josanna_minimum_score :
  ∀ (new_score : ℕ),
    ((josanna_scores.sum + new_score : ℚ) / (josanna_scores.length + 1) ≥ target_average) →
    (new_score ≥ minimum_score) := by sorry

end NUMINAMATH_CALUDE_josanna_minimum_score_l1674_167484


namespace NUMINAMATH_CALUDE_system_equation_result_l1674_167426

theorem system_equation_result (x y : ℝ) 
  (eq1 : 5 * x + y = 19) 
  (eq2 : x + 3 * y = 1) : 
  3 * x + 2 * y = 10 := by
sorry

end NUMINAMATH_CALUDE_system_equation_result_l1674_167426


namespace NUMINAMATH_CALUDE_power_division_result_l1674_167452

theorem power_division_result : 3^12 / 27^2 = 729 := by
  -- Define 27 as 3^3
  have h1 : 27 = 3^3 := by sorry
  
  -- Rewrite the division using the definition of 27
  have h2 : 3^12 / 27^2 = 3^12 / (3^3)^2 := by sorry
  
  -- Simplify the exponents
  have h3 : 3^12 / (3^3)^2 = 3^(12 - 3*2) := by sorry
  
  -- Evaluate the final result
  have h4 : 3^(12 - 3*2) = 3^6 := by sorry
  have h5 : 3^6 = 729 := by sorry
  
  -- Combine all steps
  sorry

#check power_division_result

end NUMINAMATH_CALUDE_power_division_result_l1674_167452


namespace NUMINAMATH_CALUDE_absent_boys_l1674_167414

/-- Proves the number of absent boys in a class with given conditions -/
theorem absent_boys (total_students : ℕ) (girls_present : ℕ) : 
  total_students = 250 →
  girls_present = 140 →
  girls_present = 2 * (total_students - (total_students - (girls_present + girls_present / 2))) →
  total_students - (girls_present + girls_present / 2) = 40 :=
by sorry

end NUMINAMATH_CALUDE_absent_boys_l1674_167414


namespace NUMINAMATH_CALUDE_ian_lap_length_l1674_167453

/-- Represents the jogging scenario for Ian --/
structure JoggingScenario where
  laps_per_night : ℕ
  feet_per_calorie : ℕ
  calories_burned : ℕ
  days_jogged : ℕ

/-- Calculates the number of feet in each lap --/
def feet_per_lap (scenario : JoggingScenario) : ℕ :=
  (scenario.calories_burned * scenario.feet_per_calorie) / (scenario.laps_per_night * scenario.days_jogged)

/-- Theorem stating that Ian's lap length is 100 feet --/
theorem ian_lap_length :
  let scenario : JoggingScenario := {
    laps_per_night := 5,
    feet_per_calorie := 25,
    calories_burned := 100,
    days_jogged := 5
  }
  feet_per_lap scenario = 100 := by
  sorry

end NUMINAMATH_CALUDE_ian_lap_length_l1674_167453


namespace NUMINAMATH_CALUDE_binomial_cube_example_l1674_167400

theorem binomial_cube_example : 4^3 + 3*(4^2)*2 + 3*4*(2^2) + 2^3 = 216 := by
  sorry

end NUMINAMATH_CALUDE_binomial_cube_example_l1674_167400


namespace NUMINAMATH_CALUDE_inequality_sum_l1674_167494

theorem inequality_sum (a b c d : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a > b) (h4 : c > d) : 
  a + c > b + d := by
  sorry

end NUMINAMATH_CALUDE_inequality_sum_l1674_167494


namespace NUMINAMATH_CALUDE_integral_cube_root_problem_l1674_167406

open Real MeasureTheory

theorem integral_cube_root_problem :
  ∫ x in (1 : ℝ)..64, (2 + x^(1/3)) / ((x^(1/6) + 2*x^(1/3) + x^(1/2)) * x^(1/2)) = 6 := by
  sorry

end NUMINAMATH_CALUDE_integral_cube_root_problem_l1674_167406
