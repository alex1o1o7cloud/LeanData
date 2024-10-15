import Mathlib

namespace NUMINAMATH_CALUDE_committee_probability_l3104_310457

def total_members : ℕ := 30
def num_boys : ℕ := 12
def num_girls : ℕ := 18
def committee_size : ℕ := 6

def prob_at_least_one_boy_and_girl : ℚ :=
  1 - (Nat.choose num_boys committee_size + Nat.choose num_girls committee_size) / Nat.choose total_members committee_size

theorem committee_probability :
  prob_at_least_one_boy_and_girl = 574287 / 593775 :=
sorry

end NUMINAMATH_CALUDE_committee_probability_l3104_310457


namespace NUMINAMATH_CALUDE_bus_arrival_probabilities_l3104_310438

def prob_bus_A : ℝ := 0.7
def prob_bus_B : ℝ := 0.75

theorem bus_arrival_probabilities :
  (3 * prob_bus_A^2 * (1 - prob_bus_A) = 0.441) ∧
  (1 - (1 - prob_bus_A) * (1 - prob_bus_B) = 0.925) :=
by sorry

end NUMINAMATH_CALUDE_bus_arrival_probabilities_l3104_310438


namespace NUMINAMATH_CALUDE_x_minus_p_equals_five_minus_two_p_l3104_310431

theorem x_minus_p_equals_five_minus_two_p (x p : ℝ) 
  (h1 : |x - 5| = p) (h2 : x < 5) : x - p = 5 - 2*p := by
  sorry

end NUMINAMATH_CALUDE_x_minus_p_equals_five_minus_two_p_l3104_310431


namespace NUMINAMATH_CALUDE_imaginary_power_sum_l3104_310462

/-- The imaginary unit i -/
def i : ℂ := Complex.I

/-- Statement of the problem -/
theorem imaginary_power_sum : i^23 + i^52 + i^103 = 1 - 2*i := by sorry

end NUMINAMATH_CALUDE_imaginary_power_sum_l3104_310462


namespace NUMINAMATH_CALUDE_problem_1_l3104_310499

theorem problem_1 : 7 - (-3) + (-4) - |(-8)| = -2 := by sorry

end NUMINAMATH_CALUDE_problem_1_l3104_310499


namespace NUMINAMATH_CALUDE_seating_theorem_l3104_310435

/-- Represents a taxi with 4 seats -/
structure Taxi :=
  (front_seat : Fin 1)
  (back_seats : Fin 3)

/-- Represents the number of window seats in a taxi -/
def window_seats : Nat := 2

/-- Represents the total number of passengers -/
def total_passengers : Nat := 4

/-- Calculates the number of seating arrangements in a taxi -/
def seating_arrangements (t : Taxi) (w : Nat) (p : Nat) : Nat :=
  w * (p - 1) * (p - 2) * (p - 3)

/-- Theorem stating that the number of seating arrangements is 12 -/
theorem seating_theorem (t : Taxi) :
  seating_arrangements t window_seats total_passengers = 12 := by
  sorry

end NUMINAMATH_CALUDE_seating_theorem_l3104_310435


namespace NUMINAMATH_CALUDE_two_bedroom_units_l3104_310444

theorem two_bedroom_units (total_units : ℕ) (one_bedroom_cost two_bedroom_cost : ℕ) (total_cost : ℕ) :
  total_units = 12 →
  one_bedroom_cost = 360 →
  two_bedroom_cost = 450 →
  total_cost = 4950 →
  ∃ (one_bedroom_count two_bedroom_count : ℕ),
    one_bedroom_count + two_bedroom_count = total_units ∧
    one_bedroom_count * one_bedroom_cost + two_bedroom_count * two_bedroom_cost = total_cost ∧
    two_bedroom_count = 7 :=
by sorry

end NUMINAMATH_CALUDE_two_bedroom_units_l3104_310444


namespace NUMINAMATH_CALUDE_right_pentagonal_prism_characterization_cone_characterization_l3104_310441

-- Define geometric shapes
def RightPentagonalPrism : Type := sorry
def Cone : Type := sorry

-- Define properties of shapes
def has_seven_faces (shape : Type) : Prop := sorry
def has_two_parallel_congruent_pentagons (shape : Type) : Prop := sorry
def has_congruent_rectangle_faces (shape : Type) : Prop := sorry
def formed_by_rotating_isosceles_triangle (shape : Type) : Prop := sorry
def rotated_180_degrees (shape : Type) : Prop := sorry
def rotated_around_height_line (shape : Type) : Prop := sorry

-- Theorem 1
theorem right_pentagonal_prism_characterization (shape : Type) :
  has_seven_faces shape ∧
  has_two_parallel_congruent_pentagons shape ∧
  has_congruent_rectangle_faces shape →
  shape = RightPentagonalPrism :=
sorry

-- Theorem 2
theorem cone_characterization (shape : Type) :
  formed_by_rotating_isosceles_triangle shape ∧
  rotated_180_degrees shape ∧
  rotated_around_height_line shape →
  shape = Cone :=
sorry

end NUMINAMATH_CALUDE_right_pentagonal_prism_characterization_cone_characterization_l3104_310441


namespace NUMINAMATH_CALUDE_cost_effectiveness_theorem_l3104_310408

/-- Represents the cost of a plan based on the number of students -/
def plan_cost (students : ℕ) (teacher_free : Bool) (discount : ℚ) : ℚ :=
  if teacher_free then
    25 * students
  else
    25 * discount * (students + 1)

/-- Determines which plan is more cost-effective based on the number of students -/
def cost_effective_plan (students : ℕ) : String :=
  let plan1_cost := plan_cost students true 1
  let plan2_cost := plan_cost students false (4/5)
  if plan1_cost < plan2_cost then "Plan 1"
  else if plan1_cost > plan2_cost then "Plan 2"
  else "Both plans are equally cost-effective"

theorem cost_effectiveness_theorem (students : ℕ) :
  cost_effective_plan students =
    if students < 4 then "Plan 1"
    else if students > 4 then "Plan 2"
    else "Both plans are equally cost-effective" :=
  sorry

end NUMINAMATH_CALUDE_cost_effectiveness_theorem_l3104_310408


namespace NUMINAMATH_CALUDE_BaSO4_molecular_weight_l3104_310464

/-- The atomic weight of Barium in g/mol -/
def Ba_weight : ℝ := 137.327

/-- The atomic weight of Sulfur in g/mol -/
def S_weight : ℝ := 32.065

/-- The atomic weight of Oxygen in g/mol -/
def O_weight : ℝ := 15.999

/-- The number of Oxygen atoms in BaSO4 -/
def O_count : ℕ := 4

/-- The molecular weight of BaSO4 in g/mol -/
def BaSO4_weight : ℝ := Ba_weight + S_weight + O_count * O_weight

theorem BaSO4_molecular_weight : BaSO4_weight = 233.388 := by
  sorry

end NUMINAMATH_CALUDE_BaSO4_molecular_weight_l3104_310464


namespace NUMINAMATH_CALUDE_floor_sum_example_l3104_310417

theorem floor_sum_example : ⌊(23.7 : ℝ)⌋ + ⌊(-23.7 : ℝ)⌋ = -1 := by sorry

end NUMINAMATH_CALUDE_floor_sum_example_l3104_310417


namespace NUMINAMATH_CALUDE_chord_equation_l3104_310480

structure Curve where
  equation : ℝ → ℝ × ℝ

structure Line where
  equation : ℝ × ℝ → Prop

def parabola : Curve :=
  { equation := λ t => (4 * t^2, 4 * t) }

def point_on_curve (c : Curve) (p : ℝ × ℝ) : Prop :=
  ∃ t, c.equation t = p

def perpendicular (l1 l2 : Line) : Prop :=
  sorry

def chord_length_product (c : Curve) (l : Line) (p : ℝ × ℝ) : ℝ :=
  sorry

theorem chord_equation (c : Curve) (ab cd : Line) (p : ℝ × ℝ) :
  c = parabola →
  point_on_curve c p →
  p = (2, 2) →
  perpendicular ab cd →
  chord_length_product c ab p = chord_length_product c cd p →
  (ab.equation = λ (x, y) => y = x) ∨ 
  (ab.equation = λ (x, y) => x + y = 4) :=
sorry

end NUMINAMATH_CALUDE_chord_equation_l3104_310480


namespace NUMINAMATH_CALUDE_cos_2alpha_plus_2pi_3_l3104_310498

theorem cos_2alpha_plus_2pi_3 (α : Real) (h : Real.sin (α - π/6) = 2/3) :
  Real.cos (2*α + 2*π/3) = -1/9 := by
  sorry

end NUMINAMATH_CALUDE_cos_2alpha_plus_2pi_3_l3104_310498


namespace NUMINAMATH_CALUDE_units_digit_of_5_pow_17_times_4_l3104_310432

theorem units_digit_of_5_pow_17_times_4 : ∃ n : ℕ, 5^17 * 4 = 10 * n :=
by sorry

end NUMINAMATH_CALUDE_units_digit_of_5_pow_17_times_4_l3104_310432


namespace NUMINAMATH_CALUDE_cube_sum_reciprocal_l3104_310429

theorem cube_sum_reciprocal (x : ℝ) (h : x + 1/x = 4) : x^3 + 1/x^3 = 52 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_reciprocal_l3104_310429


namespace NUMINAMATH_CALUDE_coat_drive_l3104_310415

theorem coat_drive (total_coats : ℕ) (high_school_coats : ℕ) (elementary_coats : ℕ) :
  total_coats = 9437 →
  high_school_coats = 6922 →
  elementary_coats = total_coats - high_school_coats →
  elementary_coats = 2515 := by
  sorry

end NUMINAMATH_CALUDE_coat_drive_l3104_310415


namespace NUMINAMATH_CALUDE_iron_wire_remainder_l3104_310467

theorem iron_wire_remainder (total_length : ℚ) : 
  total_length > 0 → 
  total_length - (2/9 * total_length) - (3/9 * total_length) = 4/9 * total_length := by
sorry

end NUMINAMATH_CALUDE_iron_wire_remainder_l3104_310467


namespace NUMINAMATH_CALUDE_decreasing_function_positive_l3104_310411

/-- A decreasing function satisfying the given condition is always positive -/
theorem decreasing_function_positive (f : ℝ → ℝ) (hf : Monotone (fun x ↦ -f x)) 
    (h : ∀ x, f x / (deriv f x) + x < 1) : ∀ x, f x > 0 := by
  sorry

end NUMINAMATH_CALUDE_decreasing_function_positive_l3104_310411


namespace NUMINAMATH_CALUDE_ratio_percentage_difference_l3104_310456

theorem ratio_percentage_difference (A B : ℝ) (hA : A > 0) (hB : B > 0) (h_ratio : A / B = 1/8 * 7) :
  (B - A) / A * 100 = 100/7 := by
sorry

end NUMINAMATH_CALUDE_ratio_percentage_difference_l3104_310456


namespace NUMINAMATH_CALUDE_f_negative_five_l3104_310494

noncomputable def f (a b x : ℝ) : ℝ := a * Real.sin x + b * Real.tan x + 1

theorem f_negative_five (a b : ℝ) 
  (h : f a b 5 = 7) : 
  f a b (-5) = -5 := by
sorry

end NUMINAMATH_CALUDE_f_negative_five_l3104_310494


namespace NUMINAMATH_CALUDE_tuning_day_method_pi_approximation_l3104_310478

/-- The "Tuning Day Method" for approximating a real number -/
def tuningDayMethod (a b c d : ℕ) : ℚ := (b + d) / (a + c)

/-- Apply the Tuning Day Method n times -/
def applyTuningDayMethod (n : ℕ) (a b c d : ℕ) : ℚ :=
  match n with
  | 0 => b / a
  | n+1 => tuningDayMethod a b c d

theorem tuning_day_method_pi_approximation :
  applyTuningDayMethod 3 10 31 5 16 = 22 / 7 := by
  sorry


end NUMINAMATH_CALUDE_tuning_day_method_pi_approximation_l3104_310478


namespace NUMINAMATH_CALUDE_units_digit_problem_l3104_310437

theorem units_digit_problem (n : ℤ) : n = (30 * 31 * 32 * 33 * 34 * 35) / 2500 → n % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_problem_l3104_310437


namespace NUMINAMATH_CALUDE_white_towels_count_l3104_310404

theorem white_towels_count (green_towels : ℕ) (given_away : ℕ) (remaining : ℕ) : 
  green_towels = 35 → given_away = 34 → remaining = 22 → 
  ∃ white_towels : ℕ, white_towels = 21 ∧ green_towels + white_towels - given_away = remaining :=
by
  sorry

end NUMINAMATH_CALUDE_white_towels_count_l3104_310404


namespace NUMINAMATH_CALUDE_min_abs_sum_with_constraints_l3104_310461

theorem min_abs_sum_with_constraints (α β γ : ℝ) 
  (sum_constraint : α + β + γ = 2)
  (product_constraint : α * β * γ = 4) :
  ∃ v : ℝ, v = 6 ∧ ∀ α' β' γ' : ℝ, 
    (α' + β' + γ' = 2) → (α' * β' * γ' = 4) → 
    v ≤ |α'| + |β'| + |γ'| :=
by sorry

end NUMINAMATH_CALUDE_min_abs_sum_with_constraints_l3104_310461


namespace NUMINAMATH_CALUDE_sin_cos_product_l3104_310436

theorem sin_cos_product (θ : Real) 
  (h : (Real.sin θ + Real.cos θ) / (Real.sin θ - Real.cos θ) = 2) : 
  Real.sin θ * Real.cos θ = 3/10 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_product_l3104_310436


namespace NUMINAMATH_CALUDE_altitude_to_largerBase_ratio_l3104_310434

/-- An isosceles trapezoid with specific properties -/
structure IsoscelesTrapezoid where
  /-- The length of the smaller base -/
  smallerBase : ℝ
  /-- The length of the larger base -/
  largerBase : ℝ
  /-- The length of the altitude -/
  altitude : ℝ
  /-- The smaller base is positive -/
  smallerBase_pos : 0 < smallerBase
  /-- The larger base is positive -/
  largerBase_pos : 0 < largerBase
  /-- The altitude is positive -/
  altitude_pos : 0 < altitude
  /-- The smaller base is less than the larger base -/
  smallerBase_lt_largerBase : smallerBase < largerBase
  /-- The smaller base equals the length of a diagonal -/
  smallerBase_eq_diagonal : smallerBase = Real.sqrt (smallerBase^2 + altitude^2)
  /-- The larger base equals twice the altitude -/
  largerBase_eq_twice_altitude : largerBase = 2 * altitude

/-- The ratio of the altitude to the larger base is 1/2 -/
theorem altitude_to_largerBase_ratio (t : IsoscelesTrapezoid) : 
  t.altitude / t.largerBase = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_altitude_to_largerBase_ratio_l3104_310434


namespace NUMINAMATH_CALUDE_negation_of_quadratic_inequality_l3104_310409

theorem negation_of_quadratic_inequality :
  (∃ x : ℝ, x^2 - x + 3 ≤ 0) ↔ ¬(∀ x : ℝ, x^2 - x + 3 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_quadratic_inequality_l3104_310409


namespace NUMINAMATH_CALUDE_largest_divisor_of_sequence_l3104_310447

theorem largest_divisor_of_sequence :
  ∃ (x : ℕ), x = 18 ∧ 
  (∀ y : ℕ, x ∣ (7^y + 12*y - 1)) ∧
  (∀ z : ℕ, z > x → ∃ w : ℕ, ¬(z ∣ (7^w + 12*w - 1))) := by
  sorry

end NUMINAMATH_CALUDE_largest_divisor_of_sequence_l3104_310447


namespace NUMINAMATH_CALUDE_constant_term_is_60_l3104_310454

/-- The constant term in the expansion of (√x - 2/x)^6 -/
def constantTerm : ℕ :=
  -- We define the constant term without using the solution steps
  -- This definition should be completed in the proof
  sorry

/-- Proof that the constant term in the expansion of (√x - 2/x)^6 is 60 -/
theorem constant_term_is_60 : constantTerm = 60 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_is_60_l3104_310454


namespace NUMINAMATH_CALUDE_average_height_calculation_l3104_310440

theorem average_height_calculation (total_members : ℕ) (average_height : ℝ) 
  (two_member_height : ℝ) (remaining_members : ℕ) :
  total_members = 11 →
  average_height = 145.7 →
  two_member_height = 142.1 →
  remaining_members = total_members - 2 →
  (total_members * average_height - 2 * two_member_height) / remaining_members = 146.5 := by
sorry

end NUMINAMATH_CALUDE_average_height_calculation_l3104_310440


namespace NUMINAMATH_CALUDE_box_volume_calculation_l3104_310439

/-- Given a rectangular metallic sheet and squares cut from each corner, calculate the volume of the resulting box. -/
theorem box_volume_calculation (sheet_length sheet_width cut_square_side : ℝ) 
  (h1 : sheet_length = 48)
  (h2 : sheet_width = 36)
  (h3 : cut_square_side = 4) :
  (sheet_length - 2 * cut_square_side) * (sheet_width - 2 * cut_square_side) * cut_square_side = 4480 := by
  sorry

#check box_volume_calculation

end NUMINAMATH_CALUDE_box_volume_calculation_l3104_310439


namespace NUMINAMATH_CALUDE_parallel_lines_probability_l3104_310423

/-- The number of points (centers of cube faces) -/
def num_points : ℕ := 6

/-- The number of ways to select 2 points from num_points -/
def num_lines : ℕ := num_points.choose 2

/-- The total number of ways for two people to each select a line -/
def total_selections : ℕ := num_lines * num_lines

/-- The number of pairs of lines that are parallel but not coincident -/
def parallel_pairs : ℕ := 12

/-- The probability of selecting two parallel but not coincident lines -/
def probability : ℚ := parallel_pairs / total_selections

theorem parallel_lines_probability :
  probability = 4 / 75 := by sorry

end NUMINAMATH_CALUDE_parallel_lines_probability_l3104_310423


namespace NUMINAMATH_CALUDE_sequence_non_positive_l3104_310489

theorem sequence_non_positive
  (n : ℕ)
  (a : ℕ → ℝ)
  (h0 : a 0 = 0)
  (hn : a n = 0)
  (h_ineq : ∀ k : ℕ, 1 ≤ k ∧ k < n → a (k - 1) + a (k + 1) - 2 * a k ≥ 0) :
  ∀ k : ℕ, k ≤ n → a k ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_sequence_non_positive_l3104_310489


namespace NUMINAMATH_CALUDE_simplify_expression_calculate_expression_l3104_310476

-- Part 1
theorem simplify_expression (x y : ℝ) :
  3 * (2 * x - y) - 2 * (4 * x + 1/2 * y) = -2 * x - 4 * y := by sorry

-- Part 2
theorem calculate_expression (x y : ℝ) (h1 : x * y = 4) (h2 : x - y = -7.5) :
  3 * (x * y - 2/3 * y) - 1/2 * (2 * x + 4 * x * y) - (-2 * x - y) = -7/2 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_calculate_expression_l3104_310476


namespace NUMINAMATH_CALUDE_binomial_10_choose_3_l3104_310475

theorem binomial_10_choose_3 : Nat.choose 10 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_choose_3_l3104_310475


namespace NUMINAMATH_CALUDE_marble_problem_l3104_310412

theorem marble_problem (x y : ℕ) : 
  (y - 4 = 2 * (x + 4)) → 
  (y + 2 = 11 * (x - 2)) → 
  (y = 20 ∧ x = 4) :=
by sorry

end NUMINAMATH_CALUDE_marble_problem_l3104_310412


namespace NUMINAMATH_CALUDE_mrs_heine_biscuits_l3104_310401

theorem mrs_heine_biscuits (num_dogs : ℕ) (biscuits_per_dog : ℕ) 
  (h1 : num_dogs = 2) (h2 : biscuits_per_dog = 3) : 
  num_dogs * biscuits_per_dog = 6 := by
  sorry

end NUMINAMATH_CALUDE_mrs_heine_biscuits_l3104_310401


namespace NUMINAMATH_CALUDE_cone_volume_from_cylinder_l3104_310470

/-- The volume of a cone with the same radius and height as a cylinder with volume 54π cm³ is 18π cm³ -/
theorem cone_volume_from_cylinder (r h : ℝ) (h1 : r > 0) (h2 : h > 0) : 
  π * r^2 * h = 54 * π → (1/3) * π * r^2 * h = 18 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_from_cylinder_l3104_310470


namespace NUMINAMATH_CALUDE_max_daily_revenue_l3104_310443

def price (t : ℕ) : ℝ :=
  if 0 < t ∧ t < 25 then t + 20
  else if 25 ≤ t ∧ t ≤ 30 then -t + 100
  else 0

def sales_volume (t : ℕ) : ℝ :=
  if 0 < t ∧ t ≤ 30 then -t + 40
  else 0

def daily_revenue (t : ℕ) : ℝ :=
  price t * sales_volume t

theorem max_daily_revenue :
  ∃ t : ℕ, 0 < t ∧ t ≤ 30 ∧ daily_revenue t = 1125 ∧
  ∀ s : ℕ, 0 < s ∧ s ≤ 30 → daily_revenue s ≤ daily_revenue t :=
sorry

end NUMINAMATH_CALUDE_max_daily_revenue_l3104_310443


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_min_reciprocal_sum_equality_l3104_310413

theorem min_reciprocal_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_sum : x + y + z = 3) :
  (1 / x + 1 / y + 1 / z) ≥ 3 := by
  sorry

theorem min_reciprocal_sum_equality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_sum : x + y + z = 3) :
  (1 / x + 1 / y + 1 / z = 3) ↔ (x = 1 ∧ y = 1 ∧ z = 1) := by
  sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_min_reciprocal_sum_equality_l3104_310413


namespace NUMINAMATH_CALUDE_square_roots_of_2011_sum_l3104_310426

theorem square_roots_of_2011_sum (x y : ℝ) : 
  x^2 = 2011 → y^2 = 2011 → x + y = 0 := by
sorry

end NUMINAMATH_CALUDE_square_roots_of_2011_sum_l3104_310426


namespace NUMINAMATH_CALUDE_intersection_on_y_axis_l3104_310418

/-- Proves that the intersection point of two specific polynomial graphs is on the y-axis -/
theorem intersection_on_y_axis (a b c : ℝ) (hb : b ≠ 0) (hc : c ≠ 0) :
  ∃! x y : ℝ, (a * x^2 + b^2 * x^3 + c = y) ∧ (a * x^2 - b^2 * x^3 + c = y) ∧ x = 0 ∧ y = c := by
  sorry

end NUMINAMATH_CALUDE_intersection_on_y_axis_l3104_310418


namespace NUMINAMATH_CALUDE_apple_cost_l3104_310452

theorem apple_cost (total_cost : ℝ) (initial_dozen : ℕ) (target_dozen : ℕ) :
  total_cost = 62.40 ∧ initial_dozen = 8 ∧ target_dozen = 5 →
  (target_dozen : ℝ) * (total_cost / initial_dozen) = 39.00 :=
by sorry

end NUMINAMATH_CALUDE_apple_cost_l3104_310452


namespace NUMINAMATH_CALUDE_function_root_implies_a_range_l3104_310479

theorem function_root_implies_a_range (a : ℝ) :
  (∃ x : ℝ, a * x^2 + 2 * x - 1 = 0) → a ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_function_root_implies_a_range_l3104_310479


namespace NUMINAMATH_CALUDE_smallest_integer_y_l3104_310495

theorem smallest_integer_y : ∀ y : ℤ, (7 - 3*y < 22) → y ≥ -4 :=
  sorry

end NUMINAMATH_CALUDE_smallest_integer_y_l3104_310495


namespace NUMINAMATH_CALUDE_quadratic_rewrite_l3104_310469

theorem quadratic_rewrite (b : ℝ) (h1 : b > 0) : 
  (∃ n : ℝ, ∀ x : ℝ, x^2 + b*x + 60 = (x + n)^2 + 16) → b = 4 * Real.sqrt 11 := by
sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_l3104_310469


namespace NUMINAMATH_CALUDE_reflection_property_l3104_310414

/-- A reflection in R² --/
structure Reflection where
  line : ℝ × ℝ  -- Vector representing the line of reflection

/-- Apply a reflection to a point --/
def apply_reflection (r : Reflection) (p : ℝ × ℝ) : ℝ × ℝ := sorry

theorem reflection_property :
  ∃ (r : Reflection),
    apply_reflection r (3, 5) = (7, 1) ∧
    apply_reflection r (2, 7) = (-7, -2) := by
  sorry

end NUMINAMATH_CALUDE_reflection_property_l3104_310414


namespace NUMINAMATH_CALUDE_phi_value_for_even_shifted_function_l3104_310484

/-- Given a function f and a real number φ, proves that if f(x) = (1/2) * sin(2x + π/6)
    and f(x - φ) is an even function, then φ = -π/6 -/
theorem phi_value_for_even_shifted_function 
  (f : ℝ → ℝ) 
  (φ : ℝ) 
  (h1 : ∀ x, f x = (1/2) * Real.sin (2*x + π/6))
  (h2 : ∀ x, f (x - φ) = f (φ - x)) :
  φ = -π/6 := by
  sorry


end NUMINAMATH_CALUDE_phi_value_for_even_shifted_function_l3104_310484


namespace NUMINAMATH_CALUDE_rational_equation_power_l3104_310471

theorem rational_equation_power (x y : ℚ) 
  (h : |x + 5| + (y - 5)^2 = 0) : (x / y)^2023 = -1 := by
  sorry

end NUMINAMATH_CALUDE_rational_equation_power_l3104_310471


namespace NUMINAMATH_CALUDE_rectangle_width_l3104_310446

/-- Given a rectangle with perimeter 48 cm and width 2 cm shorter than length, prove width is 11 cm -/
theorem rectangle_width (length width : ℝ) : 
  (2 * length + 2 * width = 48) →  -- Perimeter condition
  (width = length - 2) →           -- Width-length relation
  (width = 11) :=                  -- Conclusion to prove
by
  sorry

end NUMINAMATH_CALUDE_rectangle_width_l3104_310446


namespace NUMINAMATH_CALUDE_inequality_preservation_l3104_310474

theorem inequality_preservation (a b : ℝ) (h : a < b) : a / 3 < b / 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l3104_310474


namespace NUMINAMATH_CALUDE_i_to_2016_l3104_310473

theorem i_to_2016 (i : ℂ) (h : i^2 = -1) : i^2016 = 1 := by
  sorry

end NUMINAMATH_CALUDE_i_to_2016_l3104_310473


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l3104_310406

/-- A point in polar coordinates -/
structure PolarPoint where
  ρ : ℝ
  θ : ℝ

/-- Curve C₁ in polar coordinates -/
def C₁ (p : PolarPoint) : Prop :=
  p.ρ * (Real.sqrt 2 * Real.cos p.θ + Real.sin p.θ) = 1

/-- Curve C₂ in polar coordinates -/
def C₂ (a : ℝ) (p : PolarPoint) : Prop :=
  p.ρ = a

/-- A point is on the polar axis if its θ coordinate is 0 or π -/
def onPolarAxis (p : PolarPoint) : Prop :=
  p.θ = 0 ∨ p.θ = Real.pi

theorem intersection_implies_a_value (a : ℝ) (h_a_pos : a > 0) :
  (∃ p : PolarPoint, C₁ p ∧ C₂ a p ∧ onPolarAxis p) →
  a = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l3104_310406


namespace NUMINAMATH_CALUDE_cylinder_not_triangle_l3104_310497

-- Define the possible shapes
inductive Shape
  | Cylinder
  | Cone
  | Prism
  | Pyramid

-- Define a function to check if a shape can appear as a triangle
def canAppearAsTriangle (s : Shape) : Prop :=
  match s with
  | Shape.Cylinder => False
  | _ => True

-- Theorem statement
theorem cylinder_not_triangle :
  ∀ s : Shape, canAppearAsTriangle s ↔ s ≠ Shape.Cylinder :=
by
  sorry


end NUMINAMATH_CALUDE_cylinder_not_triangle_l3104_310497


namespace NUMINAMATH_CALUDE_tangent_lines_to_circle_l3104_310448

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A circle in 2D space represented by the equation (x - h)² + (y - k)² = r² -/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Check if a line is tangent to a circle -/
def tangent (l : Line) (c : Circle) : Prop :=
  (c.h * l.a + c.k * l.b + l.c)^2 = (l.a^2 + l.b^2) * c.r^2

theorem tangent_lines_to_circle (given_line : Line) (c : Circle) :
  given_line = Line.mk 2 (-1) 1 ∧ c = Circle.mk 0 0 (Real.sqrt 5) →
  ∃ (l1 l2 : Line),
    l1 = Line.mk 2 (-1) 5 ∧
    l2 = Line.mk 2 (-1) (-5) ∧
    parallel l1 given_line ∧
    parallel l2 given_line ∧
    tangent l1 c ∧
    tangent l2 c ∧
    ∀ (l : Line), parallel l given_line ∧ tangent l c → l = l1 ∨ l = l2 :=
by sorry

end NUMINAMATH_CALUDE_tangent_lines_to_circle_l3104_310448


namespace NUMINAMATH_CALUDE_intersection_with_complement_l3104_310483

-- Define the sets A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def B : Set ℝ := {x | 2 ≤ x ∧ x < 3}

-- State the theorem
theorem intersection_with_complement :
  A ∩ (Set.univ \ B) = {x : ℝ | -1 ≤ x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l3104_310483


namespace NUMINAMATH_CALUDE_only_36_satisfies_conditions_l3104_310420

/-- A two-digit integer is represented by 10a + b, where a and b are single digits -/
def TwoDigitInteger (a b : ℕ) : Prop :=
  a ≥ 1 ∧ a ≤ 9 ∧ b ≥ 0 ∧ b ≤ 9

/-- The sum of digits of a two-digit integer -/
def SumOfDigits (a b : ℕ) : ℕ := a + b

/-- Twice the product of digits of a two-digit integer -/
def TwiceProductOfDigits (a b : ℕ) : ℕ := 2 * a * b

/-- The value of a two-digit integer -/
def IntegerValue (a b : ℕ) : ℕ := 10 * a + b

theorem only_36_satisfies_conditions :
  ∀ a b : ℕ,
    TwoDigitInteger a b →
    (IntegerValue a b % SumOfDigits a b = 0 ∧
     IntegerValue a b % TwiceProductOfDigits a b = 0) →
    IntegerValue a b = 36 :=
by sorry

end NUMINAMATH_CALUDE_only_36_satisfies_conditions_l3104_310420


namespace NUMINAMATH_CALUDE_zoo_population_increase_l3104_310422

theorem zoo_population_increase (c p : ℕ) (h1 : c * 3 = p) (h2 : (c + 2) * 3 = p + 6) : True :=
by sorry

end NUMINAMATH_CALUDE_zoo_population_increase_l3104_310422


namespace NUMINAMATH_CALUDE_one_positive_real_solution_l3104_310493

def f (x : ℝ) : ℝ := x^8 + 5*x^7 + 10*x^6 + 2023*x^5 - 2021*x^4

theorem one_positive_real_solution :
  ∃! (x : ℝ), x > 0 ∧ f x = 0 :=
sorry

end NUMINAMATH_CALUDE_one_positive_real_solution_l3104_310493


namespace NUMINAMATH_CALUDE_existence_of_hundredth_square_l3104_310490

/-- Represents a square grid -/
structure Grid :=
  (size : ℕ)

/-- Represents a square that can be cut out from the grid -/
structure Square :=
  (size : ℕ)
  (position : ℕ × ℕ)

/-- The total number of 2×2 squares that can fit in a grid -/
def total_squares (g : Grid) : ℕ :=
  (g.size - 1) * (g.size - 1)

/-- Predicate to check if a square can be cut out from the grid -/
def can_cut_square (g : Grid) (s : Square) : Prop :=
  s.size = 2 ∧ 
  s.position.1 ≤ g.size - 1 ∧ 
  s.position.2 ≤ g.size - 1

theorem existence_of_hundredth_square (g : Grid) (cut_squares : Finset Square) :
  g.size = 29 →
  cut_squares.card = 99 →
  (∀ s ∈ cut_squares, can_cut_square g s) →
  ∃ s : Square, can_cut_square g s ∧ s ∉ cut_squares :=
sorry

end NUMINAMATH_CALUDE_existence_of_hundredth_square_l3104_310490


namespace NUMINAMATH_CALUDE_unique_solution_exponential_equation_l3104_310491

theorem unique_solution_exponential_equation :
  ∃! x : ℝ, (2 : ℝ) ^ (7 * x + 2) * (4 : ℝ) ^ (2 * x + 5) = (8 : ℝ) ^ (5 * x + 3) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_exponential_equation_l3104_310491


namespace NUMINAMATH_CALUDE_system_two_solutions_l3104_310460

/-- The system of equations has exactly two solutions when a is in the specified interval -/
theorem system_two_solutions (a b : ℝ) : 
  (∃ x y : ℝ, 
    Real.arcsin ((a - y) / 3) = Real.arcsin ((4 - x) / 4) ∧
    x^2 + y^2 - 8*x - 8*y = b) ∧
  (∀ x₁ y₁ x₂ y₂ : ℝ, 
    Real.arcsin ((a - y₁) / 3) = Real.arcsin ((4 - x₁) / 4) ∧
    x₁^2 + y₁^2 - 8*x₁ - 8*y₁ = b ∧
    Real.arcsin ((a - y₂) / 3) = Real.arcsin ((4 - x₂) / 4) ∧
    x₂^2 + y₂^2 - 8*x₂ - 8*y₂ = b ∧
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) →
    ∀ x₃ y₃ : ℝ, 
      Real.arcsin ((a - y₃) / 3) = Real.arcsin ((4 - x₃) / 4) ∧
      x₃^2 + y₃^2 - 8*x₃ - 8*y₃ = b →
      (x₃ = x₁ ∧ y₃ = y₁) ∨ (x₃ = x₂ ∧ y₃ = y₂)) ↔
  -13/3 < a ∧ a < 37/3 :=
by sorry

end NUMINAMATH_CALUDE_system_two_solutions_l3104_310460


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l3104_310410

theorem necessary_not_sufficient_condition (a b : ℝ) :
  (∀ a b, a > b → a + 1 > b) ∧
  (∃ a b, a + 1 > b ∧ ¬(a > b)) :=
by sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l3104_310410


namespace NUMINAMATH_CALUDE_janice_age_l3104_310421

theorem janice_age (current_year : ℕ) (mark_birth_year : ℕ) (graham_age_difference : ℕ) :
  current_year = 2021 →
  mark_birth_year = 1976 →
  graham_age_difference = 3 →
  (current_year - mark_birth_year - graham_age_difference) / 2 = 21 :=
by sorry

end NUMINAMATH_CALUDE_janice_age_l3104_310421


namespace NUMINAMATH_CALUDE_train_distance_problem_l3104_310442

/-- Theorem: Train Distance Problem
Given:
- A passenger train travels from A to B at 60 km/h for 2/3 of the journey, then at 30 km/h for the rest.
- A high-speed train travels at 120 km/h and catches up with the passenger train 80 km before B.
Prove that the distance from A to B is 360 km. -/
theorem train_distance_problem (D : ℝ) 
  (h1 : D > 0)  -- Distance is positive
  (h2 : ∃ t : ℝ, t > 0 ∧ (2/3 * D) / 60 + (1/3 * D) / 30 = (D - 80) / 120 + t)
  : D = 360 := by
  sorry

end NUMINAMATH_CALUDE_train_distance_problem_l3104_310442


namespace NUMINAMATH_CALUDE_f_monotone_decreasing_l3104_310405

-- Define the function f(x) = x^3 - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- State the theorem
theorem f_monotone_decreasing : 
  ∀ x y, -1 < x ∧ x < y ∧ y < 1 → f x > f y :=
sorry

end NUMINAMATH_CALUDE_f_monotone_decreasing_l3104_310405


namespace NUMINAMATH_CALUDE_prize_problem_solution_l3104_310481

/-- Represents the cost and quantity of pens and notebooks --/
structure PrizeInfo where
  pen_cost : ℚ
  notebook_cost : ℚ
  total_prizes : ℕ
  max_total_cost : ℚ

/-- Theorem stating the solution to the prize problem --/
theorem prize_problem_solution (info : PrizeInfo) 
  (h1 : 2 * info.pen_cost + 3 * info.notebook_cost = 62)
  (h2 : 5 * info.pen_cost + info.notebook_cost = 90)
  (h3 : info.total_prizes = 80)
  (h4 : info.max_total_cost = 1100) :
  info.pen_cost = 16 ∧ 
  info.notebook_cost = 10 ∧ 
  (∀ m : ℕ, m * info.pen_cost + (info.total_prizes - m) * info.notebook_cost ≤ info.max_total_cost → m ≤ 50) :=
by sorry


end NUMINAMATH_CALUDE_prize_problem_solution_l3104_310481


namespace NUMINAMATH_CALUDE_inlet_pipe_fill_rate_l3104_310403

/-- Given a tank with specified properties, prove the inlet pipe's fill rate --/
theorem inlet_pipe_fill_rate 
  (tank_capacity : ℝ) 
  (leak_empty_time : ℝ) 
  (combined_empty_time : ℝ) 
  (h1 : tank_capacity = 4320)
  (h2 : leak_empty_time = 6)
  (h3 : combined_empty_time = 8) :
  let leak_rate := tank_capacity / leak_empty_time
  let net_empty_rate := tank_capacity / combined_empty_time
  let inlet_rate := net_empty_rate + leak_rate
  inlet_rate / 60 = 21 := by sorry

end NUMINAMATH_CALUDE_inlet_pipe_fill_rate_l3104_310403


namespace NUMINAMATH_CALUDE_female_managers_count_l3104_310472

/-- Represents a company with employees and managers -/
structure Company where
  total_employees : ℕ
  total_managers : ℕ
  male_employees : ℕ
  male_managers : ℕ
  female_employees : ℕ
  female_managers : ℕ

/-- The conditions of the company as described in the problem -/
def company_conditions (c : Company) : Prop :=
  c.female_employees = 500 ∧
  c.total_managers = (2 * c.total_employees) / 5 ∧
  c.male_managers = (2 * c.male_employees) / 5 ∧
  c.total_employees = c.male_employees + c.female_employees ∧
  c.total_managers = c.male_managers + c.female_managers

/-- The theorem to be proved -/
theorem female_managers_count (c : Company) :
  company_conditions c → c.female_managers = 200 := by
  sorry


end NUMINAMATH_CALUDE_female_managers_count_l3104_310472


namespace NUMINAMATH_CALUDE_max_sum_squared_distances_l3104_310450

/-- The incircle of a triangle -/
def Incircle (A B O : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {P | ∃ θ : ℝ, P = (1 + Real.cos θ, 4/3 + Real.sin θ)}

/-- The squared distance between two points -/
def squaredDistance (P Q : ℝ × ℝ) : ℝ :=
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2

theorem max_sum_squared_distances :
  let A : ℝ × ℝ := (3, 0)
  let B : ℝ × ℝ := (0, 4)
  let O : ℝ × ℝ := (0, 0)
  ∀ P ∈ Incircle A B O,
    squaredDistance P A + squaredDistance P B + squaredDistance P O ≤ 22 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_squared_distances_l3104_310450


namespace NUMINAMATH_CALUDE_loan_principal_calculation_l3104_310427

/-- Calculates the principal amount given the interest rate, time, and total interest. -/
def calculate_principal (rate : ℚ) (time : ℕ) (interest : ℕ) : ℚ :=
  (interest : ℚ) * 100 / (rate * (time : ℚ))

/-- Proves that for a loan with 12% p.a. simple interest, if the interest after 3 years
    is Rs. 5400, then the principal amount borrowed was Rs. 15000. -/
theorem loan_principal_calculation (rate : ℚ) (time : ℕ) (interest : ℕ) :
  rate = 12 → time = 3 → interest = 5400 →
  calculate_principal rate time interest = 15000 := by
  sorry

#eval calculate_principal 12 3 5400

end NUMINAMATH_CALUDE_loan_principal_calculation_l3104_310427


namespace NUMINAMATH_CALUDE_journey_time_calculation_l3104_310482

theorem journey_time_calculation (total_distance : ℝ) (initial_fraction : ℝ) (initial_time : ℝ) (lunch_time : ℝ) :
  total_distance = 200 →
  initial_fraction = 1/4 →
  initial_time = 1 →
  lunch_time = 1 →
  ∃ (total_time : ℝ), total_time = 5 := by
  sorry

end NUMINAMATH_CALUDE_journey_time_calculation_l3104_310482


namespace NUMINAMATH_CALUDE_min_operations_for_square_l3104_310453

-- Define the points
variable (A B C D : Point)

-- Define the operations
def measure_distance (P Q : Point) : ℝ := sorry
def compare_numbers (x y : ℝ) : Bool := sorry

-- Define what it means for ABCD to be a square
def is_square (A B C D : Point) : Prop :=
  let AB := measure_distance A B
  let BC := measure_distance B C
  let CD := measure_distance C D
  let DA := measure_distance D A
  let AC := measure_distance A C
  let BD := measure_distance B D
  (AB = BC) ∧ (BC = CD) ∧ (CD = DA) ∧ (AC = BD)

-- The theorem to prove
theorem min_operations_for_square (A B C D : Point) :
  ∃ (n : ℕ), n = 7 ∧ 
  (∀ (m : ℕ), m < n → ¬∃ (algorithm : Unit → Bool), 
    (algorithm () = true ↔ is_square A B C D)) :=
sorry

end NUMINAMATH_CALUDE_min_operations_for_square_l3104_310453


namespace NUMINAMATH_CALUDE_smaller_omelette_has_three_eggs_l3104_310463

/-- Represents the number of eggs in a smaller omelette -/
def smaller_omelette_eggs : ℕ := sorry

/-- Represents the number of eggs in a larger omelette -/
def larger_omelette_eggs : ℕ := 4

/-- Represents the number of smaller omelettes ordered in the first hour -/
def first_hour_smaller : ℕ := 5

/-- Represents the number of larger omelettes ordered in the second hour -/
def second_hour_larger : ℕ := 7

/-- Represents the number of smaller omelettes ordered in the third hour -/
def third_hour_smaller : ℕ := 3

/-- Represents the number of larger omelettes ordered in the fourth hour -/
def fourth_hour_larger : ℕ := 8

/-- Represents the total number of eggs used -/
def total_eggs : ℕ := 84

/-- Theorem stating that the number of eggs in a smaller omelette is 3 -/
theorem smaller_omelette_has_three_eggs :
  smaller_omelette_eggs = 3 :=
by
  have h1 : first_hour_smaller * smaller_omelette_eggs +
            second_hour_larger * larger_omelette_eggs +
            third_hour_smaller * smaller_omelette_eggs +
            fourth_hour_larger * larger_omelette_eggs = total_eggs := sorry
  sorry

end NUMINAMATH_CALUDE_smaller_omelette_has_three_eggs_l3104_310463


namespace NUMINAMATH_CALUDE_square_perimeter_from_area_l3104_310468

theorem square_perimeter_from_area (s : Real) (area : Real) (perimeter : Real) :
  (s ^ 2 = area) → (area = 36) → (perimeter = 4 * s) → (perimeter = 24) := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_from_area_l3104_310468


namespace NUMINAMATH_CALUDE_farmer_cows_l3104_310407

theorem farmer_cows (initial_cows : ℕ) (added_cows : ℕ) (sold_fraction : ℚ) 
  (h1 : initial_cows = 51)
  (h2 : added_cows = 5)
  (h3 : sold_fraction = 1 / 4) :
  initial_cows + added_cows - ⌊(initial_cows + added_cows : ℚ) * sold_fraction⌋ = 42 := by
  sorry

end NUMINAMATH_CALUDE_farmer_cows_l3104_310407


namespace NUMINAMATH_CALUDE_min_width_proof_l3104_310458

/-- The minimum width of a rectangular area satisfying the given conditions -/
def min_width : ℝ := 5

/-- The length of the rectangular area in terms of its width -/
def length (w : ℝ) : ℝ := w + 15

/-- The area of the rectangular enclosure -/
def area (w : ℝ) : ℝ := w * length w

theorem min_width_proof :
  (∀ w : ℝ, w > 0 → area w ≥ 100 → w ≥ min_width) ∧
  (area min_width ≥ 100) ∧
  (min_width > 0) :=
sorry

end NUMINAMATH_CALUDE_min_width_proof_l3104_310458


namespace NUMINAMATH_CALUDE_largest_three_digit_product_l3104_310449

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

theorem largest_three_digit_product (n x y : ℕ) : 
  100 ≤ n ∧ n < 1000 ∧                  -- n is a three-digit number
  n = x * y * (5 * x + 2 * y) ∧         -- n is the product of x, y, and (5x+2y)
  x < 10 ∧ y < 10 ∧                     -- x and y are less than 10
  is_composite (5 * x + 2 * y) →        -- (5x+2y) is composite
  n ≤ 336 :=                            -- The largest possible value of n is 336
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_product_l3104_310449


namespace NUMINAMATH_CALUDE_ellipse_k_range_l3104_310486

/-- An ellipse equation with parameter k -/
def ellipse_equation (x y k : ℝ) : Prop := x^2 + k*y^2 = 2

/-- Foci of the ellipse are on the y-axis -/
def foci_on_y_axis (k : ℝ) : Prop := sorry

/-- The equation represents an ellipse -/
def is_ellipse (k : ℝ) : Prop := sorry

theorem ellipse_k_range (k : ℝ) : 
  (∀ x y : ℝ, ellipse_equation x y k) → 
  is_ellipse k → 
  foci_on_y_axis k → 
  0 < k ∧ k < 1 :=
sorry

end NUMINAMATH_CALUDE_ellipse_k_range_l3104_310486


namespace NUMINAMATH_CALUDE_triangle_case1_triangle_case2_l3104_310425

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Theorem 1: In a triangle ABC with c = 2, C = π/3, and area = √3, a = 2 and b = 2 -/
theorem triangle_case1 (t : Triangle) 
  (h1 : t.c = 2)
  (h2 : t.C = π / 3)
  (h3 : (1 / 2) * t.a * t.b * Real.sin t.C = Real.sqrt 3) :
  t.a = 2 ∧ t.b = 2 := by
  sorry

/-- Theorem 2: In a triangle ABC with c = 2, C = π/3, and sin C + sin(B-A) = sin 2A,
    either (a = 4√3/3 and b = 2√3/3) or (a = 2 and b = 2) -/
theorem triangle_case2 (t : Triangle)
  (h1 : t.c = 2)
  (h2 : t.C = π / 3)
  (h3 : Real.sin t.C + Real.sin (t.B - t.A) = Real.sin (2 * t.A)) :
  (t.a = (4 * Real.sqrt 3) / 3 ∧ t.b = (2 * Real.sqrt 3) / 3) ∨ 
  (t.a = 2 ∧ t.b = 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_case1_triangle_case2_l3104_310425


namespace NUMINAMATH_CALUDE_prob_one_success_value_min_institutes_l3104_310488

-- Define the probabilities of success for each institute
def prob_A : ℚ := 1/2
def prob_B : ℚ := 1/3
def prob_C : ℚ := 1/4

-- Define the probability of exactly one institute succeeding
def prob_one_success : ℚ := 
  prob_A * (1 - prob_B) * (1 - prob_C) + 
  (1 - prob_A) * prob_B * (1 - prob_C) + 
  (1 - prob_A) * (1 - prob_B) * prob_C

-- Define the function to calculate the probability of at least one success
-- given n institutes with probability p
def prob_at_least_one (n : ℕ) (p : ℚ) : ℚ := 1 - (1 - p)^n

-- Theorem 1: The probability of exactly one institute succeeding is 11/24
theorem prob_one_success_value : prob_one_success = 11/24 := by sorry

-- Theorem 2: The minimum number of institutes with success probability 1/3
-- needed to achieve at least 99/100 overall success probability is 12
theorem min_institutes : 
  (∀ n < 12, prob_at_least_one n (1/3) < 99/100) ∧ 
  prob_at_least_one 12 (1/3) ≥ 99/100 := by sorry

end NUMINAMATH_CALUDE_prob_one_success_value_min_institutes_l3104_310488


namespace NUMINAMATH_CALUDE_largest_k_for_2_pow_15_l3104_310492

/-- The sum of k consecutive odd integers starting from 2m + 1 -/
def sumConsecutiveOdds (m k : ℕ) : ℕ := k * (2 * m + k)

/-- Proposition: The largest value of k for which 2^15 is expressible as the sum of k consecutive odd integers is 128 -/
theorem largest_k_for_2_pow_15 : 
  (∃ (m : ℕ), sumConsecutiveOdds m 128 = 2^15) ∧ 
  (∀ (k : ℕ), k > 128 → ¬∃ (m : ℕ), sumConsecutiveOdds m k = 2^15) := by
  sorry

end NUMINAMATH_CALUDE_largest_k_for_2_pow_15_l3104_310492


namespace NUMINAMATH_CALUDE_simplify_fraction_a_l3104_310445

theorem simplify_fraction_a (a b c d : ℝ) :
  (3 * a^4 * c + 2 * a^4 * d - 3 * b^4 * c - 2 * b^4 * d) /
  ((9 * c^2 * (a - b) - 4 * d^2 * (a - b)) * ((a + b)^2 - 2 * a * b)) =
  (a + b) / (3 * c - 2 * d) :=
sorry


end NUMINAMATH_CALUDE_simplify_fraction_a_l3104_310445


namespace NUMINAMATH_CALUDE_circle_through_points_equation_l3104_310433

/-- A circle passing through three given points -/
structure CircleThroughPoints where
  -- Define the three points
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  -- Ensure the points are distinct
  distinct_points : A ≠ B ∧ B ≠ C ∧ A ≠ C

/-- The equation of a circle in standard form -/
def circle_equation (h k r : ℝ) (x y : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

/-- Main theorem: The circle through the given points has the specified equation -/
theorem circle_through_points_equation (c : CircleThroughPoints) :
  c.A = (-1, -1) →
  c.B = (-8, 0) →
  c.C = (0, 6) →
  ∃ (h k r : ℝ), 
    (h = -4 ∧ k = 3 ∧ r = 5) ∧
    (∀ x y, circle_equation h k r x y ↔ 
      ((x, y) = c.A ∨ (x, y) = c.B ∨ (x, y) = c.C)) :=
by sorry

#check circle_through_points_equation

end NUMINAMATH_CALUDE_circle_through_points_equation_l3104_310433


namespace NUMINAMATH_CALUDE_sqrt_three_irrational_l3104_310466

theorem sqrt_three_irrational : Irrational (Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_irrational_l3104_310466


namespace NUMINAMATH_CALUDE_apples_left_after_pies_l3104_310496

theorem apples_left_after_pies (initial_apples : ℕ) (difference : ℕ) (apples_left : ℕ) : 
  initial_apples = 46 → difference = 32 → apples_left = initial_apples - difference → apples_left = 14 := by
  sorry

end NUMINAMATH_CALUDE_apples_left_after_pies_l3104_310496


namespace NUMINAMATH_CALUDE_even_function_composition_l3104_310400

def is_even_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = g x

theorem even_function_composition (g : ℝ → ℝ) (h : is_even_function g) :
  is_even_function (g ∘ g) :=
by
  sorry

end NUMINAMATH_CALUDE_even_function_composition_l3104_310400


namespace NUMINAMATH_CALUDE_unique_total_prices_l3104_310419

def gift_prices : Finset ℕ := {2, 5, 8, 11, 14}
def box_prices : Finset ℕ := {3, 6, 9, 12, 15}

def total_prices : Finset ℕ := 
  Finset.image (λ (p : ℕ × ℕ) => p.1 + p.2) (gift_prices.product box_prices)

theorem unique_total_prices : Finset.card total_prices = 9 := by
  sorry

end NUMINAMATH_CALUDE_unique_total_prices_l3104_310419


namespace NUMINAMATH_CALUDE_a_property_l3104_310487

def gcd_notation (a b : ℕ) : ℕ := Nat.gcd a b

theorem a_property (a : ℕ) : 
  gcd_notation (gcd_notation a 16) (gcd_notation 18 24) = 2 → 
  Even a ∧ ¬(4 ∣ a) := by
  sorry

end NUMINAMATH_CALUDE_a_property_l3104_310487


namespace NUMINAMATH_CALUDE_go_stones_count_l3104_310451

theorem go_stones_count (n : ℕ) (h1 : n^2 + 3 + 44 = (n + 2)^2) : n^2 + 3 = 103 := by
  sorry

#check go_stones_count

end NUMINAMATH_CALUDE_go_stones_count_l3104_310451


namespace NUMINAMATH_CALUDE_initial_eggs_correct_l3104_310428

/-- The number of eggs initially in the basket -/
def initial_eggs : ℕ := 14

/-- The number of eggs remaining after a customer buys eggs -/
def remaining_eggs (n : ℕ) (eggs : ℕ) : ℕ :=
  eggs - (eggs / 2 + 1)

/-- Theorem stating that the initial number of eggs satisfies the given conditions -/
theorem initial_eggs_correct : 
  let eggs1 := remaining_eggs initial_eggs initial_eggs
  let eggs2 := remaining_eggs eggs1 eggs1
  let eggs3 := remaining_eggs eggs2 eggs2
  eggs3 = 0 := by sorry

end NUMINAMATH_CALUDE_initial_eggs_correct_l3104_310428


namespace NUMINAMATH_CALUDE_exists_valid_numbering_scheme_l3104_310459

/-- Represents a numbering scheme for 7 pins and 7 holes -/
def NumberingScheme := Fin 7 → Fin 7

/-- Checks if a numbering scheme satisfies the condition for a given rotation -/
def isValidForRotation (scheme : NumberingScheme) (rotation : Fin 7) : Prop :=
  ∃ k : Fin 7, scheme k = (k + rotation : Fin 7)

/-- The main theorem stating that there exists a valid numbering scheme -/
theorem exists_valid_numbering_scheme :
  ∃ scheme : NumberingScheme, ∀ rotation : Fin 7, isValidForRotation scheme rotation := by
  sorry

end NUMINAMATH_CALUDE_exists_valid_numbering_scheme_l3104_310459


namespace NUMINAMATH_CALUDE_problem_statement_l3104_310424

theorem problem_statement (a b : ℝ) : 
  ({1, a, b/a} : Set ℝ) = {0, a^2, a+b} → a^2016 + b^2016 = 1 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3104_310424


namespace NUMINAMATH_CALUDE_ornament_profit_theorem_l3104_310416

-- Define the cost and selling prices
def costPriceA : ℝ := 2000
def costPriceB : ℝ := 1500
def sellingPriceA : ℝ := 2500
def sellingPriceB : ℝ := 1800

-- Define the total number of ornaments and maximum budget
def totalOrnaments : ℕ := 20
def maxBudget : ℝ := 36000

-- Define the profit function
def profitFunction (x : ℝ) : ℝ := 200 * x + 6000

-- Theorem statement
theorem ornament_profit_theorem :
  -- Condition 1: Cost price difference
  (costPriceA - costPriceB = 500) →
  -- Condition 2: Equal quantity purchased
  (40000 / costPriceA = 30000 / costPriceB) →
  -- Condition 3: Budget constraint
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ totalOrnaments → costPriceA * x + costPriceB * (totalOrnaments - x) ≤ maxBudget) →
  -- Conclusion 1: Correct profit function
  (∀ x : ℝ, profitFunction x = (sellingPriceA - costPriceA) * x + (sellingPriceB - costPriceB) * (totalOrnaments - x)) ∧
  -- Conclusion 2: Maximum profit
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ totalOrnaments ∧ 
    ∀ y : ℝ, 0 ≤ y ∧ y ≤ totalOrnaments → profitFunction x ≥ profitFunction y) ∧
  profitFunction 12 = 8400 := by
  sorry

end NUMINAMATH_CALUDE_ornament_profit_theorem_l3104_310416


namespace NUMINAMATH_CALUDE_parking_lot_width_l3104_310430

/-- Calculates the width of a parking lot given its specifications -/
theorem parking_lot_width
  (total_length : ℝ)
  (usable_percentage : ℝ)
  (area_per_car : ℝ)
  (total_cars : ℕ)
  (h1 : total_length = 500)
  (h2 : usable_percentage = 0.8)
  (h3 : area_per_car = 10)
  (h4 : total_cars = 16000) :
  (total_length * usable_percentage * (total_cars : ℝ) * area_per_car) / (total_length * usable_percentage) = 400 := by
  sorry

#check parking_lot_width

end NUMINAMATH_CALUDE_parking_lot_width_l3104_310430


namespace NUMINAMATH_CALUDE_min_intersection_cardinality_l3104_310402

-- Define the sets A, B, and C
variable (A B C : Set α)

-- Define the cardinality function
variable (card : Set α → ℕ)

-- Define the conditions
variable (h1 : card A = 50)
variable (h2 : card B = 50)
variable (h3 : card (A ∩ B) = 45)
variable (h4 : card (B ∩ C) = 40)
variable (h5 : card A + card B + card C = card (A ∪ B ∪ C))

-- State the theorem
theorem min_intersection_cardinality :
  ∃ (x : ℕ), x = card (A ∩ B ∩ C) ∧ 
  (∀ (y : ℕ), y = card (A ∩ B ∩ C) → x ≤ y) ∧
  x = 21 := by
sorry

end NUMINAMATH_CALUDE_min_intersection_cardinality_l3104_310402


namespace NUMINAMATH_CALUDE_sports_club_intersection_l3104_310485

theorem sports_club_intersection (total : ℕ) (badminton : ℕ) (tennis : ℕ) (neither : ℕ)
  (h1 : total = 30)
  (h2 : badminton = 17)
  (h3 : tennis = 17)
  (h4 : neither = 2) :
  badminton + tennis - (total - neither) = 6 :=
by sorry

end NUMINAMATH_CALUDE_sports_club_intersection_l3104_310485


namespace NUMINAMATH_CALUDE_simplify_fraction_l3104_310477

theorem simplify_fraction : (120 : ℚ) / 180 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3104_310477


namespace NUMINAMATH_CALUDE_eighth_root_of_256289062500_l3104_310465

theorem eighth_root_of_256289062500 : (256289062500 : ℝ) ^ (1/8 : ℝ) = 52 := by
  sorry

end NUMINAMATH_CALUDE_eighth_root_of_256289062500_l3104_310465


namespace NUMINAMATH_CALUDE_max_crosses_on_10x11_board_l3104_310455

/-- Represents a chessboard -/
structure Chessboard :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a cross shape -/
structure CrossShape :=
  (size : ℕ := 3)
  (coverage : ℕ := 5)

/-- Defines the maximum number of non-overlapping cross shapes on a chessboard -/
def max_non_overlapping_crosses (board : Chessboard) (cross : CrossShape) : ℕ := sorry

/-- Theorem stating the maximum number of non-overlapping cross shapes on a 10x11 chessboard -/
theorem max_crosses_on_10x11_board :
  ∃ (board : Chessboard) (cross : CrossShape),
    board.rows = 10 ∧ board.cols = 11 ∧
    cross.size = 3 ∧ cross.coverage = 5 ∧
    max_non_overlapping_crosses board cross = 15 := by sorry

end NUMINAMATH_CALUDE_max_crosses_on_10x11_board_l3104_310455
