import Mathlib

namespace inequality_theorem_l434_43465

theorem inequality_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (b^2 / a) + (a^2 / b) ≥ a + b := by sorry

end inequality_theorem_l434_43465


namespace intersection_point_of_lines_l434_43461

/-- The intersection point of two lines in 2D space -/
structure IntersectionPoint where
  x : ℝ
  y : ℝ

/-- First line equation: y = x + 1 -/
def line1 (x y : ℝ) : Prop := y = x + 1

/-- Second line equation: y = -x + 1 -/
def line2 (x y : ℝ) : Prop := y = -x + 1

/-- The theorem stating that the intersection point of the two lines is (0, 1) -/
theorem intersection_point_of_lines : 
  ∃ p : IntersectionPoint, line1 p.x p.y ∧ line2 p.x p.y ∧ p.x = 0 ∧ p.y = 1 := by
  sorry

end intersection_point_of_lines_l434_43461


namespace jame_annual_earnings_difference_l434_43485

/-- Calculates the difference in annual earnings between Jame's new job and old job -/
def annual_earnings_difference (
  new_hourly_rate : ℕ) 
  (new_weekly_hours : ℕ)
  (old_hourly_rate : ℕ)
  (old_weekly_hours : ℕ)
  (weeks_per_year : ℕ) : ℕ :=
  ((new_hourly_rate * new_weekly_hours) - (old_hourly_rate * old_weekly_hours)) * weeks_per_year

/-- Proves that the difference in annual earnings between Jame's new job and old job is $20,800 -/
theorem jame_annual_earnings_difference :
  annual_earnings_difference 20 40 16 25 52 = 20800 := by
  sorry

end jame_annual_earnings_difference_l434_43485


namespace outfits_count_l434_43463

/-- The number of possible outfits with different colored shirt and hat -/
def number_of_outfits (red_shirts green_shirts pants green_hats red_hats : ℕ) : ℕ :=
  (red_shirts * green_hats + green_shirts * red_hats) * pants

/-- Theorem stating the number of outfits given the specific quantities -/
theorem outfits_count : number_of_outfits 6 4 7 10 9 = 672 := by
  sorry

end outfits_count_l434_43463


namespace fraction_product_simplification_l434_43473

theorem fraction_product_simplification :
  let fractions : List Rat := 
    (7 / 3) :: 
    (List.range 124).map (fun n => ((8 * (n + 1) + 7) : ℚ) / (8 * (n + 1) - 1))
  (fractions.prod : ℚ) = 333 := by
  sorry

end fraction_product_simplification_l434_43473


namespace tea_consumption_l434_43404

/-- The total number of cups of tea consumed by three merchants -/
def total_cups (s o p : ℝ) : ℝ := s + o + p

/-- Theorem stating that the total cups of tea consumed is 19.5 -/
theorem tea_consumption (s o p : ℝ) 
  (h1 : s + o = 11) 
  (h2 : p + o = 15) 
  (h3 : p + s = 13) : 
  total_cups s o p = 19.5 := by
  sorry

end tea_consumption_l434_43404


namespace smallest_sum_of_squares_l434_43498

theorem smallest_sum_of_squares (x₁ x₂ x₃ : ℝ) (h_pos₁ : 0 < x₁) (h_pos₂ : 0 < x₂) (h_pos₃ : 0 < x₃)
  (h_sum : 2 * x₁ + 3 * x₂ + 4 * x₃ = 120) :
  ∃ (min : ℝ), min = 14400 / 29 ∧ x₁^2 + x₂^2 + x₃^2 ≥ min ∧
  ∃ (y₁ y₂ y₃ : ℝ), 0 < y₁ ∧ 0 < y₂ ∧ 0 < y₃ ∧
    2 * y₁ + 3 * y₂ + 4 * y₃ = 120 ∧ y₁^2 + y₂^2 + y₃^2 = min := by
  sorry

end smallest_sum_of_squares_l434_43498


namespace walkers_meet_at_calculated_point_l434_43422

/-- Two people walking around a loop -/
structure WalkersOnLoop where
  loop_length : ℕ
  speed_ratio : ℕ

/-- The meeting point of two walkers -/
def meeting_point (w : WalkersOnLoop) : ℕ × ℕ :=
  (w.loop_length / (w.speed_ratio + 1), w.speed_ratio * w.loop_length / (w.speed_ratio + 1))

/-- Theorem: Walkers meet at the calculated point -/
theorem walkers_meet_at_calculated_point (w : WalkersOnLoop) 
  (h1 : w.loop_length = 24) 
  (h2 : w.speed_ratio = 3) : 
  meeting_point w = (6, 18) := by
  sorry

#eval meeting_point ⟨24, 3⟩

end walkers_meet_at_calculated_point_l434_43422


namespace not_divisible_by_61_l434_43400

theorem not_divisible_by_61 (x y : ℕ) 
  (h1 : ¬(61 ∣ x))
  (h2 : ¬(61 ∣ y))
  (h3 : 61 ∣ (7*x + 34*y)) :
  ¬(61 ∣ (5*x + 16*y)) := by
sorry

end not_divisible_by_61_l434_43400


namespace min_x_plus_y_l434_43481

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - (2*a + 1) * x + 2

-- State the theorem
theorem min_x_plus_y (a : ℝ) (h : a > 0) :
  (∀ x y : ℝ, y ≥ f a (|x|)) →
  (∃ x₀ y₀ : ℝ, y₀ ≥ f a (|x₀|) ∧ x₀ + y₀ = -a - 1/a) ∧
  (∀ x y : ℝ, y ≥ f a (|x|) → x + y ≥ -a - 1/a) :=
by sorry

end min_x_plus_y_l434_43481


namespace tangent_line_at_one_symmetry_condition_extreme_values_condition_l434_43446

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (1/x + a) * Real.log (1 + x)

-- State the theorems
theorem tangent_line_at_one (a : ℝ) :
  a = -1 → ∃ m b, ∀ x, f a x = m * (x - 1) + b ∧ m = -Real.log 2 ∧ b = 0 := by sorry

theorem symmetry_condition (a : ℝ) :
  (∀ x > 0, f a (1/x) = f a (1/(-2 * x))) ↔ a = 1/2 := by sorry

theorem extreme_values_condition (a : ℝ) :
  (∃ x > 0, ∀ y > 0, f a x ≥ f a y ∨ f a x ≤ f a y) ↔ 0 < a ∧ a < 1/2 := by sorry

end

end tangent_line_at_one_symmetry_condition_extreme_values_condition_l434_43446


namespace last_number_proof_l434_43450

theorem last_number_proof (a b c d : ℝ) : 
  (a + b + c) / 3 = 20 → 
  (b + c + d) / 3 = 15 → 
  a = 33 → 
  d = 18 := by
sorry

end last_number_proof_l434_43450


namespace quadratic_equation_solution_l434_43471

theorem quadratic_equation_solution : 
  ∀ x : ℝ, x^2 - 2*x = 0 ↔ x = 0 ∨ x = 2 := by sorry

end quadratic_equation_solution_l434_43471


namespace calculator_sale_loss_l434_43476

theorem calculator_sale_loss :
  ∀ (x y : ℝ),
    x * (1 + 0.2) = 60 →
    y * (1 - 0.2) = 60 →
    60 + 60 - (x + y) = -5 :=
by
  sorry

end calculator_sale_loss_l434_43476


namespace quadratic_root_l434_43441

theorem quadratic_root (a b c : ℝ) (h_arithmetic : b - a = c - b) 
  (h_a : a = 5) (h_c : c = 1) (h_nonneg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0) 
  (h_one_root : ∃! x, a * x^2 + b * x + c = 0) : 
  ∃ x, a * x^2 + b * x + c = 0 ∧ x = -Real.sqrt 5 / 5 := by
sorry

end quadratic_root_l434_43441


namespace cylinder_surface_area_l434_43456

/-- The total surface area of a cylinder with height 15 and radius 2 is 68π. -/
theorem cylinder_surface_area : 
  let h : ℝ := 15
  let r : ℝ := 2
  let circle_area := π * r^2
  let lateral_area := 2 * π * r * h
  circle_area * 2 + lateral_area = 68 * π := by
sorry

end cylinder_surface_area_l434_43456


namespace exam_fail_percentage_l434_43475

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

end exam_fail_percentage_l434_43475


namespace matt_profit_l434_43454

/-- Represents a baseball card with its value -/
structure Card where
  value : ℕ

/-- Represents a trade of cards -/
structure Trade where
  cardsGiven : List Card
  cardsReceived : List Card

def initialCards : List Card := List.replicate 8 ⟨6⟩

def trade1 : Trade := {
  cardsGiven := [⟨6⟩, ⟨6⟩],
  cardsReceived := [⟨2⟩, ⟨2⟩, ⟨2⟩, ⟨9⟩]
}

def trade2 : Trade := {
  cardsGiven := [⟨2⟩, ⟨6⟩],
  cardsReceived := [⟨5⟩, ⟨5⟩, ⟨8⟩]
}

def trade3 : Trade := {
  cardsGiven := [⟨5⟩, ⟨9⟩],
  cardsReceived := [⟨3⟩, ⟨3⟩, ⟨3⟩, ⟨10⟩, ⟨1⟩]
}

def cardValue (c : Card) : ℕ := c.value

def tradeProfit (t : Trade) : ℤ :=
  (t.cardsReceived.map cardValue).sum - (t.cardsGiven.map cardValue).sum

theorem matt_profit :
  (tradeProfit trade1 + tradeProfit trade2 + tradeProfit trade3 : ℤ) = 19 := by
  sorry

end matt_profit_l434_43454


namespace klinker_age_proof_l434_43496

/-- Mr. Klinker's current age -/
def klinker_age : ℕ := 35

/-- Mr. Klinker's daughter's current age -/
def daughter_age : ℕ := 10

/-- Years into the future when the age relation holds -/
def years_future : ℕ := 15

theorem klinker_age_proof :
  klinker_age = 35 ∧
  daughter_age = 10 ∧
  klinker_age + years_future = 2 * (daughter_age + years_future) := by
  sorry

#check klinker_age_proof

end klinker_age_proof_l434_43496


namespace min_value_trig_expression_l434_43486

theorem min_value_trig_expression (α β : ℝ) :
  (3 * Real.cos α + 6 * Real.sin β - 10)^2 + (3 * Real.sin α + 6 * Real.cos β - 18)^2 ≥ 121 ∧
  ∃ α₀ β₀ : ℝ, (3 * Real.cos α₀ + 6 * Real.sin β₀ - 10)^2 + (3 * Real.sin α₀ + 6 * Real.cos β₀ - 18)^2 = 121 :=
by sorry

end min_value_trig_expression_l434_43486


namespace complex_power_problem_l434_43447

theorem complex_power_problem (z : ℂ) (i : ℂ) (h : i^2 = -1) (eq : (1 + z) / (1 - z) = i) : z^2019 = -i := by
  sorry

end complex_power_problem_l434_43447


namespace actual_speed_proof_l434_43445

theorem actual_speed_proof (v : ℝ) (h : (v / (v + 10) = 3 / 4)) : v = 30 := by
  sorry

end actual_speed_proof_l434_43445


namespace initial_percentage_chemical_x_l434_43457

/-- Given an 80-liter mixture and adding 20 liters of pure chemical x resulting in a 100-liter mixture that is 44% chemical x, prove that the initial percentage of chemical x was 30%. -/
theorem initial_percentage_chemical_x : 
  ∀ (initial_percentage : ℝ),
  initial_percentage ≥ 0 ∧ initial_percentage ≤ 1 →
  (80 * initial_percentage + 20) / 100 = 0.44 →
  initial_percentage = 0.3 := by
sorry

end initial_percentage_chemical_x_l434_43457


namespace min_coach_handshakes_l434_43489

def total_handshakes (na nb : ℕ) : ℕ :=
  (na + nb) * (na + nb - 1) / 2 + na + nb

def is_valid_configuration (na nb : ℕ) : Prop :=
  na < nb ∧ total_handshakes na nb = 465

theorem min_coach_handshakes :
  ∃ (na nb : ℕ), is_valid_configuration na nb ∧
  ∀ (ma mb : ℕ), is_valid_configuration ma mb → na ≤ ma :=
by sorry

end min_coach_handshakes_l434_43489


namespace catfish_count_l434_43421

theorem catfish_count (C : ℕ) (total_fish : ℕ) : 
  C + 10 + (3 * C / 2) = total_fish ∧ total_fish = 50 → C = 16 := by
  sorry

end catfish_count_l434_43421


namespace equal_intersection_areas_exist_l434_43451

/-- Represents a tetrahedron with specific properties -/
structure Tetrahedron where
  opposite_edges_perpendicular : Bool
  opposite_edges_horizontal : Bool
  vertical_midline : Bool

/-- Represents a sphere -/
structure Sphere where
  radius : ℝ

/-- Represents the configuration of a tetrahedron and a sphere -/
structure Configuration where
  tetrahedron : Tetrahedron
  sphere : Sphere
  sphere_centered_on_midline : Bool

/-- Represents a horizontal plane -/
structure HorizontalPlane where
  height : ℝ

/-- Function to calculate the area of intersection between a horizontal plane and the tetrahedron -/
def tetrahedron_intersection_area (t : Tetrahedron) (p : HorizontalPlane) : ℝ := sorry

/-- Function to calculate the area of intersection between a horizontal plane and the sphere -/
def sphere_intersection_area (s : Sphere) (p : HorizontalPlane) : ℝ := sorry

/-- Theorem stating that there exists a configuration where all horizontal plane intersections have equal areas -/
theorem equal_intersection_areas_exist : 
  ∃ (c : Configuration), ∀ (p : HorizontalPlane), 
    tetrahedron_intersection_area c.tetrahedron p = sphere_intersection_area c.sphere p :=
sorry

end equal_intersection_areas_exist_l434_43451


namespace function_properties_l434_43411

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log x + a * x^2 - 3

-- State the theorem
theorem function_properties (a : ℝ) :
  (∀ x > 0, HasDerivAt (f a) ((-1 : ℝ)) 1) →
  (∃ m : ℝ, m = -1 ∧ 
    (∀ x > 0, f (-1) x - m * x ≤ -3) ∧
    (∀ m' < m, ∃ x > 0, f (-1) x - m' * x > -3)) ∧
  (∀ x > 0, x * Real.log x - x^2 - 3 - x * Real.exp x + x^2 < -2 * x - 3) :=
by sorry

end function_properties_l434_43411


namespace seventh_house_number_l434_43435

theorem seventh_house_number (k : ℕ) (p : ℕ) : 
  p = 5 →
  k * (p + k - 1) = 2021 →
  p + 2 * (7 - 1) = 17 :=
by
  sorry

end seventh_house_number_l434_43435


namespace luca_drink_cost_l434_43416

/-- The cost of Luca's lunch items and the total bill -/
structure LunchCost where
  sandwich : ℝ
  discount_rate : ℝ
  avocado : ℝ
  salad : ℝ
  total_bill : ℝ

/-- Calculate the cost of Luca's drink given his lunch costs -/
def drink_cost (lunch : LunchCost) : ℝ :=
  lunch.total_bill - (lunch.sandwich * (1 - lunch.discount_rate) + lunch.avocado + lunch.salad)

/-- Theorem: Given Luca's lunch costs, the cost of his drink is $2 -/
theorem luca_drink_cost :
  let lunch : LunchCost := {
    sandwich := 8,
    discount_rate := 0.25,
    avocado := 1,
    salad := 3,
    total_bill := 12
  }
  drink_cost lunch = 2 := by sorry

end luca_drink_cost_l434_43416


namespace age_difference_proof_l434_43452

theorem age_difference_proof (elder_age younger_age : ℕ) : 
  elder_age > younger_age →
  elder_age - 10 = 5 * (younger_age - 10) →
  elder_age = 35 →
  younger_age = 15 →
  elder_age - younger_age = 20 := by
sorry

end age_difference_proof_l434_43452


namespace cubic_root_ratio_l434_43407

theorem cubic_root_ratio (p q r s : ℝ) (h : ∀ x, p * x^3 + q * x^2 + r * x + s = 0 ↔ x = -1 ∨ x = 3 ∨ x = 4) :
  r / s = -5 / 12 := by sorry

end cubic_root_ratio_l434_43407


namespace gcd_372_684_l434_43414

theorem gcd_372_684 : Nat.gcd 372 684 = 12 := by
  sorry

end gcd_372_684_l434_43414


namespace cherry_olive_discount_l434_43436

theorem cherry_olive_discount (cherry_price olives_price bags_count total_cost : ℝ) :
  cherry_price = 5 →
  olives_price = 7 →
  bags_count = 50 →
  total_cost = 540 →
  let original_cost := cherry_price * bags_count + olives_price * bags_count
  let discount_amount := original_cost - total_cost
  let discount_percentage := (discount_amount / original_cost) * 100
  discount_percentage = 10 := by
sorry

end cherry_olive_discount_l434_43436


namespace judson_contribution_is_500_l434_43402

def house_painting_problem (judson_contribution : ℝ) : Prop :=
  let kenny_contribution := 1.2 * judson_contribution
  let camilo_contribution := kenny_contribution + 200
  judson_contribution + kenny_contribution + camilo_contribution = 1900

theorem judson_contribution_is_500 :
  ∃ (judson_contribution : ℝ),
    house_painting_problem judson_contribution ∧ judson_contribution = 500 :=
by
  sorry

end judson_contribution_is_500_l434_43402


namespace sin_x_plus_pi_l434_43426

theorem sin_x_plus_pi (x : ℝ) (h1 : x ∈ Set.Ioo (-π/2) 0) (h2 : Real.tan x = -4/3) :
  Real.sin (x + π) = 4/5 := by
  sorry

end sin_x_plus_pi_l434_43426


namespace theater_ticket_pricing_l434_43477

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

end theater_ticket_pricing_l434_43477


namespace tenth_ring_squares_l434_43460

/-- The number of unit squares in the nth ring around a 3x3 center block -/
def ring_squares (n : ℕ) : ℕ := 8 * n + 8

/-- The 10th ring around a 3x3 center block contains 88 unit squares -/
theorem tenth_ring_squares : ring_squares 10 = 88 := by sorry

end tenth_ring_squares_l434_43460


namespace daisy_field_count_l434_43459

theorem daisy_field_count : ∃! n : ℕ,
  (n : ℚ) / 14 + 2 * ((n : ℚ) / 14) + 4 * ((n : ℚ) / 14) + 7000 = n ∧
  n > 0 :=
by
  sorry

end daisy_field_count_l434_43459


namespace sqrt_nine_minus_sqrt_four_l434_43448

theorem sqrt_nine_minus_sqrt_four : Real.sqrt 9 - Real.sqrt 4 = 1 := by
  sorry

end sqrt_nine_minus_sqrt_four_l434_43448


namespace vector_magnitude_l434_43419

/-- Given plane vectors a and b with angle π/2 between them, |a| = 1, and |b| = √3, prove |2a - b| = √7 -/
theorem vector_magnitude (a b : ℝ × ℝ) : 
  (a • b = 0) → (‖a‖ = 1) → (‖b‖ = Real.sqrt 3) → ‖2 • a - b‖ = Real.sqrt 7 := by
  sorry

end vector_magnitude_l434_43419


namespace fixed_point_of_log_function_l434_43432

-- Define the logarithm function
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Define the function f(x) = 1 + logₐ x
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1 + log a x

-- Theorem statement
theorem fixed_point_of_log_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  f a 1 = 1 := by
  sorry

end fixed_point_of_log_function_l434_43432


namespace drug_efficacy_rate_l434_43433

/-- Calculates the efficacy rate of a drug based on a survey --/
def efficacyRate (totalSamples : ℕ) (positiveResponses : ℕ) : ℚ :=
  (positiveResponses : ℚ) / (totalSamples : ℚ)

theorem drug_efficacy_rate :
  let totalSamples : ℕ := 20
  let positiveResponses : ℕ := 16
  efficacyRate totalSamples positiveResponses = 4/5 := by
sorry

end drug_efficacy_rate_l434_43433


namespace f_minimum_f_le_g_iff_exists_three_roots_l434_43428

noncomputable section

-- Define the functions f and g
def f (x : ℝ) : ℝ := x * Real.log x
def g (a : ℝ) (x : ℝ) : ℝ := a * x^2 - x

-- Statement 1: f has a minimum at x = 1/e
theorem f_minimum : ∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → f y ≥ f x := by sorry

-- Statement 2: f(x) ≤ g(x) for all x > 0 iff a ≥ 1
theorem f_le_g_iff (a : ℝ) : (∀ x > 0, f x ≤ g a x) ↔ a ≥ 1 := by sorry

-- Statement 3: When a = 1/8, there exists m such that 3f(x)/(4x) + m + g(x) = 0 has three distinct real roots iff 7/8 < m < 15/8 - 3/4 * ln 3
theorem exists_three_roots :
  ∃ (m : ℝ), (∃ (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    (3 * f x) / (4 * x) + m + g (1/8) x = 0 ∧
    (3 * f y) / (4 * y) + m + g (1/8) y = 0 ∧
    (3 * f z) / (4 * z) + m + g (1/8) z = 0) ↔
  (7/8 < m ∧ m < 15/8 - 3/4 * Real.log 3) := by sorry

end

end f_minimum_f_le_g_iff_exists_three_roots_l434_43428


namespace train_speed_and_length_l434_43410

/-- Given a train passing a stationary observer in 7 seconds and taking 25 seconds to pass a 378-meter platform at constant speed, prove that the train's speed is 21 m/s and its length is 147 m. -/
theorem train_speed_and_length :
  ∀ (V l : ℝ),
  (7 * V = l) →
  (25 * V = 378 + l) →
  (V = 21 ∧ l = 147) :=
by sorry

end train_speed_and_length_l434_43410


namespace inequality_range_theorem_l434_43401

/-- The range of a for which |x+3| - |x-1| ≤ a^2 - 3a holds for all real x -/
theorem inequality_range_theorem (a : ℝ) : 
  (∀ x : ℝ, |x + 3| - |x - 1| ≤ a^2 - 3*a) ↔ 
  (a ≤ -1 ∨ a ≥ 4) :=
by sorry

end inequality_range_theorem_l434_43401


namespace min_value_reciprocal_sum_l434_43466

theorem min_value_reciprocal_sum (a b : ℝ) : 
  a > 0 → b > 0 → 2*a + 2*b = 2 → (1/a + 1/b ≥ 4) ∧ (∃ a b, 1/a + 1/b = 4) :=
by sorry

end min_value_reciprocal_sum_l434_43466


namespace remainder_sum_mod_l434_43468

theorem remainder_sum_mod (x y : ℤ) (hx : x ≠ y) 
  (hx_mod : x % 124 = 13) (hy_mod : y % 186 = 17) : 
  (x + y + 19) % 62 = 49 := by
  sorry

end remainder_sum_mod_l434_43468


namespace function_inequality_implies_upper_bound_l434_43405

open Real

theorem function_inequality_implies_upper_bound (f : ℝ → ℝ) (a : ℝ) :
  (∀ x > 0, f x = x * log x) →
  (∀ x > 0, f x ≥ -x^2 + a*x - 6) →
  a ≤ 5 + log 2 := by
  sorry

end function_inequality_implies_upper_bound_l434_43405


namespace peaches_for_juice_l434_43438

def total_peaches : ℝ := 7.5

def drying_percentage : ℝ := 0.3

def juice_percentage_of_remainder : ℝ := 0.4

theorem peaches_for_juice :
  let remaining_after_drying := total_peaches * (1 - drying_percentage)
  let juice_amount := remaining_after_drying * juice_percentage_of_remainder
  juice_amount = 2.1 := by sorry

end peaches_for_juice_l434_43438


namespace cats_sold_during_sale_l434_43403

theorem cats_sold_during_sale 
  (initial_siamese : ℕ) 
  (initial_house : ℕ) 
  (remaining : ℕ) 
  (h1 : initial_siamese = 13) 
  (h2 : initial_house = 5) 
  (h3 : remaining = 8) : 
  initial_siamese + initial_house - remaining = 10 := by
sorry

end cats_sold_during_sale_l434_43403


namespace distinct_nonneg_inequality_l434_43430

theorem distinct_nonneg_inequality (a b c : ℝ) 
  (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) : 
  a^2 + b^2 + c^2 > Real.sqrt (a*b*c) * (Real.sqrt a + Real.sqrt b + Real.sqrt c) := by
  sorry

end distinct_nonneg_inequality_l434_43430


namespace train_b_length_l434_43491

/-- Calculates the length of Train B given the conditions of the problem -/
theorem train_b_length : 
  let train_a_speed : ℝ := 10  -- Initial speed of Train A in m/s
  let train_b_speed : ℝ := 12.5  -- Initial speed of Train B in m/s
  let train_a_accel : ℝ := 1  -- Acceleration of Train A in m/s²
  let train_b_decel : ℝ := 0.5  -- Deceleration of Train B in m/s²
  let passing_time : ℝ := 10  -- Time to pass each other in seconds
  
  let train_a_final_speed := train_a_speed + train_a_accel * passing_time
  let train_b_final_speed := train_b_speed - train_b_decel * passing_time
  let relative_speed := train_a_final_speed + train_b_final_speed
  
  relative_speed * passing_time = 275 := by
  sorry

#check train_b_length

end train_b_length_l434_43491


namespace regular_tetrahedron_face_center_volume_ratio_l434_43495

/-- The ratio of the volume of a tetrahedron formed by the centers of the faces of a regular tetrahedron to the volume of the original tetrahedron -/
def face_center_tetrahedron_volume_ratio : ℚ :=
  8 / 27

/-- Theorem stating that in a regular tetrahedron, the ratio of the volume of the tetrahedron 
    formed by the centers of the faces to the volume of the original tetrahedron is 8/27 -/
theorem regular_tetrahedron_face_center_volume_ratio :
  face_center_tetrahedron_volume_ratio = 8 / 27 := by
  sorry

#eval Nat.gcd 8 27  -- To verify that 8 and 27 are coprime

#eval 8 + 27  -- To compute the final answer

end regular_tetrahedron_face_center_volume_ratio_l434_43495


namespace profit_percentage_calculation_l434_43482

theorem profit_percentage_calculation (cost_price selling_price : ℚ) : 
  cost_price = 500 → selling_price = 750 → 
  (selling_price - cost_price) / cost_price * 100 = 50 := by
  sorry

end profit_percentage_calculation_l434_43482


namespace distinct_triangles_count_l434_43497

/-- Represents a 2x4 grid of points -/
def Grid := Fin 2 × Fin 4

/-- Represents a triangle formed by three points on the grid -/
def Triangle := Fin 3 → Grid

/-- Checks if three points are collinear -/
def collinear (p q r : Grid) : Prop := sorry

/-- Counts the number of distinct triangles in a 2x4 grid -/
def count_distinct_triangles : ℕ := sorry

/-- Theorem stating that the number of distinct triangles in a 2x4 grid is 44 -/
theorem distinct_triangles_count : count_distinct_triangles = 44 := by sorry

end distinct_triangles_count_l434_43497


namespace existence_of_special_divisibility_pair_l434_43455

theorem existence_of_special_divisibility_pair : 
  ∃ (a b : ℕ+), 
    a ∣ b^2 ∧ 
    b^2 ∣ a^3 ∧ 
    a^3 ∣ b^4 ∧ 
    b^4 ∣ a^5 ∧ 
    ¬(a^5 ∣ b^6) := by
  sorry

end existence_of_special_divisibility_pair_l434_43455


namespace extreme_values_and_tangent_lines_l434_43413

-- Define the function f(x) = x³ - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the closed interval [0, 2]
def I : Set ℝ := Set.Icc 0 2

theorem extreme_values_and_tangent_lines :
  -- Part 1: Extreme values
  (∃ x ∈ I, f x = 2 ∧ ∀ y ∈ I, f y ≤ 2) ∧
  (∃ x ∈ I, f x = -2 ∧ ∀ y ∈ I, f y ≥ -2) ∧
  -- Part 2: Range of m for three tangent lines
  ∀ m : ℝ, (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    (f x₁ - m) / (x₁ - 2) = 3 * x₁^2 - 3 ∧
    (f x₂ - m) / (x₂ - 2) = 3 * x₂^2 - 3 ∧
    (f x₃ - m) / (x₃ - 2) = 3 * x₃^2 - 3) ↔
  -6 < m ∧ m < 2 :=
sorry

end extreme_values_and_tangent_lines_l434_43413


namespace matrix_power_property_l434_43406

theorem matrix_power_property (A : Matrix (Fin 2) (Fin 2) ℝ) :
  A.mulVec (![5, -2]) = ![(-15), 6] →
  (A ^ 5).mulVec (![5, -2]) = ![(-1215), 486] := by
  sorry

end matrix_power_property_l434_43406


namespace dragon_can_be_defeated_l434_43429

/-- Represents the possible strikes and their corresponding regrowth --/
inductive Strike : Type
| one : Strike
| seventeen : Strike
| twentyone : Strike
| thirtythree : Strike

/-- Returns the number of heads chopped for a given strike --/
def heads_chopped (s : Strike) : ℕ :=
  match s with
  | Strike.one => 1
  | Strike.seventeen => 17
  | Strike.twentyone => 21
  | Strike.thirtythree => 33

/-- Returns the number of heads that grow back for a given strike --/
def heads_regrown (s : Strike) : ℕ :=
  match s with
  | Strike.one => 10
  | Strike.seventeen => 14
  | Strike.twentyone => 0
  | Strike.thirtythree => 48

/-- Represents the state of the dragon --/
structure DragonState :=
  (heads : ℕ)

/-- Applies a strike to the dragon state --/
def apply_strike (state : DragonState) (s : Strike) : DragonState :=
  let new_heads := state.heads - heads_chopped s + heads_regrown s
  ⟨max new_heads 0⟩

/-- Theorem: There exists a sequence of strikes that defeats the dragon --/
theorem dragon_can_be_defeated :
  ∃ (sequence : List Strike), (sequence.foldl apply_strike ⟨2000⟩).heads = 0 :=
sorry

end dragon_can_be_defeated_l434_43429


namespace cycle_selling_price_l434_43479

theorem cycle_selling_price (cost_price : ℝ) (loss_percentage : ℝ) (selling_price : ℝ) : 
  cost_price = 1900 →
  loss_percentage = 18 →
  selling_price = cost_price * (1 - loss_percentage / 100) →
  selling_price = 1558 := by
sorry

end cycle_selling_price_l434_43479


namespace elise_remaining_money_l434_43418

/-- Calculates the remaining money for Elise given her initial amount, savings, and expenses. -/
def remaining_money (initial : ℕ) (savings : ℕ) (comic_expense : ℕ) (puzzle_expense : ℕ) : ℕ :=
  initial + savings - comic_expense - puzzle_expense

/-- Proves that Elise's remaining money is $1 given her initial amount, savings, and expenses. -/
theorem elise_remaining_money :
  remaining_money 8 13 2 18 = 1 := by
  sorry

#eval remaining_money 8 13 2 18

end elise_remaining_money_l434_43418


namespace shadow_problem_l434_43488

/-- Given a cube with edge length 2 cm and a light source y cm above one of its upper vertices
    casting a shadow of 324 sq cm (excluding the area beneath the cube),
    prove that the largest integer less than or equal to 500y is 8000. -/
theorem shadow_problem (y : ℝ) : 
  (2 : ℝ) > 0 ∧ y > 0 ∧ 
  (((18 : ℝ)^2 - 2^2) = 324) ∧
  ((y / 2) = ((18 : ℝ) - 2) / 2) →
  ⌊500 * y⌋ = 8000 := by sorry

end shadow_problem_l434_43488


namespace grandpas_tomatoes_l434_43490

/-- The number of tomatoes that grew in Grandpa's absence -/
def tomatoesGrown (initialCount : ℕ) (growthFactor : ℕ) : ℕ :=
  initialCount * growthFactor - initialCount

theorem grandpas_tomatoes :
  tomatoesGrown 36 100 = 3564 := by
  sorry

end grandpas_tomatoes_l434_43490


namespace jeds_cards_after_four_weeks_l434_43467

/-- Calculates the number of cards Jed has after a given number of weeks -/
def cards_after_weeks (initial_cards : ℕ) (cards_per_week : ℕ) (cards_given_away : ℕ) (weeks : ℕ) : ℕ :=
  initial_cards + cards_per_week * weeks - cards_given_away * (weeks / 2)

/-- Proves that Jed will have 40 cards after 4 weeks -/
theorem jeds_cards_after_four_weeks :
  ∃ (weeks : ℕ), cards_after_weeks 20 6 2 weeks = 40 ∧ weeks = 4 :=
by
  sorry

#check jeds_cards_after_four_weeks

end jeds_cards_after_four_weeks_l434_43467


namespace four_number_sequence_l434_43444

theorem four_number_sequence : ∃ (a b c d : ℝ), 
  (∃ (q : ℝ), b = a * q ∧ c = b * q) ∧  -- Geometric progression
  (∃ (r : ℝ), c = b + r ∧ d = c + r) ∧  -- Arithmetic progression
  a + d = 21 ∧                          -- Sum of first and last
  b + c = 18 := by                      -- Sum of middle two
sorry

end four_number_sequence_l434_43444


namespace circle_radius_from_spherical_coords_l434_43423

/-- The radius of the circle formed by points with spherical coordinates (1, θ, π/4) is √2/2 -/
theorem circle_radius_from_spherical_coords :
  let r : ℝ := Real.sqrt 2 / 2
  ∀ θ : ℝ,
  let x : ℝ := Real.sin (π/4 : ℝ) * Real.cos θ
  let y : ℝ := Real.sin (π/4 : ℝ) * Real.sin θ
  Real.sqrt (x^2 + y^2) = r :=
by sorry

end circle_radius_from_spherical_coords_l434_43423


namespace gh_length_is_60_over_77_l434_43484

/-- Represents a right triangle with squares inscribed -/
structure RightTriangleWithSquares where
  -- Right triangle ABC
  AC : ℝ
  BC : ℝ
  -- Square DEFG
  DE : ℝ
  -- Square GHIJ
  GH : ℝ
  -- Condition that E lies on AC and I lies on BC
  E_on_AC : ℝ
  I_on_BC : ℝ
  -- J is the midpoint of DG
  DJ : ℝ

/-- The length of GH in the inscribed square configuration -/
def ghLength (t : RightTriangleWithSquares) : ℝ := t.GH

/-- Theorem stating the length of GH in the given configuration -/
theorem gh_length_is_60_over_77 (t : RightTriangleWithSquares) 
  (h1 : t.AC = 4) 
  (h2 : t.BC = 3) 
  (h3 : t.DE = 2 * t.GH) 
  (h4 : t.DJ = t.GH) 
  (h5 : t.E_on_AC + t.DE + t.GH = t.AC) 
  (h6 : t.I_on_BC + t.GH = t.BC) :
  ghLength t = 60 / 77 := by
  sorry


end gh_length_is_60_over_77_l434_43484


namespace anoop_join_time_l434_43449

/-- Prove that Anoop joined after 6 months given the investment conditions -/
theorem anoop_join_time (arjun_investment : ℕ) (anoop_investment : ℕ) (total_months : ℕ) :
  arjun_investment = 20000 →
  anoop_investment = 40000 →
  total_months = 12 →
  ∃ x : ℕ, 
    (arjun_investment * total_months = anoop_investment * (total_months - x)) ∧
    x = 6 := by
  sorry

end anoop_join_time_l434_43449


namespace gift_wrapping_calculation_l434_43434

/-- Represents the gift wrapping scenario for Edmund's shop. -/
structure GiftWrapping where
  wrapper_per_day : ℕ        -- inches of gift wrapper per day
  boxes_per_period : ℕ       -- number of gift boxes wrapped in a period
  days_per_period : ℕ        -- number of days in a period
  wrapper_per_box : ℕ        -- inches of gift wrapper per gift box

/-- Theorem stating the relationship between gift wrapper usage and gift boxes wrapped. -/
theorem gift_wrapping_calculation (g : GiftWrapping)
  (h1 : g.wrapper_per_day = 90)
  (h2 : g.boxes_per_period = 15)
  (h3 : g.days_per_period = 3)
  : g.wrapper_per_box = 18 := by
  sorry


end gift_wrapping_calculation_l434_43434


namespace top_view_area_is_eight_l434_43469

/-- Represents the dimensions of a rectangular prism -/
structure PrismDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the area of the top view of a rectangular prism -/
def topViewArea (d : PrismDimensions) : ℝ := d.length * d.width

/-- Theorem: The area of the top view of the given rectangular prism is 8 square units -/
theorem top_view_area_is_eight :
  let d : PrismDimensions := { length := 4, width := 2, height := 3 }
  topViewArea d = 8 := by
  sorry

end top_view_area_is_eight_l434_43469


namespace stone_piles_sum_l434_43443

/-- Represents the number of stones in each pile -/
structure StonePiles where
  pile1 : ℕ
  pile2 : ℕ
  pile3 : ℕ
  pile4 : ℕ
  pile5 : ℕ

/-- Conditions for the stone piles problem -/
def validStonePiles (p : StonePiles) : Prop :=
  p.pile5 = 6 * p.pile3 ∧
  p.pile2 = 2 * (p.pile3 + p.pile5) ∧
  p.pile1 = p.pile5 / 3 ∧
  p.pile1 = p.pile4 - 10 ∧
  p.pile4 = p.pile2 / 2

theorem stone_piles_sum (p : StonePiles) (h : validStonePiles p) :
  p.pile1 + p.pile2 + p.pile3 + p.pile4 + p.pile5 = 60 := by
  sorry

#check stone_piles_sum

end stone_piles_sum_l434_43443


namespace quadratic_real_roots_m_range_l434_43470

/-- 
Given a quadratic equation x^2 - 2x + m = 0, if it has real roots, 
then m ≤ 1.
-/
theorem quadratic_real_roots_m_range (m : ℝ) : 
  (∃ x : ℝ, x^2 - 2*x + m = 0) → m ≤ 1 := by
  sorry

end quadratic_real_roots_m_range_l434_43470


namespace quadratic_maximum_l434_43480

theorem quadratic_maximum : 
  (∃ (p : ℝ), -3 * p^2 + 18 * p + 24 = 51) ∧ 
  (∀ (p : ℝ), -3 * p^2 + 18 * p + 24 ≤ 51) := by
  sorry

end quadratic_maximum_l434_43480


namespace rectangle_largest_side_l434_43440

/-- Given a rectangle with perimeter 240 feet and area equal to fifteen times its perimeter,
    prove that the length of its largest side is 60 feet. -/
theorem rectangle_largest_side (l w : ℝ) : 
  l > 0 ∧ w > 0 ∧                   -- positive dimensions
  2 * (l + w) = 240 ∧               -- perimeter is 240 feet
  l * w = 15 * 240 →                -- area is fifteen times perimeter
  max l w = 60 := by sorry

end rectangle_largest_side_l434_43440


namespace smallest_m_for_identical_digits_l434_43493

theorem smallest_m_for_identical_digits : ∃ (n : ℕ), 
  (∀ (m : ℕ), m < 671 → 
    ¬∃ (k : ℕ), (2015^(3*m+1) - 2015^(6*k+2)) % 10^2014 = 0 ∧ 2015^(3*m+1) < 2015^(6*k+2)) ∧
  ∃ (n : ℕ), (2015^(3*671+1) - 2015^(6*n+2)) % 10^2014 = 0 ∧ 2015^(3*671+1) < 2015^(6*n+2) :=
sorry

end smallest_m_for_identical_digits_l434_43493


namespace jessica_chocolate_bar_cost_l434_43499

/-- Represents Jessica's purchase --/
structure Purchase where
  total_cost : ℕ
  gummy_bear_packs : ℕ
  chocolate_chip_bags : ℕ
  gummy_bear_cost : ℕ
  chocolate_chip_cost : ℕ

/-- Calculates the cost of chocolate bars in Jessica's purchase --/
def chocolate_bar_cost (p : Purchase) : ℕ :=
  p.total_cost - (p.gummy_bear_packs * p.gummy_bear_cost + p.chocolate_chip_bags * p.chocolate_chip_cost)

/-- Theorem stating that the cost of chocolate bars in Jessica's purchase is $30 --/
theorem jessica_chocolate_bar_cost :
  let p : Purchase := {
    total_cost := 150,
    gummy_bear_packs := 10,
    chocolate_chip_bags := 20,
    gummy_bear_cost := 2,
    chocolate_chip_cost := 5
  }
  chocolate_bar_cost p = 30 := by
  sorry


end jessica_chocolate_bar_cost_l434_43499


namespace xy_problem_l434_43439

theorem xy_problem (x y : ℝ) (h1 : x + y = 7) (h2 : x * y = 6) : 
  ((x - y)^2 = 25) ∧ (x^3 * y + x * y^3 = 222) := by
  sorry

end xy_problem_l434_43439


namespace p_iff_q_l434_43417

theorem p_iff_q : ∀ x : ℝ, (x > 1 ∨ x < -1) ↔ |x + 1| + |x - 1| > 2 := by
  sorry

end p_iff_q_l434_43417


namespace bus_children_difference_l434_43424

theorem bus_children_difference (initial : ℕ) (got_off : ℕ) (final : ℕ) :
  initial = 5 → got_off = 63 → final = 14 →
  ∃ (got_on : ℕ), got_on - got_off = 9 ∧ initial - got_off + got_on = final :=
by sorry

end bus_children_difference_l434_43424


namespace zachary_cans_l434_43408

def can_sequence (n : ℕ) : ℕ := 4 + 5 * (n - 1)

theorem zachary_cans : can_sequence 7 = 34 := by
  sorry

end zachary_cans_l434_43408


namespace max_value_constraint_l434_43492

theorem max_value_constraint (a b c : ℝ) (h : 9 * a^2 + 4 * b^2 + 25 * c^2 = 1) :
  (6 * a + 3 * b + 10 * c) ≤ Real.sqrt 41 / 2 ∧
  ∃ a₀ b₀ c₀ : ℝ, 9 * a₀^2 + 4 * b₀^2 + 25 * c₀^2 = 1 ∧ 
    6 * a₀ + 3 * b₀ + 10 * c₀ = Real.sqrt 41 / 2 :=
by sorry

end max_value_constraint_l434_43492


namespace quadratic_inequality_integer_set_l434_43462

theorem quadratic_inequality_integer_set :
  {x : ℤ | x^2 - 3*x - 4 < 0} = {0, 1, 2, 3} := by sorry

end quadratic_inequality_integer_set_l434_43462


namespace triangle_inequality_sum_l434_43442

theorem triangle_inequality_sum (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b) : 
  (a^2 + 2*b*c)/(b^2 + c^2) + (b^2 + 2*a*c)/(c^2 + a^2) + (c^2 + 2*a*b)/(a^2 + b^2) > 3 := by
  sorry

end triangle_inequality_sum_l434_43442


namespace smallest_n_with_seven_in_squares_l434_43478

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

end smallest_n_with_seven_in_squares_l434_43478


namespace cards_given_to_jeff_l434_43487

/-- Proves that the number of cards Nell gave to Jeff is equal to the difference between her initial number of cards and the number of cards she has left. -/
theorem cards_given_to_jeff (initial_cards : ℕ) (remaining_cards : ℕ) 
  (h1 : initial_cards = 304)
  (h2 : remaining_cards = 276)
  (h3 : initial_cards ≥ remaining_cards) :
  initial_cards - remaining_cards = 28 := by
  sorry

end cards_given_to_jeff_l434_43487


namespace lines_neither_perpendicular_nor_parallel_l434_43431

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Line → Prop)
variable (parallel : Line → Line → Prop)
variable (perpendicularPlanes : Plane → Plane → Prop)
variable (contained : Line → Plane → Prop)
variable (intersect : Plane → Plane → Line → Prop)

-- State the theorem
theorem lines_neither_perpendicular_nor_parallel
  (m n l : Line) (α β : Plane)
  (h1 : contained m α)
  (h2 : contained n β)
  (h3 : perpendicularPlanes α β)
  (h4 : intersect α β l)
  (h5 : ¬ perpendicular m l ∧ ¬ parallel m l)
  (h6 : ¬ perpendicular n l ∧ ¬ parallel n l) :
  ¬ perpendicular m n ∧ ¬ parallel m n :=
by
  sorry

end lines_neither_perpendicular_nor_parallel_l434_43431


namespace a6_is_2_in_factorial_base_of_1735_l434_43453

def factorial : ℕ → ℕ
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def factorial_base_coefficient (n : ℕ) (k : ℕ) : ℕ :=
  (n / factorial k) % (k + 1)

theorem a6_is_2_in_factorial_base_of_1735 :
  factorial_base_coefficient 1735 6 = 2 := by sorry

end a6_is_2_in_factorial_base_of_1735_l434_43453


namespace book_arrangement_theorem_l434_43409

/-- The number of ways to arrange books on a shelf -/
def arrange_books (num_math_books : Nat) (num_history_books : Nat) : Nat :=
  if num_math_books ≥ 2 then
    num_math_books * (num_math_books - 1) * Nat.factorial (num_math_books + num_history_books - 2)
  else
    0

/-- Theorem: The number of ways to arrange 3 math books and 5 history books with math books on both ends is 4320 -/
theorem book_arrangement_theorem :
  arrange_books 3 5 = 4320 := by
  sorry

end book_arrangement_theorem_l434_43409


namespace blue_cars_most_l434_43412

def total_cars : ℕ := 24

def red_cars : ℕ := total_cars / 4

def blue_cars : ℕ := red_cars + 6

def yellow_cars : ℕ := total_cars - (red_cars + blue_cars)

theorem blue_cars_most : blue_cars > red_cars ∧ blue_cars > yellow_cars := by
  sorry

end blue_cars_most_l434_43412


namespace euclid_middle_school_contest_l434_43474

/-- The number of distinct students preparing for the math contest at Euclid Middle School -/
def total_students (euler_students fibonacci_students gauss_students overlap : ℕ) : ℕ :=
  euler_students + fibonacci_students + gauss_students - overlap

theorem euclid_middle_school_contest :
  let euler_students := 12
  let fibonacci_students := 10
  let gauss_students := 11
  let overlap := 3
  total_students euler_students fibonacci_students gauss_students overlap = 27 := by
  sorry

#eval total_students 12 10 11 3

end euclid_middle_school_contest_l434_43474


namespace maximum_marks_calculation_l434_43420

theorem maximum_marks_calculation (percentage : ℝ) (received_marks : ℝ) (max_marks : ℝ) : 
  percentage = 80 → received_marks = 240 → percentage / 100 * max_marks = received_marks → max_marks = 300 := by
  sorry

end maximum_marks_calculation_l434_43420


namespace expression_simplification_l434_43483

theorem expression_simplification (a : ℝ) 
  (h1 : a ≠ 2) (h2 : a ≠ -2) (h3 : a ≠ 3) :
  ((a + 3) / (a^2 - 4) - a / (a^2 - a - 6)) / ((2*a - 9) / (5*a - 10)) = 
  5 / (a^2 - a - 6) := by
  sorry

-- Verifying the result for a = 5
example : 
  let a : ℝ := 5
  ((a + 3) / (a^2 - 4) - a / (a^2 - a - 6)) / ((2*a - 9) / (5*a - 10)) = 
  5 / 14 := by
  sorry

end expression_simplification_l434_43483


namespace least_clock_equivalent_l434_43437

def clock_equivalent (n : ℕ) : Prop :=
  n > 5 ∧ (n^2 - n) % 12 = 0

theorem least_clock_equivalent : ∃ (n : ℕ), clock_equivalent n ∧ ∀ m, m < n → ¬ clock_equivalent m :=
  sorry

end least_clock_equivalent_l434_43437


namespace probability_VIP_ticket_specific_l434_43494

/-- The probability of drawing a VIP ticket from a set of tickets -/
def probability_VIP_ticket (num_VIP : ℕ) (num_regular : ℕ) : ℚ :=
  num_VIP / (num_VIP + num_regular)

/-- Theorem: The probability of drawing a VIP ticket from a set of 1 VIP ticket and 2 regular tickets is 1/3 -/
theorem probability_VIP_ticket_specific : probability_VIP_ticket 1 2 = 1 / 3 := by
  sorry

end probability_VIP_ticket_specific_l434_43494


namespace smallest_number_negative_l434_43464

theorem smallest_number_negative (a : ℝ) :
  (∀ x : ℝ, min (2^(x-1) - 3^(4-x) + a) (a + 5 - x^3 - 2*x) < 0) ↔ a < -7 :=
by sorry

end smallest_number_negative_l434_43464


namespace circle_center_correct_l434_43425

/-- Definition of the circle C -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 4*y = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (-1, 2)

/-- Theorem stating that circle_center is the center of the circle defined by circle_equation -/
theorem circle_center_correct :
  ∀ (x y : ℝ), circle_equation x y ↔ (x - circle_center.1)^2 + (y - circle_center.2)^2 = 5 :=
by sorry

end circle_center_correct_l434_43425


namespace bob_distance_when_met_l434_43415

/-- The distance between X and Y in miles -/
def total_distance : ℝ := 60

/-- Yolanda's walking rate in miles per hour -/
def yolanda_rate : ℝ := 5

/-- Bob's walking rate in miles per hour -/
def bob_rate : ℝ := 6

/-- The time difference in hours between Yolanda's and Bob's start -/
def time_difference : ℝ := 1

/-- The theorem stating that Bob walked 30 miles when they met -/
theorem bob_distance_when_met : 
  ∃ (t : ℝ), 
    t > 0 ∧ 
    yolanda_rate * (t + time_difference) + bob_rate * t = total_distance ∧ 
    bob_rate * t = 30 := by
  sorry

end bob_distance_when_met_l434_43415


namespace total_spent_l434_43458

def weekend_expenses (adidas nike skechers clothes : ℕ) : Prop :=
  nike = 3 * adidas ∧
  adidas = skechers / 5 ∧
  adidas = 600 ∧
  clothes = 2600

theorem total_spent (adidas nike skechers clothes : ℕ) :
  weekend_expenses adidas nike skechers clothes →
  adidas + nike + skechers + clothes = 8000 :=
by sorry

end total_spent_l434_43458


namespace rectangle_triangle_equal_area_l434_43472

/-- Given a rectangle with perimeter 60 and a triangle with height 60, 
    if their areas are equal, then the base of the triangle is 20/3 -/
theorem rectangle_triangle_equal_area (rect_width rect_height tri_base : ℝ) : 
  rect_width > 0 → 
  rect_height > 0 → 
  tri_base > 0 → 
  rect_width + rect_height = 30 → 
  rect_width * rect_height = 30 * tri_base → 
  tri_base = 20 / 3 := by
sorry

end rectangle_triangle_equal_area_l434_43472


namespace total_albums_l434_43427

/-- The number of albums each person has -/
structure Albums where
  adele : ℕ
  bridget : ℕ
  katrina : ℕ
  miriam : ℕ
  carlos : ℕ

/-- The conditions given in the problem -/
def album_conditions (a : Albums) : Prop :=
  a.adele = 30 ∧
  a.bridget = a.adele - 15 ∧
  a.katrina = 6 * a.bridget ∧
  a.miriam = 5 * a.katrina ∧
  a.carlos = 3 * a.miriam

/-- The theorem to prove -/
theorem total_albums (a : Albums) (h : album_conditions a) :
  a.adele + a.bridget + a.katrina + a.miriam + a.carlos = 1935 := by
  sorry

end total_albums_l434_43427
