import Mathlib

namespace NUMINAMATH_CALUDE_max_vector_difference_value_l3614_361438

-- Define the vector space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]

-- Define the theorem
theorem max_vector_difference_value (a b : V) (ha : ‖a‖ = 2) (hb : ‖b‖ = 1) :
  ∃ (c : V), ∀ (x : V), ‖a - b‖ ≤ ‖x‖ ∧ ‖x‖ = 3 :=
sorry

end NUMINAMATH_CALUDE_max_vector_difference_value_l3614_361438


namespace NUMINAMATH_CALUDE_square_difference_l3614_361475

theorem square_difference (m n : ℕ+) 
  (h : (2001 : ℕ) * m ^ 2 + m = (2002 : ℕ) * n ^ 2 + n) : 
  ∃ k : ℕ, m - n = k ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l3614_361475


namespace NUMINAMATH_CALUDE_triangle_inequality_l3614_361410

/-- Given a triangle with circumradius R, inradius r, side lengths a, b, c, and semiperimeter p,
    prove that 20Rr - 4r^2 ≤ ab + bc + ca ≤ 4(R + r)^2 -/
theorem triangle_inequality (R r a b c p : ℝ) (hR : R > 0) (hr : r > 0)
    (ha : a > 0) (hb : b > 0) (hc : c > 0) (hp : p = (a + b + c) / 2)
    (hcirc : R = a * b * c / (4 * p * r)) (hinr : r = p * (p - a) * (p - b) * (p - c) / (a * b * c)) :
    20 * R * r - 4 * r^2 ≤ a * b + b * c + c * a ∧ a * b + b * c + c * a ≤ 4 * (R + r)^2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3614_361410


namespace NUMINAMATH_CALUDE_m_range_not_p_l3614_361456

/-- Proposition p: For all x in [-1,2], m is less than or equal to x² -/
def p (m : ℝ) : Prop :=
  ∀ x : ℝ, x ∈ Set.Icc (-1) 2 → m ≤ x^2

/-- Proposition q: For all x in ℝ, x² + mx + l is greater than 0 -/
def q (m l : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + m*x + l > 0

/-- The range of m when both p and q are true -/
theorem m_range (m l : ℝ) (h_p : p m) (h_q : q m l) :
  m ∈ Set.Ioo (-2) 1 ∨ m = 1 := by
  sorry

/-- The negation of proposition p -/
theorem not_p (m : ℝ) : ¬(p m) ↔ ∃ x : ℝ, x ∈ Set.Icc (-1) 2 ∧ m > x^2 := by
  sorry

end NUMINAMATH_CALUDE_m_range_not_p_l3614_361456


namespace NUMINAMATH_CALUDE_xiao_hong_pen_purchase_l3614_361448

theorem xiao_hong_pen_purchase (total_money : ℝ) (pen_cost : ℝ) (notebook_cost : ℝ) 
  (notebooks_bought : ℕ) (h1 : total_money = 18) (h2 : pen_cost = 3) 
  (h3 : notebook_cost = 3.6) (h4 : notebooks_bought = 2) :
  ∃ (pens : ℕ), pens ∈ ({1, 2, 3} : Set ℕ) ∧ 
  (notebooks_bought : ℝ) * notebook_cost + (pens : ℝ) * pen_cost ≤ total_money :=
sorry

end NUMINAMATH_CALUDE_xiao_hong_pen_purchase_l3614_361448


namespace NUMINAMATH_CALUDE_vector_perpendicular_l3614_361468

/-- Given plane vectors a and b, prove that (a - b) is perpendicular to b -/
theorem vector_perpendicular (a b : ℝ × ℝ) (ha : a = (2, 0)) (hb : b = (1, 1)) :
  (a - b) • b = 0 := by
  sorry

end NUMINAMATH_CALUDE_vector_perpendicular_l3614_361468


namespace NUMINAMATH_CALUDE_milk_students_l3614_361455

theorem milk_students (juice_students : ℕ) (juice_angle : ℝ) (total_angle : ℝ) :
  juice_students = 80 →
  juice_angle = 90 →
  total_angle = 360 →
  (juice_angle / total_angle) * (juice_students + (total_angle - juice_angle) / juice_angle * juice_students) = 240 :=
by sorry

end NUMINAMATH_CALUDE_milk_students_l3614_361455


namespace NUMINAMATH_CALUDE_correct_average_marks_l3614_361422

theorem correct_average_marks (n : ℕ) (initial_avg : ℚ) (wrong_mark correct_mark : ℚ) :
  n = 30 →
  initial_avg = 100 →
  wrong_mark = 70 →
  correct_mark = 10 →
  (n * initial_avg - (wrong_mark - correct_mark)) / n = 98 := by
  sorry

end NUMINAMATH_CALUDE_correct_average_marks_l3614_361422


namespace NUMINAMATH_CALUDE_insurance_premium_calculation_l3614_361478

/-- Calculates the new insurance premium after accidents and tickets. -/
theorem insurance_premium_calculation
  (initial_premium : ℝ)
  (accident_increase_percent : ℝ)
  (ticket_increase : ℝ)
  (num_accidents : ℕ)
  (num_tickets : ℕ)
  (h1 : initial_premium = 50)
  (h2 : accident_increase_percent = 0.1)
  (h3 : ticket_increase = 5)
  (h4 : num_accidents = 1)
  (h5 : num_tickets = 3) :
  initial_premium * (1 + num_accidents * accident_increase_percent) + num_tickets * ticket_increase = 70 :=
by sorry


end NUMINAMATH_CALUDE_insurance_premium_calculation_l3614_361478


namespace NUMINAMATH_CALUDE_three_times_first_minus_second_l3614_361418

theorem three_times_first_minus_second (x y : ℕ) : 
  x + y = 48 → y = 17 → 3 * x - y = 76 := by
  sorry

end NUMINAMATH_CALUDE_three_times_first_minus_second_l3614_361418


namespace NUMINAMATH_CALUDE_cone_base_radius_l3614_361428

/-- Given a cone with surface area 15π cm² and lateral surface that unfolds into a semicircle,
    prove that the radius of its base is √5 cm. -/
theorem cone_base_radius (surface_area : ℝ) (r : ℝ) :
  surface_area = 15 * Real.pi ∧
  (∃ l : ℝ, π * l = 2 * π * r ∧ surface_area = π * r^2 + π * r * l) →
  r = Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_cone_base_radius_l3614_361428


namespace NUMINAMATH_CALUDE_candy_division_l3614_361433

theorem candy_division (p q r : ℕ) (h_pos_p : p > 0) (h_pos_q : q > 0) (h_pos_r : r > 0)
  (h_order : p < q ∧ q < r) (h_a : 20 = 3 * r - 2 * p) (h_b : 10 = r - p)
  (h_c : 9 = 3 * q - 3 * p) (h_c_sum : 3 * q = 18) :
  p = 3 ∧ q = 6 ∧ r = 13 := by sorry

end NUMINAMATH_CALUDE_candy_division_l3614_361433


namespace NUMINAMATH_CALUDE_santana_presents_difference_l3614_361403

/-- The number of presents Santana buys for her siblings in a year -/
theorem santana_presents_difference : 
  let total_siblings : ℕ := 10
  let march_birthdays : ℕ := 4
  let may_birthdays : ℕ := 1
  let june_birthdays : ℕ := 1
  let october_birthdays : ℕ := 1
  let november_birthdays : ℕ := 1
  let december_birthdays : ℕ := 2
  let first_half_presents := march_birthdays + may_birthdays + june_birthdays
  let second_half_presents := october_birthdays + november_birthdays + december_birthdays + total_siblings * 2
  (second_half_presents - first_half_presents) = 18 := by
  sorry

end NUMINAMATH_CALUDE_santana_presents_difference_l3614_361403


namespace NUMINAMATH_CALUDE_tiffany_cans_l3614_361474

theorem tiffany_cans (bags_monday : ℕ) : bags_monday = 12 :=
  by
    have h1 : bags_monday + 12 = 2 * bags_monday := by sorry
    -- The number of bags on Tuesday (bags_monday + 12) is double the number of bags on Monday (2 * bags_monday)
    sorry

end NUMINAMATH_CALUDE_tiffany_cans_l3614_361474


namespace NUMINAMATH_CALUDE_at_least_one_geq_two_l3614_361401

theorem at_least_one_geq_two (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + 1 / y ≥ 2) ∨ (y + 1 / z ≥ 2) ∨ (z + 1 / x ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_geq_two_l3614_361401


namespace NUMINAMATH_CALUDE_parallelogram_area_l3614_361404

theorem parallelogram_area (α : ℝ) (a b : ℝ) (h1 : α = 150) (h2 : a = 9) (h3 : b = 12) :
  let height := a * Real.sqrt 3 / 2
  b * height = 54 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_area_l3614_361404


namespace NUMINAMATH_CALUDE_pizza_theorem_l3614_361451

def pizza_problem (total_slices pepperoni_slices mushroom_slices olive_slices : ℕ) : Prop :=
  ∃ (n a b c : ℕ),
    -- Total slices
    total_slices = 24 ∧
    -- Slices with each topping
    pepperoni_slices = 12 ∧
    mushroom_slices = 14 ∧
    olive_slices = 16 ∧
    -- Every slice has at least one topping
    (12 - n) + (14 - n) + (16 - n) + a + b + c + n = total_slices ∧
    -- Venn diagram constraint
    42 - 3*n - 2*(a + b + c) + a + b + c + n = total_slices ∧
    -- Number of slices with all three toppings
    n = 2

theorem pizza_theorem :
  ∀ (total_slices pepperoni_slices mushroom_slices olive_slices : ℕ),
    pizza_problem total_slices pepperoni_slices mushroom_slices olive_slices :=
by
  sorry

end NUMINAMATH_CALUDE_pizza_theorem_l3614_361451


namespace NUMINAMATH_CALUDE_expression_evaluation_l3614_361482

theorem expression_evaluation : 3^(1^(2^3)) + ((3^1)^2)^2 = 84 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3614_361482


namespace NUMINAMATH_CALUDE_second_term_is_five_l3614_361498

-- Define an arithmetic sequence
def arithmetic_sequence (a : ℝ) (d : ℝ) : ℕ → ℝ
  | 0 => a
  | n + 1 => arithmetic_sequence a d n + d

-- Theorem statement
theorem second_term_is_five
  (a d : ℝ)
  (h : arithmetic_sequence a d 0 + arithmetic_sequence a d 2 = 10) :
  arithmetic_sequence a d 1 = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_second_term_is_five_l3614_361498


namespace NUMINAMATH_CALUDE_flour_recipe_total_l3614_361465

/-- The amount of flour required for Mary's cake recipe -/
def flour_recipe (flour_added flour_needed : ℕ) : ℕ :=
  flour_added + flour_needed

/-- Theorem stating that the total amount of flour required for the recipe is 10 cups -/
theorem flour_recipe_total : flour_recipe 6 4 = 10 := by
  sorry

end NUMINAMATH_CALUDE_flour_recipe_total_l3614_361465


namespace NUMINAMATH_CALUDE_die_throws_probability_l3614_361488

/-- The probability of rolling a number greater than 4 on a single die throw -/
def prob_high : ℚ := 1/3

/-- The probability of rolling a number less than or equal to 4 on a single die throw -/
def prob_low : ℚ := 2/3

/-- The probability of getting at least two numbers greater than 4 in two die throws -/
def prob_at_least_two_high : ℚ := prob_high * prob_high + 2 * prob_high * prob_low

theorem die_throws_probability :
  prob_at_least_two_high = 5/9 := by sorry

end NUMINAMATH_CALUDE_die_throws_probability_l3614_361488


namespace NUMINAMATH_CALUDE_hostel_stay_duration_l3614_361436

/-- Cost structure for a student youth hostel stay -/
structure CostStructure where
  first_week_rate : ℝ
  additional_day_rate : ℝ

/-- Calculate the number of days stayed given the cost structure and total cost -/
def days_stayed (cs : CostStructure) (total_cost : ℝ) : ℕ :=
  sorry

/-- Theorem stating that for the given cost structure and total cost, the stay is 23 days -/
theorem hostel_stay_duration :
  let cs : CostStructure := { first_week_rate := 18, additional_day_rate := 14 }
  let total_cost : ℝ := 350
  days_stayed cs total_cost = 23 := by
  sorry

end NUMINAMATH_CALUDE_hostel_stay_duration_l3614_361436


namespace NUMINAMATH_CALUDE_complex_equation_unit_modulus_l3614_361458

theorem complex_equation_unit_modulus (z : ℂ) (h : 11 * z^10 + 10 * Complex.I * z^9 + 10 * Complex.I * z - 11 = 0) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_unit_modulus_l3614_361458


namespace NUMINAMATH_CALUDE_quadratic_roots_product_l3614_361435

theorem quadratic_roots_product (c d : ℝ) : 
  (3 * c^2 + 9 * c - 21 = 0) → 
  (3 * d^2 + 9 * d - 21 = 0) → 
  (3 * c - 4) * (6 * d - 8) = 14 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_product_l3614_361435


namespace NUMINAMATH_CALUDE_candy_necklaces_remaining_l3614_361416

/-- Proves that given 9 packs of candy necklaces with 8 necklaces in each pack,
    if 4 packs are opened, then at least 40 candy necklaces remain unopened. -/
theorem candy_necklaces_remaining (total_packs : ℕ) (necklaces_per_pack : ℕ) (opened_packs : ℕ) :
  total_packs = 9 →
  necklaces_per_pack = 8 →
  opened_packs = 4 →
  (total_packs - opened_packs) * necklaces_per_pack ≥ 40 := by
  sorry

end NUMINAMATH_CALUDE_candy_necklaces_remaining_l3614_361416


namespace NUMINAMATH_CALUDE_sin_sum_of_roots_l3614_361446

theorem sin_sum_of_roots (a b c : ℝ) (α β : ℝ) : 
  (∀ x, a * Real.cos x + b * Real.sin x + c = 0 ↔ x = α ∨ x = β) →
  0 < α → α < π →
  0 < β → β < π →
  α ≠ β →
  Real.sin (α + β) = (2 * a * b) / (a^2 + b^2) := by
sorry

end NUMINAMATH_CALUDE_sin_sum_of_roots_l3614_361446


namespace NUMINAMATH_CALUDE_one_millionth_digit_of_3_div_41_l3614_361489

/-- The decimal representation of a rational number -/
def decimal_representation (q : ℚ) : ℕ → ℕ := sorry

/-- The nth digit after the decimal point in the decimal representation of q -/
def nth_digit_after_decimal (q : ℚ) (n : ℕ) : ℕ := 
  decimal_representation q n

/-- The one-millionth digit after the decimal point in 3/41 is 7 -/
theorem one_millionth_digit_of_3_div_41 : 
  nth_digit_after_decimal (3/41) 1000000 = 7 := by sorry

end NUMINAMATH_CALUDE_one_millionth_digit_of_3_div_41_l3614_361489


namespace NUMINAMATH_CALUDE_TUVW_product_l3614_361415

def letter_value (c : Char) : ℕ :=
  c.toNat - 'A'.toNat + 1

theorem TUVW_product : 
  (letter_value 'T') * (letter_value 'U') * (letter_value 'V') * (letter_value 'W') = 
  2^3 * 3 * 5 * 7 * 11 * 23 := by
  sorry

end NUMINAMATH_CALUDE_TUVW_product_l3614_361415


namespace NUMINAMATH_CALUDE_point_second_quadrant_sum_distances_l3614_361457

/-- A point in the second quadrant with coordinates (2a, 1-3a) whose sum of distances to x and y axes is 6 has a = -1 -/
theorem point_second_quadrant_sum_distances (a : ℝ) : 
  (2 * a < 0) →  -- Point is in second quadrant (x-coordinate negative)
  (1 - 3 * a > 0) →  -- Point is in second quadrant (y-coordinate positive)
  (abs (2 * a) + abs (1 - 3 * a) = 6) →  -- Sum of distances to axes is 6
  a = -1 := by
sorry

end NUMINAMATH_CALUDE_point_second_quadrant_sum_distances_l3614_361457


namespace NUMINAMATH_CALUDE_bottle_cost_difference_l3614_361454

/-- Represents a bottle of capsules -/
structure Bottle where
  capsules : ℕ
  cost : ℚ

/-- Calculate the cost per capsule for a given bottle -/
def costPerCapsule (b : Bottle) : ℚ :=
  b.cost / b.capsules

/-- The difference in cost per capsule between two bottles -/
def costDifference (b1 b2 : Bottle) : ℚ :=
  costPerCapsule b1 - costPerCapsule b2

theorem bottle_cost_difference :
  let bottleR : Bottle := { capsules := 250, cost := 25/4 }
  let bottleT : Bottle := { capsules := 100, cost := 3 }
  costDifference bottleT bottleR = 1/200 := by
sorry

end NUMINAMATH_CALUDE_bottle_cost_difference_l3614_361454


namespace NUMINAMATH_CALUDE_range_of_a_l3614_361445

-- Define propositions p and q
def p (a : ℝ) : Prop := ∀ x, x^2 - (a-1)*x + 1 > 0

def q (a : ℝ) : Prop := ∀ x y, x < y → (a+1)^x < (a+1)^y

-- Define the theorem
theorem range_of_a (a : ℝ) : 
  (¬(p a ∧ q a) ∧ (p a ∨ q a)) → 
  ((-1 < a ∧ a ≤ 0) ∨ a ≥ 3) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3614_361445


namespace NUMINAMATH_CALUDE_additional_rate_calculation_l3614_361441

/-- Telephone company charging model -/
structure TelephoneCharge where
  initial_rate : ℚ  -- Rate for the first 1/5 minute in cents
  additional_rate : ℚ  -- Rate for each additional 1/5 minute in cents

/-- Calculate the total charge for a given duration -/
def total_charge (model : TelephoneCharge) (duration : ℚ) : ℚ :=
  model.initial_rate + (duration * 5 - 1) * model.additional_rate

theorem additional_rate_calculation (model : TelephoneCharge) 
  (h1 : model.initial_rate = 310/100)  -- 3.10 cents for the first 1/5 minute
  (h2 : total_charge model (8 : ℚ) = 1870/100)  -- 18.70 cents for 8 minutes
  : model.additional_rate = 40/100 := by
  sorry

end NUMINAMATH_CALUDE_additional_rate_calculation_l3614_361441


namespace NUMINAMATH_CALUDE_container_capacity_l3614_361443

theorem container_capacity (container_capacity : ℝ) 
  (h1 : 8 = 0.2 * container_capacity) 
  (num_containers : ℕ := 40) : 
  num_containers * container_capacity = 1600 := by
sorry

end NUMINAMATH_CALUDE_container_capacity_l3614_361443


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l3614_361419

-- Define the ellipse
def ellipse (k : ℝ) (x y : ℝ) : Prop :=
  x^2 / (k + 2) + y^2 / (k + 1) = 1

-- Define the foci
structure Foci (k : ℝ) where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ

-- Define the chord AB
structure Chord (k : ℝ) where
  A : ℝ × ℝ
  B : ℝ × ℝ

-- Define the eccentricity
def eccentricity (k : ℝ) : ℝ := sorry

theorem ellipse_eccentricity 
  (k : ℝ) 
  (hk : k > -1) 
  (f : Foci k)
  (c : Chord k)
  (hF₁ : c.A.1 = f.F₁.1 ∧ c.A.2 = f.F₁.2) -- Chord AB passes through F₁
  (hPerimeter : Real.sqrt ((c.A.1 - f.F₂.1)^2 + (c.A.2 - f.F₂.2)^2) + 
                Real.sqrt ((c.B.1 - f.F₂.1)^2 + (c.B.2 - f.F₂.2)^2) + 
                Real.sqrt ((c.A.1 - c.B.1)^2 + (c.A.2 - c.B.2)^2) = 8) -- Perimeter of ABF₂ is 8
  : eccentricity k = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ellipse_eccentricity_l3614_361419


namespace NUMINAMATH_CALUDE_total_components_is_900_l3614_361460

/-- Represents the total number of components --/
def total_components : ℕ := 900

/-- Represents the number of type B components --/
def type_b_components : ℕ := 300

/-- Represents the number of type C components --/
def type_c_components : ℕ := 200

/-- Represents the sample size --/
def sample_size : ℕ := 45

/-- Represents the number of type A components in the sample --/
def sample_type_a : ℕ := 20

/-- Represents the number of type C components in the sample --/
def sample_type_c : ℕ := 10

/-- Theorem stating that the total number of components is 900 --/
theorem total_components_is_900 :
  total_components = 900 ∧
  type_b_components = 300 ∧
  type_c_components = 200 ∧
  sample_size = 45 ∧
  sample_type_a = 20 ∧
  sample_type_c = 10 ∧
  (sample_type_c : ℚ) / (sample_size : ℚ) = (type_c_components : ℚ) / (total_components : ℚ) :=
by sorry

#check total_components_is_900

end NUMINAMATH_CALUDE_total_components_is_900_l3614_361460


namespace NUMINAMATH_CALUDE_integer_solution_equation_l3614_361427

theorem integer_solution_equation (x y : ℤ) : 
  9 * x + 2 = y * (y + 1) ↔ ∃ k : ℤ, x = k * (k + 1) ∧ y = 3 * k + 1 := by
  sorry

end NUMINAMATH_CALUDE_integer_solution_equation_l3614_361427


namespace NUMINAMATH_CALUDE_phoenix_hiking_problem_l3614_361409

/-- Phoenix's hiking problem -/
theorem phoenix_hiking_problem (a b c d e : ℝ) 
  (h1 : a + b + c = 36)
  (h2 : (b + c + d) / 3 = 16)
  (h3 : c + d + e = 45)
  (h4 : a + d = 31) :
  a + b + c + d + e = 81 := by
  sorry

#check phoenix_hiking_problem

end NUMINAMATH_CALUDE_phoenix_hiking_problem_l3614_361409


namespace NUMINAMATH_CALUDE_cosine_rationality_l3614_361402

theorem cosine_rationality (k : ℤ) (θ : ℝ) 
  (h1 : k ≥ 3)
  (h2 : ∃ q₁ : ℚ, (↑q₁ : ℝ) = Real.cos ((k - 1) * θ))
  (h3 : ∃ q₂ : ℚ, (↑q₂ : ℝ) = Real.cos (k * θ)) :
  ∃ (n : ℕ), n > k ∧ 
    (∃ q₃ : ℚ, (↑q₃ : ℝ) = Real.cos ((n - 1) * θ)) ∧ 
    (∃ q₄ : ℚ, (↑q₄ : ℝ) = Real.cos (n * θ)) := by
  sorry

end NUMINAMATH_CALUDE_cosine_rationality_l3614_361402


namespace NUMINAMATH_CALUDE_problem_solution_l3614_361472

theorem problem_solution : ∃ y : ℕ, (8000 * 6000 : ℕ) = 480 * (10 ^ y) ∧ y = 5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3614_361472


namespace NUMINAMATH_CALUDE_joshua_toy_cars_l3614_361480

theorem joshua_toy_cars (box1 box2 box3 : ℕ) 
  (h1 : box1 = 21) 
  (h2 : box2 = 31) 
  (h3 : box3 = 19) : 
  box1 + box2 + box3 = 71 := by
sorry

end NUMINAMATH_CALUDE_joshua_toy_cars_l3614_361480


namespace NUMINAMATH_CALUDE_lecture_scheduling_l3614_361444

-- Define the number of lecturers
def n : ℕ := 7

-- Theorem statement
theorem lecture_scheduling (n : ℕ) (h : n = 7) : 
  (n! : ℕ) / 2 = 2520 :=
sorry

end NUMINAMATH_CALUDE_lecture_scheduling_l3614_361444


namespace NUMINAMATH_CALUDE_father_and_xiaolin_ages_l3614_361426

theorem father_and_xiaolin_ages :
  ∀ (f x : ℕ),
  f = 11 * x →
  f + 7 = 4 * (x + 7) →
  f = 33 ∧ x = 3 := by
sorry

end NUMINAMATH_CALUDE_father_and_xiaolin_ages_l3614_361426


namespace NUMINAMATH_CALUDE_fraction_to_zero_power_is_one_l3614_361497

theorem fraction_to_zero_power_is_one (a b : ℤ) (hb : b ≠ 0) : (a / b : ℚ) ^ (0 : ℕ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_zero_power_is_one_l3614_361497


namespace NUMINAMATH_CALUDE_cube_difference_simplification_l3614_361467

theorem cube_difference_simplification (a b : ℝ) (ha_pos : a > 0) (hb_neg : b < 0)
  (ha_sq : a^2 = 9/25) (hb_sq : b^2 = (3 + Real.sqrt 2)^2 / 14) :
  (a - b)^3 = 88 * Real.sqrt 2 / 12750 := by
  sorry

end NUMINAMATH_CALUDE_cube_difference_simplification_l3614_361467


namespace NUMINAMATH_CALUDE_existence_of_plane_only_properties_l3614_361442

-- Define abstract types for plane and solid geometry
def PlaneGeometry : Type := Unit
def SolidGeometry : Type := Unit

-- Define a property as a function that takes a geometry and returns a proposition
def GeometricProperty : Type := (PlaneGeometry ⊕ SolidGeometry) → Prop

-- Define a function to check if a property holds in plane geometry
def holdsInPlaneGeometry (prop : GeometricProperty) : Prop :=
  prop (Sum.inl ())

-- Define a function to check if a property holds in solid geometry
def holdsInSolidGeometry (prop : GeometricProperty) : Prop :=
  prop (Sum.inr ())

-- State the theorem
theorem existence_of_plane_only_properties :
  ∃ (prop : GeometricProperty),
    holdsInPlaneGeometry prop ∧ ¬holdsInSolidGeometry prop := by
  sorry

-- Examples of properties (these are just placeholders and not actual proofs)
def perpendicularLinesParallel : GeometricProperty := fun _ => True
def uniquePerpendicularLine : GeometricProperty := fun _ => True
def equalSidedQuadrilateralIsRhombus : GeometricProperty := fun _ => True

end NUMINAMATH_CALUDE_existence_of_plane_only_properties_l3614_361442


namespace NUMINAMATH_CALUDE_stock_worth_equation_l3614_361459

/-- Proves that the total worth of stock satisfies the given equation based on the problem conditions --/
theorem stock_worth_equation (W : ℝ) 
  (h1 : 0.25 * W * 0.15 - 0.40 * W * 0.05 + 0.35 * W * 0.10 = 750) : 
  0.0525 * W = 750 := by
  sorry

end NUMINAMATH_CALUDE_stock_worth_equation_l3614_361459


namespace NUMINAMATH_CALUDE_tangent_circle_equations_l3614_361473

def given_circle (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 2*y - 4 = 0

def tangent_line (y : ℝ) : Prop :=
  y = 0

def is_tangent_circles (x1 y1 r1 x2 y2 r2 : ℝ) : Prop :=
  (x1 - x2)^2 + (y1 - y2)^2 = (r1 + r2)^2 ∨ (x1 - x2)^2 + (y1 - y2)^2 = (r1 - r2)^2

def is_tangent_circle_line (x y r : ℝ) : Prop :=
  y = r ∨ y = -r

theorem tangent_circle_equations :
  ∃ (a b c d : ℝ),
    (∀ x y : ℝ, ((x - (2 + 2 * Real.sqrt 10))^2 + (y - 4)^2 = 16 ↔ (x - a)^2 + (y - 4)^2 = 16) ∧
                ((x - (2 - 2 * Real.sqrt 10))^2 + (y - 4)^2 = 16 ↔ (x - b)^2 + (y - 4)^2 = 16) ∧
                ((x - (2 + 2 * Real.sqrt 6))^2 + (y + 4)^2 = 16 ↔ (x - c)^2 + (y + 4)^2 = 16) ∧
                ((x - (2 - 2 * Real.sqrt 6))^2 + (y + 4)^2 = 16 ↔ (x - d)^2 + (y + 4)^2 = 16)) ∧
    (∀ x y : ℝ, given_circle x y →
      (is_tangent_circles x y 3 a 4 4 ∧ is_tangent_circle_line a 4 4) ∨
      (is_tangent_circles x y 3 b 4 4 ∧ is_tangent_circle_line b 4 4) ∨
      (is_tangent_circles x y 3 c (-4) 4 ∧ is_tangent_circle_line c (-4) 4) ∨
      (is_tangent_circles x y 3 d (-4) 4 ∧ is_tangent_circle_line d (-4) 4)) ∧
    (∀ x y : ℝ, tangent_line y →
      ((x - a)^2 + (y - 4)^2 = 16 ∨
       (x - b)^2 + (y - 4)^2 = 16 ∨
       (x - c)^2 + (y + 4)^2 = 16 ∨
       (x - d)^2 + (y + 4)^2 = 16)) := by
  sorry

end NUMINAMATH_CALUDE_tangent_circle_equations_l3614_361473


namespace NUMINAMATH_CALUDE_binomial_plus_four_l3614_361405

theorem binomial_plus_four : (Nat.choose 18 17) + 4 = 22 := by
  sorry

end NUMINAMATH_CALUDE_binomial_plus_four_l3614_361405


namespace NUMINAMATH_CALUDE_points_per_question_l3614_361400

theorem points_per_question (correct_answers : ℕ) (final_score : ℕ) : 
  correct_answers = 5 → final_score = 15 → (final_score / correct_answers : ℚ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_points_per_question_l3614_361400


namespace NUMINAMATH_CALUDE_students_walking_home_l3614_361470

theorem students_walking_home (bus auto bike scooter : ℚ)
  (h_bus : bus = 2/5)
  (h_auto : auto = 1/5)
  (h_bike : bike = 1/10)
  (h_scooter : scooter = 1/10)
  : 1 - (bus + auto + bike + scooter) = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_students_walking_home_l3614_361470


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3614_361483

-- Define the sets M and N
def M : Set ℕ := {0, 1, 3}
def N : Set ℕ := {0, 1, 7}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3614_361483


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l3614_361493

/-- The speed of a boat in still water, given downstream and upstream speeds and current speed. -/
theorem boat_speed_in_still_water 
  (current_speed : ℝ) 
  (downstream_speed : ℝ) 
  (upstream_speed : ℝ) 
  (h1 : current_speed = 17)
  (h2 : downstream_speed = 77)
  (h3 : upstream_speed = 43) :
  ∃ (still_water_speed : ℝ), 
    still_water_speed = 60 ∧ 
    still_water_speed + current_speed = downstream_speed ∧ 
    still_water_speed - current_speed = upstream_speed :=
by sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l3614_361493


namespace NUMINAMATH_CALUDE_algebraic_equality_l3614_361471

theorem algebraic_equality (a b : ℝ) : 
  (2*a^2 - 4*a*b + b^2 = -3*a^2 + 2*a*b - 5*b^2) → 
  (2*a^2 - 4*a*b + b^2 + 3*a^2 - 2*a*b + 5*b^2 = 5*a^2 + 6*b^2 - 6*a*b) := by
  sorry

end NUMINAMATH_CALUDE_algebraic_equality_l3614_361471


namespace NUMINAMATH_CALUDE_solar_panel_flat_fee_l3614_361453

def land_acres : ℕ := 30
def land_cost_per_acre : ℕ := 20
def house_cost : ℕ := 120000
def cow_count : ℕ := 20
def cow_cost_per_unit : ℕ := 1000
def chicken_count : ℕ := 100
def chicken_cost_per_unit : ℕ := 5
def solar_installation_hours : ℕ := 6
def solar_installation_cost_per_hour : ℕ := 100
def total_cost : ℕ := 147700

theorem solar_panel_flat_fee :
  total_cost - (land_acres * land_cost_per_acre + house_cost + 
    cow_count * cow_cost_per_unit + chicken_count * chicken_cost_per_unit + 
    solar_installation_hours * solar_installation_cost_per_hour) = 26000 := by
  sorry

end NUMINAMATH_CALUDE_solar_panel_flat_fee_l3614_361453


namespace NUMINAMATH_CALUDE_distance_sum_theorem_l3614_361462

/-- The curve C in the xy-plane -/
def C (x y : ℝ) : Prop := x^2/9 + y^2 = 1

/-- The line l in the xy-plane -/
def l (x y : ℝ) : Prop := y - x = Real.sqrt 2

/-- The point P -/
def P : ℝ × ℝ := (0, 2)

/-- Points A and B are the intersection points of C and l -/
def intersectionPoints (A B : ℝ × ℝ) : Prop :=
  C A.1 A.2 ∧ l A.1 A.2 ∧ C B.1 B.2 ∧ l B.1 B.2 ∧ A ≠ B

theorem distance_sum_theorem (A B : ℝ × ℝ) (h : intersectionPoints A B) :
  Real.sqrt ((A.1 - P.1)^2 + (A.2 - P.2)^2) +
  Real.sqrt ((B.1 - P.1)^2 + (B.2 - P.2)^2) =
  18 * Real.sqrt 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_sum_theorem_l3614_361462


namespace NUMINAMATH_CALUDE_simplify_square_roots_l3614_361431

theorem simplify_square_roots : Real.sqrt 7 - Real.sqrt 28 + Real.sqrt 63 = 2 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_roots_l3614_361431


namespace NUMINAMATH_CALUDE_distance_to_directrix_l3614_361420

/-- Parabola type representing y² = 2px -/
structure Parabola where
  p : ℝ

/-- Point type representing (x, y) coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- Theorem: Distance from point A to directrix of parabola C -/
theorem distance_to_directrix 
  (C : Parabola) 
  (A : Point) 
  (h1 : A.y^2 = 2 * C.p * A.x) 
  (h2 : A.x = 1) 
  (h3 : A.y = Real.sqrt 5) : 
  A.x + C.p / 2 = 9 / 4 := by
  sorry

#check distance_to_directrix

end NUMINAMATH_CALUDE_distance_to_directrix_l3614_361420


namespace NUMINAMATH_CALUDE_opposite_sqrt_nine_is_negative_three_l3614_361424

theorem opposite_sqrt_nine_is_negative_three :
  -(Real.sqrt 9) = -3 := by
  sorry

end NUMINAMATH_CALUDE_opposite_sqrt_nine_is_negative_three_l3614_361424


namespace NUMINAMATH_CALUDE_pens_left_after_giving_away_l3614_361499

/-- Given that a student's parents bought her 56 pens and she gave 22 pens to her friends,
    prove that the number of pens left for her to use is 34. -/
theorem pens_left_after_giving_away (total_pens : ℕ) (pens_given_away : ℕ) :
  total_pens = 56 → pens_given_away = 22 → total_pens - pens_given_away = 34 := by
  sorry

end NUMINAMATH_CALUDE_pens_left_after_giving_away_l3614_361499


namespace NUMINAMATH_CALUDE_circle_properties_l3614_361491

def circle_equation (x y : ℝ) : ℝ := x^2 + y^2 - 12*x - 12*y - 88

def line_equation (x y : ℝ) : ℝ := x + 3*y + 16

def point_A : ℝ × ℝ := (-6, 10)
def point_B : ℝ × ℝ := (2, -6)

theorem circle_properties :
  (circle_equation point_A.1 point_A.2 = 0) ∧
  (circle_equation point_B.1 point_B.2 = 0) ∧
  (line_equation point_B.1 point_B.2 = 0) ∧
  (∃ (t : ℝ), t ≠ 0 ∧
    (2 * point_B.1 - 12) * 1 + (2 * point_B.2 - 12) * 3 = t * (1^2 + 3^2)) :=
sorry

end NUMINAMATH_CALUDE_circle_properties_l3614_361491


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3614_361432

theorem quadratic_inequality_solution_set (a : ℝ) : 
  (∀ x : ℝ, x^2 - a*x + a > 0) ↔ (0 < a ∧ a < 4) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3614_361432


namespace NUMINAMATH_CALUDE_nested_fraction_equality_l3614_361464

theorem nested_fraction_equality : 2 - (1 / (2 - (1 / (2 + 2)))) = 10 / 7 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_equality_l3614_361464


namespace NUMINAMATH_CALUDE_line_ellipse_intersection_slopes_l3614_361476

theorem line_ellipse_intersection_slopes (m : ℝ) : 
  (∃ x y : ℝ, 4 * x^2 + 25 * y^2 = 100 ∧ y = m * x + 8) ↔ 
  m ≤ -Real.sqrt 2.4 ∨ m ≥ Real.sqrt 2.4 := by
sorry

end NUMINAMATH_CALUDE_line_ellipse_intersection_slopes_l3614_361476


namespace NUMINAMATH_CALUDE_evaluate_expression_l3614_361494

theorem evaluate_expression (y : ℝ) (h : y = -3) : 
  (5 + y * (5 + y) - 5^2) / (y - 5 + y^2) = -26 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3614_361494


namespace NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l3614_361423

theorem tan_alpha_plus_pi_fourth (α : Real) 
  (h : 6 * Real.sin α * Real.cos α = 1 + Real.cos (2 * α)) : 
  Real.tan (α + π/4) = 2 ∨ Real.tan (α + π/4) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l3614_361423


namespace NUMINAMATH_CALUDE_interior_triangle_area_l3614_361429

theorem interior_triangle_area (a b c : ℝ) (ha : a^2 = 49) (hb : b^2 = 64) (hc : c^2 = 225) :
  (1/2 : ℝ) * a * b = 28 :=
sorry

end NUMINAMATH_CALUDE_interior_triangle_area_l3614_361429


namespace NUMINAMATH_CALUDE_trig_problem_l3614_361496

theorem trig_problem (α : Real) 
  (h1 : 0 < α) (h2 : α < Real.pi / 2) (h3 : Real.sin α = 4 / 5) : 
  (Real.sin α ^ 2 + Real.sin (2 * α)) / (Real.cos α ^ 2 + Real.cos (2 * α)) = 20 ∧ 
  Real.tan (α - 5 * Real.pi / 4) = 1 / 7 := by sorry

end NUMINAMATH_CALUDE_trig_problem_l3614_361496


namespace NUMINAMATH_CALUDE_l_shape_area_and_perimeter_l3614_361479

/-- Represents the dimensions of a rectangle -/
structure RectangleDimensions where
  length : Real
  width : Real

/-- Calculates the area of a rectangle -/
def rectangleArea (d : RectangleDimensions) : Real :=
  d.length * d.width

/-- Calculates the perimeter of a rectangle -/
def rectanglePerimeter (d : RectangleDimensions) : Real :=
  2 * (d.length + d.width)

/-- Represents an L-shaped region formed by two rectangles -/
structure LShape where
  rect1 : RectangleDimensions
  rect2 : RectangleDimensions

/-- Calculates the area of an L-shaped region -/
def lShapeArea (l : LShape) : Real :=
  rectangleArea l.rect1 + rectangleArea l.rect2

/-- Calculates the perimeter of an L-shaped region -/
def lShapePerimeter (l : LShape) : Real :=
  rectanglePerimeter l.rect1 + rectanglePerimeter l.rect2 - 2 * l.rect1.length

theorem l_shape_area_and_perimeter :
  let l : LShape := {
    rect1 := { length := 0.5, width := 0.3 },
    rect2 := { length := 0.2, width := 0.5 }
  }
  lShapeArea l = 0.25 ∧ lShapePerimeter l = 2.0 := by sorry

end NUMINAMATH_CALUDE_l_shape_area_and_perimeter_l3614_361479


namespace NUMINAMATH_CALUDE_total_earnings_theorem_l3614_361469

/-- Represents the different car models --/
inductive CarModel
| A
| B
| C
| D

/-- Represents the different services offered --/
inductive Service
| OilChange
| Repair
| CarWash
| TireRotation

/-- Returns the price of a service for a given car model --/
def servicePrice (model : CarModel) (service : Service) : ℕ :=
  match model, service with
  | CarModel.A, Service.OilChange => 20
  | CarModel.A, Service.Repair => 30
  | CarModel.A, Service.CarWash => 5
  | CarModel.A, Service.TireRotation => 15
  | CarModel.B, Service.OilChange => 25
  | CarModel.B, Service.Repair => 40
  | CarModel.B, Service.CarWash => 8
  | CarModel.B, Service.TireRotation => 18
  | CarModel.C, Service.OilChange => 30
  | CarModel.C, Service.Repair => 50
  | CarModel.C, Service.CarWash => 10
  | CarModel.C, Service.TireRotation => 20
  | CarModel.D, Service.OilChange => 35
  | CarModel.D, Service.Repair => 60
  | CarModel.D, Service.CarWash => 12
  | CarModel.D, Service.TireRotation => 22

/-- Applies discount if the number of services is 3 or more --/
def applyDiscount (total : ℕ) (numServices : ℕ) : ℕ :=
  if numServices ≥ 3 then
    total - (total * 10 / 100)
  else
    total

/-- Calculates the total price for a car model with given services --/
def totalPrice (model : CarModel) (services : List Service) : ℕ :=
  let total := services.foldl (fun acc service => acc + servicePrice model service) 0
  applyDiscount total services.length

/-- The main theorem to prove --/
theorem total_earnings_theorem :
  let modelA_services := [Service.OilChange, Service.Repair, Service.CarWash]
  let modelB_services := [Service.OilChange, Service.Repair, Service.CarWash, Service.TireRotation]
  let modelC_services := [Service.OilChange, Service.Repair, Service.TireRotation, Service.CarWash]
  let modelD_services := [Service.OilChange, Service.Repair, Service.TireRotation]
  
  5 * (totalPrice CarModel.A modelA_services) +
  3 * (totalPrice CarModel.B modelB_services) +
  2 * (totalPrice CarModel.C modelC_services) +
  4 * (totalPrice CarModel.D modelD_services) = 111240 :=
by sorry


end NUMINAMATH_CALUDE_total_earnings_theorem_l3614_361469


namespace NUMINAMATH_CALUDE_infinitely_many_a_for_perfect_cube_l3614_361490

theorem infinitely_many_a_for_perfect_cube (n : ℕ) : 
  ∃ (f : ℕ → ℤ), Function.Injective f ∧ ∀ (k : ℕ), ∃ (m : ℕ), (n^6 + 3 * (f k) : ℤ) = m^3 := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_a_for_perfect_cube_l3614_361490


namespace NUMINAMATH_CALUDE_painted_faces_count_l3614_361450

/-- Represents a cube with side length n -/
structure Cube (n : ℕ) where
  side_length : ℕ := n

/-- Represents a painted cube -/
structure PaintedCube (n : ℕ) extends Cube n where
  is_painted : Bool := true

/-- Counts the number of smaller cubes with at least two painted faces when a painted cube is cut into unit cubes -/
def count_painted_faces (c : PaintedCube 4) : ℕ :=
  sorry

/-- Theorem stating that a 4x4x4 painted cube cut into 1x1x1 cubes has 32 smaller cubes with at least two painted faces -/
theorem painted_faces_count (c : PaintedCube 4) : count_painted_faces c = 32 :=
  sorry

end NUMINAMATH_CALUDE_painted_faces_count_l3614_361450


namespace NUMINAMATH_CALUDE_two_week_riding_hours_l3614_361434

/-- Represents the number of hours Bethany rides on a given day -/
def daily_riding_hours (day : Nat) : Real :=
  match day % 7 with
  | 1 | 3 | 5 => 1    -- Monday, Wednesday, Friday
  | 2 | 4 => 0.5      -- Tuesday, Thursday
  | 6 => 2            -- Saturday
  | _ => 0            -- Sunday

/-- Calculates the total riding hours over a given number of days -/
def total_riding_hours (days : Nat) : Real :=
  (List.range days).map daily_riding_hours |>.sum

/-- Proves that Bethany rides for 12 hours over a 2-week period -/
theorem two_week_riding_hours :
  total_riding_hours 14 = 12 := by
  sorry

end NUMINAMATH_CALUDE_two_week_riding_hours_l3614_361434


namespace NUMINAMATH_CALUDE_quadratic_equations_solution_l3614_361463

theorem quadratic_equations_solution :
  -- Part 1
  (∀ x, 1969 * x^2 - 1974 * x + 5 = 0 ↔ x = 1 ∨ x = 5/1969) ∧
  -- Part 2
  (∀ a b c x,
    -- Case 1
    (a + b - 2*c = 0 ∧ b + c - 2*a ≠ 0 →
      (a + b - 2*c) * x^2 + (b + c - 2*a) * x + (c + a - 2*b) = 0 ↔
      x = -(c + a - 2*b) / (b + c - 2*a)) ∧
    (a + b - 2*c = 0 ∧ b + c - 2*a = 0 →
      (a + b - 2*c) * x^2 + (b + c - 2*a) * x + (c + a - 2*b) = 0 ↔
      True) ∧
    -- Case 2
    (a + b - 2*c ≠ 0 →
      (a + b - 2*c) * x^2 + (b + c - 2*a) * x + (c + a - 2*b) = 0 ↔
      x = 1 ∨ x = (c + a - 2*b) / (a + b - 2*c))) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solution_l3614_361463


namespace NUMINAMATH_CALUDE_number_puzzle_l3614_361486

theorem number_puzzle : ∃ x : ℝ, 3 * (2 * x + 5) = 129 ∧ x = 19 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l3614_361486


namespace NUMINAMATH_CALUDE_max_blocks_fit_l3614_361492

/-- Represents the dimensions of a rectangular solid -/
structure Dimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a rectangular solid given its dimensions -/
def volume (d : Dimensions) : ℕ :=
  d.length * d.width * d.height

/-- The dimensions of the box -/
def box : Dimensions :=
  { length := 4, width := 3, height := 3 }

/-- The dimensions of a block -/
def block : Dimensions :=
  { length := 2, width := 3, height := 1 }

/-- The maximum number of blocks that can fit in the box -/
def max_blocks : ℕ :=
  volume box / volume block

theorem max_blocks_fit :
  max_blocks = 6 ∧
  (∀ n : ℕ, n > max_blocks → ¬ (n * volume block ≤ volume box)) :=
by sorry

end NUMINAMATH_CALUDE_max_blocks_fit_l3614_361492


namespace NUMINAMATH_CALUDE_solve_for_a_l3614_361412

theorem solve_for_a : ∃ a : ℝ, (2 * 1 - a * (-1) = 3) ∧ a = 1 := by sorry

end NUMINAMATH_CALUDE_solve_for_a_l3614_361412


namespace NUMINAMATH_CALUDE_currency_notes_theorem_l3614_361440

theorem currency_notes_theorem (x y z : ℕ) : 
  x + y + z = 130 →
  95 * x + 45 * y + 20 * z = 7000 →
  75 * x + 25 * y = 4400 := by
sorry

end NUMINAMATH_CALUDE_currency_notes_theorem_l3614_361440


namespace NUMINAMATH_CALUDE_percentage_of_girls_l3614_361487

theorem percentage_of_girls (boys girls : ℕ) (h1 : boys = 300) (h2 : girls = 450) :
  (girls : ℚ) / ((boys : ℚ) + (girls : ℚ)) * 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_girls_l3614_361487


namespace NUMINAMATH_CALUDE_starting_number_proof_l3614_361430

def has_two_fives (n : ℕ) : Prop :=
  (n / 10 = 5 ∧ n % 10 = 5) ∨ (n / 100 = 5 ∧ n % 100 / 10 = 5)

theorem starting_number_proof :
  ∀ (start : ℕ),
    start ≤ 54 →
    (∃! n : ℕ, start ≤ n ∧ n ≤ 50 ∧ has_two_fives n) →
    start = 54 :=
by sorry

end NUMINAMATH_CALUDE_starting_number_proof_l3614_361430


namespace NUMINAMATH_CALUDE_no_valid_arrangement_l3614_361411

/-- Represents a 3x3 grid of natural numbers -/
def Grid := Fin 3 → Fin 3 → ℕ

/-- Checks if a number is prime -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

/-- Checks if two positions in the grid are adjacent -/
def isAdjacent (x1 y1 x2 y2 : Fin 3) : Prop :=
  (x1 = x2 ∧ (y1 = y2 + 1 ∨ y2 = y1 + 1)) ∨
  (y1 = y2 ∧ (x1 = x2 + 1 ∨ x2 = x1 + 1))

/-- Checks if a grid arrangement is valid according to the problem conditions -/
def isValidArrangement (g : Grid) : Prop :=
  (∀ x y : Fin 3, g x y ∈ Finset.range 9) ∧
  (∀ x1 y1 x2 y2 : Fin 3, isAdjacent x1 y1 x2 y2 → isPrime (g x1 y1 + g x2 y2)) ∧
  (∀ n : Fin 9, ∃ x y : Fin 3, g x y = n + 1)

/-- The main theorem stating that no valid arrangement exists -/
theorem no_valid_arrangement : ¬∃ g : Grid, isValidArrangement g := by
  sorry

end NUMINAMATH_CALUDE_no_valid_arrangement_l3614_361411


namespace NUMINAMATH_CALUDE_room_length_is_twenty_l3614_361414

/-- Represents the dimensions and tiling of a rectangular room. -/
structure Room where
  length : ℝ
  breadth : ℝ
  tileSize : ℝ
  blackTileWidth : ℝ
  blueTileCount : ℕ

/-- Theorem stating the length of the room given specific conditions. -/
theorem room_length_is_twenty (r : Room) : 
  r.breadth = 10 ∧ 
  r.tileSize = 2 ∧ 
  r.blackTileWidth = 2 ∧ 
  r.blueTileCount = 16 ∧
  (r.length - 2 * r.blackTileWidth) * (r.breadth - 2 * r.blackTileWidth) * (2/3) = 
    (r.blueTileCount : ℝ) * r.tileSize * r.tileSize →
  r.length = 20 := by
  sorry

#check room_length_is_twenty

end NUMINAMATH_CALUDE_room_length_is_twenty_l3614_361414


namespace NUMINAMATH_CALUDE_acrobat_weight_l3614_361452

/-- Given weights of various objects, prove that an acrobat weighs twice as much as a lamb -/
theorem acrobat_weight (barrel dog acrobat lamb coil : ℝ) 
  (h1 : acrobat + dog = 2 * barrel)
  (h2 : dog = 2 * coil)
  (h3 : lamb + coil = barrel) :
  acrobat = 2 * lamb := by
  sorry

end NUMINAMATH_CALUDE_acrobat_weight_l3614_361452


namespace NUMINAMATH_CALUDE_complement_M_intersect_N_l3614_361481

-- Define the set M
def M : Set ℝ := {x : ℝ | x ≤ Real.sqrt 5}

-- Define the set N
def N : Set ℝ := {1, 2, 3, 4}

-- Theorem statement
theorem complement_M_intersect_N :
  (Set.compl M) ∩ N = {3, 4} := by sorry

end NUMINAMATH_CALUDE_complement_M_intersect_N_l3614_361481


namespace NUMINAMATH_CALUDE_tangent_line_at_2_minus_6_tangent_lines_slope_4_l3614_361466

-- Define the function f(x) = x³ + x - 16
def f (x : ℝ) : ℝ := x^3 + x - 16

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

-- Theorem for the tangent line at (2, -6)
theorem tangent_line_at_2_minus_6 :
  ∃ (m b : ℝ), m * 2 - b = f 2 ∧ 
               m = f' 2 ∧
               ∀ x, m * x - b = 13 * x - 32 :=
sorry

-- Theorem for tangent lines with slope 4
theorem tangent_lines_slope_4 :
  ∃ (x₁ x₂ b₁ b₂ : ℝ), 
    x₁ ≠ x₂ ∧
    f' x₁ = 4 ∧ f' x₂ = 4 ∧
    4 * x₁ - b₁ = f x₁ ∧
    4 * x₂ - b₂ = f x₂ ∧
    (∀ x, 4 * x - b₁ = 4 * x - 18) ∧
    (∀ x, 4 * x - b₂ = 4 * x - 14) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_2_minus_6_tangent_lines_slope_4_l3614_361466


namespace NUMINAMATH_CALUDE_players_both_games_l3614_361439

/-- Given a group of players with the following properties:
  * There are 400 players in total
  * 350 players play outdoor games
  * 110 players play indoor games
  This theorem proves that the number of players who play both indoor and outdoor games is 60. -/
theorem players_both_games (total : ℕ) (outdoor : ℕ) (indoor : ℕ) 
  (h_total : total = 400)
  (h_outdoor : outdoor = 350)
  (h_indoor : indoor = 110) :
  ∃ (both : ℕ), both = outdoor + indoor - total ∧ both = 60 := by
  sorry

end NUMINAMATH_CALUDE_players_both_games_l3614_361439


namespace NUMINAMATH_CALUDE_final_orange_count_l3614_361425

def initial_oranges : ℕ := 150

def sold_to_peter (n : ℕ) : ℕ := n - n * 20 / 100

def sold_to_paula (n : ℕ) : ℕ := n - n * 30 / 100

def give_to_neighbor (n : ℕ) : ℕ := n - 10

def give_to_teacher (n : ℕ) : ℕ := n - 1

theorem final_orange_count :
  give_to_teacher (give_to_neighbor (sold_to_paula (sold_to_peter initial_oranges))) = 73 := by
  sorry

end NUMINAMATH_CALUDE_final_orange_count_l3614_361425


namespace NUMINAMATH_CALUDE_cookie_sugar_measurement_l3614_361413

-- Define the amount of sugar needed
def sugar_needed : ℚ := 15/4

-- Define the capacity of the measuring cup
def cup_capacity : ℚ := 1/3

-- Theorem to prove
theorem cookie_sugar_measurement :
  ⌈sugar_needed / cup_capacity⌉ = 12 := by
  sorry

end NUMINAMATH_CALUDE_cookie_sugar_measurement_l3614_361413


namespace NUMINAMATH_CALUDE_intersection_with_complement_of_reals_l3614_361484

open Set

theorem intersection_with_complement_of_reals (A B : Set ℝ) 
  (hA : A = {x : ℝ | x > 0}) 
  (hB : B = {x : ℝ | x > 1}) : 
  A ∩ (Set.univ \ B) = {x : ℝ | 0 < x ∧ x ≤ 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_with_complement_of_reals_l3614_361484


namespace NUMINAMATH_CALUDE_orangeade_price_day2_l3614_361437

/-- Represents the price and volume of orangeade on a given day -/
structure OrangeadeDay where
  juice : ℝ
  water : ℝ
  price : ℝ

/-- Proves that the price per glass on the second day is $0.40 given the conditions -/
theorem orangeade_price_day2 (day1 day2 : OrangeadeDay) :
  day1.juice = day2.juice ∧ 
  day1.water = day1.juice ∧ 
  day2.water = 2 * day1.water ∧ 
  day1.price = 0.60 ∧ 
  (day1.juice + day1.water) * day1.price = (day2.juice + day2.water) * day2.price
  → day2.price = 0.40 := by
  sorry

#check orangeade_price_day2

end NUMINAMATH_CALUDE_orangeade_price_day2_l3614_361437


namespace NUMINAMATH_CALUDE_inverse_function_property_l3614_361495

-- Define a function f and its inverse f_inv
variable (f : ℝ → ℝ) (f_inv : ℝ → ℝ)

-- Define the property that f and f_inv are inverse functions
def are_inverse (f : ℝ → ℝ) (f_inv : ℝ → ℝ) : Prop :=
  ∀ x, f (f_inv x) = x ∧ f_inv (f x) = x

-- Define the property that f(x+2) and f_inv(x-1) are inverse functions
def special_inverse_property (f : ℝ → ℝ) (f_inv : ℝ → ℝ) : Prop :=
  ∀ x, f ((f_inv (x - 1)) + 2) = x ∧ f_inv (f (x + 2) - 1) = x

-- State the theorem
theorem inverse_function_property (f : ℝ → ℝ) (f_inv : ℝ → ℝ) 
  (h1 : are_inverse f f_inv) 
  (h2 : special_inverse_property f f_inv) : 
  f_inv 2007 - f_inv 1 = 4012 := by
sorry

end NUMINAMATH_CALUDE_inverse_function_property_l3614_361495


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l3614_361406

theorem fraction_to_decimal : (3 : ℚ) / 24 = 0.125 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l3614_361406


namespace NUMINAMATH_CALUDE_prime_quadratic_roots_l3614_361408

theorem prime_quadratic_roots (p : ℕ) : 
  Nat.Prime p → 
  (∃ x y : ℤ, x^2 + p*x - 444*p = 0 ∧ y^2 + p*y - 444*p = 0) → 
  p = 37 :=
sorry

end NUMINAMATH_CALUDE_prime_quadratic_roots_l3614_361408


namespace NUMINAMATH_CALUDE_race_heartbeats_l3614_361461

/-- The number of heartbeats during a race, given the race distance, heart rate, and pace. -/
def heartbeats_during_race (distance : ℕ) (heart_rate : ℕ) (pace : ℕ) : ℕ :=
  distance * pace * heart_rate

/-- Theorem stating that the number of heartbeats during a 30-mile race
    with a heart rate of 160 beats per minute and a pace of 6 minutes per mile
    is equal to 28800. -/
theorem race_heartbeats :
  heartbeats_during_race 30 160 6 = 28800 := by
  sorry

#eval heartbeats_during_race 30 160 6

end NUMINAMATH_CALUDE_race_heartbeats_l3614_361461


namespace NUMINAMATH_CALUDE_no_prime_p_and_p6_plus_6_prime_l3614_361417

theorem no_prime_p_and_p6_plus_6_prime :
  ¬ ∃ (p : ℕ), Nat.Prime p ∧ Nat.Prime (p^6 + 6) := by
  sorry

end NUMINAMATH_CALUDE_no_prime_p_and_p6_plus_6_prime_l3614_361417


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3614_361421

theorem sqrt_equation_solution :
  ∃ s : ℝ, (Real.sqrt (3 * Real.sqrt (s - 1)) = (9 - s) ^ (1/4)) ∧ s = 1.8 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3614_361421


namespace NUMINAMATH_CALUDE_minimum_rental_fee_for_class_trip_l3614_361447

/-- Calculates the minimum rental fee for a class trip --/
def minimum_rental_fee (total_students : ℕ) 
  (small_boat_capacity small_boat_cost large_boat_capacity large_boat_cost : ℕ) : ℕ :=
  let large_boats := total_students / large_boat_capacity
  let remaining_students := total_students % large_boat_capacity
  let small_boats := (remaining_students + small_boat_capacity - 1) / small_boat_capacity
  large_boats * large_boat_cost + small_boats * small_boat_cost

theorem minimum_rental_fee_for_class_trip :
  minimum_rental_fee 48 3 16 5 24 = 232 :=
by sorry

end NUMINAMATH_CALUDE_minimum_rental_fee_for_class_trip_l3614_361447


namespace NUMINAMATH_CALUDE_student_tickets_sold_l3614_361449

/-- Proves the number of student tickets sold given ticket prices and total sales information -/
theorem student_tickets_sold
  (adult_price : ℝ)
  (student_price : ℝ)
  (total_tickets : ℕ)
  (total_amount : ℝ)
  (h1 : adult_price = 4)
  (h2 : student_price = 2.5)
  (h3 : total_tickets = 59)
  (h4 : total_amount = 222.5) :
  ∃ (student_tickets : ℕ),
    student_tickets = 9 ∧
    (total_tickets - student_tickets) * adult_price + student_tickets * student_price = total_amount :=
by
  sorry

#check student_tickets_sold

end NUMINAMATH_CALUDE_student_tickets_sold_l3614_361449


namespace NUMINAMATH_CALUDE_smallest_n_divisible_by_1991_l3614_361407

theorem smallest_n_divisible_by_1991 : ∃ (n : ℕ),
  (∀ (m : ℕ), m < n → ∃ (S : Finset ℤ), S.card = m ∧
    ∀ (a b : ℤ), a ∈ S → b ∈ S → a ≠ b → ¬(1991 ∣ (a + b)) ∧ ¬(1991 ∣ (a - b))) ∧
  (∀ (S : Finset ℤ), S.card = n →
    ∃ (a b : ℤ), a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ (1991 ∣ (a + b) ∨ 1991 ∣ (a - b))) ∧
  n = 997 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_divisible_by_1991_l3614_361407


namespace NUMINAMATH_CALUDE_square_area_increase_l3614_361485

theorem square_area_increase (s : ℝ) (h : s > 0) : 
  let new_side := 1.25 * s
  let original_area := s^2
  let new_area := new_side^2
  (new_area - original_area) / original_area = 0.5625 := by
sorry

end NUMINAMATH_CALUDE_square_area_increase_l3614_361485


namespace NUMINAMATH_CALUDE_running_time_difference_l3614_361477

/-- The difference in running time between two runners -/
theorem running_time_difference
  (d : ℝ) -- Total distance
  (lawrence_distance : ℝ) -- Lawrence's running distance
  (lawrence_speed : ℝ) -- Lawrence's speed in minutes per kilometer
  (george_distance : ℝ) -- George's running distance
  (george_speed : ℝ) -- George's speed in minutes per kilometer
  (h1 : lawrence_distance = d / 2)
  (h2 : george_distance = d / 2)
  (h3 : lawrence_speed = 8)
  (h4 : george_speed = 12) :
  george_distance * george_speed - lawrence_distance * lawrence_speed = 2 * d :=
by sorry

end NUMINAMATH_CALUDE_running_time_difference_l3614_361477
