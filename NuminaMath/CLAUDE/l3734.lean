import Mathlib

namespace NUMINAMATH_CALUDE_combination_equality_l3734_373451

theorem combination_equality (x : ℕ) : 
  (Nat.choose 10 x = Nat.choose 10 (3*x - 2)) → (x = 1 ∨ x = 3) := by
sorry

end NUMINAMATH_CALUDE_combination_equality_l3734_373451


namespace NUMINAMATH_CALUDE_ellipse_equation_l3734_373474

/-- Proves that an ellipse with given conditions has the equation x^2 + 4y^2 = 1 -/
theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  let e := Real.sqrt 3 / 2
  let ellipse := fun (x y : ℝ) => x^2 / a^2 + y^2 / b^2 = 1
  let line := fun (x y : ℝ) => x - y + 1 = 0
  ∃ (A B C : ℝ × ℝ),
    ellipse A.1 A.2 ∧
    ellipse B.1 B.2 ∧
    line A.1 A.2 ∧
    line B.1 B.2 ∧
    C.1 = 0 ∧
    line C.1 C.2 ∧
    (3 * (B.1 - A.1), 3 * (B.2 - A.2)) = (2 * (C.1 - B.1), 2 * (C.2 - B.2)) →
  e^2 * a^2 = a^2 - b^2 →
  ∀ (x y : ℝ), x^2 + 4*y^2 = 1 ↔ ellipse x y := by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l3734_373474


namespace NUMINAMATH_CALUDE_square_cross_section_cylinder_volume_l3734_373400

/-- A cylinder with a square cross-section and lateral surface area 4π has volume 2π -/
theorem square_cross_section_cylinder_volume (h : ℝ) (r : ℝ) : 
  h > 0 → r > 0 → h = 2*r → h * (4*r) = 4*π → π * r^2 * h = 2*π := by
  sorry

end NUMINAMATH_CALUDE_square_cross_section_cylinder_volume_l3734_373400


namespace NUMINAMATH_CALUDE_number_equation_solution_l3734_373423

theorem number_equation_solution : ∃ x : ℝ, (27 + 2 * x = 39) ∧ (x = 6) := by
  sorry

end NUMINAMATH_CALUDE_number_equation_solution_l3734_373423


namespace NUMINAMATH_CALUDE_min_value_sine_product_equality_condition_l3734_373457

theorem min_value_sine_product (x₁ x₂ x₃ x₄ : Real) 
  (h_pos : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0) 
  (h_sum : x₁ + x₂ + x₃ + x₄ = Real.pi) : 
  (2 * Real.sin x₁ ^ 2 + 1 / Real.sin x₁ ^ 2) *
  (2 * Real.sin x₂ ^ 2 + 1 / Real.sin x₂ ^ 2) *
  (2 * Real.sin x₃ ^ 2 + 1 / Real.sin x₃ ^ 2) *
  (2 * Real.sin x₄ ^ 2 + 1 / Real.sin x₄ ^ 2) ≥ 81 :=
by sorry

theorem equality_condition (x₁ x₂ x₃ x₄ : Real) 
  (h_pos : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0) 
  (h_sum : x₁ + x₂ + x₃ + x₄ = Real.pi) :
  (2 * Real.sin x₁ ^ 2 + 1 / Real.sin x₁ ^ 2) *
  (2 * Real.sin x₂ ^ 2 + 1 / Real.sin x₂ ^ 2) *
  (2 * Real.sin x₃ ^ 2 + 1 / Real.sin x₃ ^ 2) *
  (2 * Real.sin x₄ ^ 2 + 1 / Real.sin x₄ ^ 2) = 81 ↔
  x₁ = Real.pi / 4 ∧ x₂ = Real.pi / 4 ∧ x₃ = Real.pi / 4 ∧ x₄ = Real.pi / 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sine_product_equality_condition_l3734_373457


namespace NUMINAMATH_CALUDE_max_x_value_l3734_373447

theorem max_x_value (x : ℝ) : 
  ((5*x - 25)/(4*x - 5))^2 + ((5*x - 25)/(4*x - 5)) = 18 → x ≤ 55/29 := by
  sorry

end NUMINAMATH_CALUDE_max_x_value_l3734_373447


namespace NUMINAMATH_CALUDE_mutual_choice_exists_l3734_373468

/-- A monotonic increasing function from {1,...,n} to {1,...,n} -/
def MonotonicFunction (n : ℕ) := {f : Fin n → Fin n // ∀ i j, i ≤ j → f i ≤ f j}

/-- The theorem statement -/
theorem mutual_choice_exists (n : ℕ) (hn : n > 0) (f g : MonotonicFunction n) :
  ∃ k : Fin n, (f.val ∘ g.val) k = k :=
sorry

end NUMINAMATH_CALUDE_mutual_choice_exists_l3734_373468


namespace NUMINAMATH_CALUDE_divisibility_by_48_l3734_373484

theorem divisibility_by_48 (a b c : ℤ) (h1 : a < c) (h2 : a^2 + c^2 = 2*b^2) :
  ∃ k : ℤ, c^2 - a^2 = 48 * k := by
sorry

end NUMINAMATH_CALUDE_divisibility_by_48_l3734_373484


namespace NUMINAMATH_CALUDE_solution_implies_k_value_l3734_373416

theorem solution_implies_k_value (x k : ℚ) : 
  (x = 5 ∧ 4 * x + 2 * k = 8) → k = -6 := by
  sorry

end NUMINAMATH_CALUDE_solution_implies_k_value_l3734_373416


namespace NUMINAMATH_CALUDE_equation_represents_intersecting_lines_l3734_373446

theorem equation_represents_intersecting_lines (x y : ℝ) :
  (x + y)^2 = x^2 + y^2 + 3*x*y ↔ x*y = 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_represents_intersecting_lines_l3734_373446


namespace NUMINAMATH_CALUDE_celine_collected_ten_erasers_l3734_373458

/-- The number of erasers collected by each person -/
structure EraserCollection where
  gabriel : ℕ
  celine : ℕ
  julian : ℕ
  erica : ℕ
  david : ℕ

/-- The conditions of the eraser collection problem -/
def valid_collection (ec : EraserCollection) : Prop :=
  ec.celine = 2 * ec.gabriel ∧
  ec.julian = 2 * ec.celine ∧
  ec.erica = 3 * ec.julian ∧
  ec.david = 5 * ec.erica ∧
  ec.gabriel ≥ 1 ∧ ec.celine ≥ 1 ∧ ec.julian ≥ 1 ∧ ec.erica ≥ 1 ∧ ec.david ≥ 1 ∧
  ec.gabriel + ec.celine + ec.julian + ec.erica + ec.david = 380

/-- The theorem stating that Celine collected 10 erasers -/
theorem celine_collected_ten_erasers (ec : EraserCollection) 
  (h : valid_collection ec) : ec.celine = 10 := by
  sorry

end NUMINAMATH_CALUDE_celine_collected_ten_erasers_l3734_373458


namespace NUMINAMATH_CALUDE_crayons_per_box_l3734_373439

theorem crayons_per_box 
  (total_boxes : ℕ) 
  (total_crayons : ℕ) 
  (h1 : total_boxes = 7)
  (h2 : total_crayons = 35)
  (h3 : total_crayons = total_boxes * (total_crayons / total_boxes)) :
  total_crayons / total_boxes = 5 := by
sorry

end NUMINAMATH_CALUDE_crayons_per_box_l3734_373439


namespace NUMINAMATH_CALUDE_pencil_discount_percentage_l3734_373462

/-- Proves that the discount on pencils is 20% given the problem conditions --/
theorem pencil_discount_percentage (cucumber_count : ℕ) (pencil_count : ℕ) (item_price : ℕ) (total_spent : ℕ) :
  cucumber_count = 100 →
  cucumber_count = 2 * pencil_count →
  item_price = 20 →
  total_spent = 2800 →
  (1 - (total_spent - cucumber_count * item_price) / (pencil_count * item_price : ℚ)) * 100 = 20 := by
  sorry

#check pencil_discount_percentage

end NUMINAMATH_CALUDE_pencil_discount_percentage_l3734_373462


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3734_373406

def A : Set ℝ := {-1, 0, 1, 2}
def B : Set ℝ := {x | x^2 ≤ 1}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3734_373406


namespace NUMINAMATH_CALUDE_modular_arithmetic_problem_l3734_373443

theorem modular_arithmetic_problem (n : ℕ) : 
  n < 19 ∧ (5 * n) % 19 = 1 → ((3^n)^2 - 3) % 19 = 3 := by
  sorry

end NUMINAMATH_CALUDE_modular_arithmetic_problem_l3734_373443


namespace NUMINAMATH_CALUDE_jenna_costume_cost_l3734_373428

/-- Represents the cost of material for Jenna's costume --/
def costume_cost (skirt_length : ℝ) (skirt_width : ℝ) (num_skirts : ℕ) 
                 (bodice_area : ℝ) (sleeve_area : ℝ) (num_sleeves : ℕ) 
                 (cost_per_sqft : ℝ) : ℝ :=
  let skirt_area := skirt_length * skirt_width
  let total_skirt_area := skirt_area * num_skirts
  let total_sleeve_area := sleeve_area * num_sleeves
  let total_area := total_skirt_area + total_sleeve_area + bodice_area
  total_area * cost_per_sqft

/-- Theorem: The total cost of material for Jenna's costume is $468 --/
theorem jenna_costume_cost : 
  costume_cost 12 4 3 2 5 2 3 = 468 := by
  sorry

end NUMINAMATH_CALUDE_jenna_costume_cost_l3734_373428


namespace NUMINAMATH_CALUDE_units_digit_sum_factorials_500_l3734_373433

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def units_digit (n : ℕ) : ℕ := n % 10

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem units_digit_sum_factorials_500 :
  units_digit (sum_factorials 500) = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_sum_factorials_500_l3734_373433


namespace NUMINAMATH_CALUDE_fixed_distance_from_linear_combination_l3734_373408

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Given vectors a and b, and a vector p satisfying the condition,
    prove that t = 9/8 and u = -1/8 make ‖p - (t*a + u*b)‖ constant. -/
theorem fixed_distance_from_linear_combination
  (a b p : E) (h : ‖p - b‖ = 3 * ‖p - a‖) :
  ∃ (c : ℝ), ∀ (q : E), ‖q - b‖ = 3 * ‖q - a‖ →
    ‖q - ((9/8 : ℝ) • a + (-1/8 : ℝ) • b)‖ = c :=
sorry

end NUMINAMATH_CALUDE_fixed_distance_from_linear_combination_l3734_373408


namespace NUMINAMATH_CALUDE_dogsled_race_time_difference_l3734_373456

theorem dogsled_race_time_difference 
  (course_length : ℝ) 
  (speed_T : ℝ) 
  (speed_difference : ℝ) :
  course_length = 300 →
  speed_T = 20 →
  speed_difference = 5 →
  let speed_A := speed_T + speed_difference
  let time_T := course_length / speed_T
  let time_A := course_length / speed_A
  time_T - time_A = 3 := by
sorry

end NUMINAMATH_CALUDE_dogsled_race_time_difference_l3734_373456


namespace NUMINAMATH_CALUDE_asterisk_replacement_l3734_373463

theorem asterisk_replacement : ∃! x : ℝ, (x / 21) * (42 / 84) = 1 := by
  sorry

end NUMINAMATH_CALUDE_asterisk_replacement_l3734_373463


namespace NUMINAMATH_CALUDE_ninth_root_of_unity_l3734_373413

theorem ninth_root_of_unity (y : ℂ) : 
  y = Complex.exp (2 * Real.pi * I / 9) → y^9 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ninth_root_of_unity_l3734_373413


namespace NUMINAMATH_CALUDE_no_negative_log_base_exists_positive_fraction_log_base_l3734_373466

-- Define the property of being a valid logarithm base
def IsValidLogBase (b : ℝ) : Prop := b > 0 ∧ b ≠ 1

-- Theorem 1: No negative number can be a valid logarithm base
theorem no_negative_log_base :
  ∀ b : ℝ, b < 0 → ¬(IsValidLogBase b) :=
sorry

-- Theorem 2: There exists a positive fraction that is a valid logarithm base
theorem exists_positive_fraction_log_base :
  ∃ b : ℝ, 0 < b ∧ b < 1 ∧ IsValidLogBase b :=
sorry

end NUMINAMATH_CALUDE_no_negative_log_base_exists_positive_fraction_log_base_l3734_373466


namespace NUMINAMATH_CALUDE_paiges_drawers_l3734_373473

theorem paiges_drawers (clothing_per_drawer : ℕ) (total_clothing : ℕ) (num_drawers : ℕ) :
  clothing_per_drawer = 2 →
  total_clothing = 8 →
  num_drawers * clothing_per_drawer = total_clothing →
  num_drawers = 4 := by
sorry

end NUMINAMATH_CALUDE_paiges_drawers_l3734_373473


namespace NUMINAMATH_CALUDE_two_pencils_one_pen_cost_l3734_373477

-- Define the cost of a pencil and a pen
variable (pencil_cost pen_cost : ℚ)

-- Define the given conditions
axiom condition1 : 3 * pencil_cost + pen_cost = 3
axiom condition2 : 3 * pencil_cost + 4 * pen_cost = (15/2)

-- State the theorem to be proved
theorem two_pencils_one_pen_cost : 
  2 * pencil_cost + pen_cost = (5/2) := by
sorry

end NUMINAMATH_CALUDE_two_pencils_one_pen_cost_l3734_373477


namespace NUMINAMATH_CALUDE_range_of_a_l3734_373499

def proposition_p (a : ℝ) : Prop :=
  ∀ x : ℝ, x ∈ ({1, 2} : Set ℝ) → 3 * x^2 - a ≥ 0

def proposition_q (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0

theorem range_of_a (a : ℝ) :
  (proposition_p a ∧ proposition_q a) → (a ≤ -2 ∨ (1 ≤ a ∧ a ≤ 3)) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3734_373499


namespace NUMINAMATH_CALUDE_square_rectangle_area_difference_l3734_373402

theorem square_rectangle_area_difference :
  let square_side : ℝ := 2
  let rect_length : ℝ := 2
  let rect_width : ℝ := 2
  let square_area := square_side ^ 2
  let rect_area := rect_length * rect_width
  square_area - rect_area = 0 := by
sorry

end NUMINAMATH_CALUDE_square_rectangle_area_difference_l3734_373402


namespace NUMINAMATH_CALUDE_product_zero_implies_factor_zero_l3734_373479

theorem product_zero_implies_factor_zero (a b c : ℝ) : a * b * c = 0 → (a = 0 ∨ b = 0 ∨ c = 0) := by
  sorry

end NUMINAMATH_CALUDE_product_zero_implies_factor_zero_l3734_373479


namespace NUMINAMATH_CALUDE_river_depth_l3734_373489

/-- The depth of a river given its width, flow rate, and volume of water per minute. -/
theorem river_depth (width : ℝ) (flow_rate : ℝ) (volume_per_minute : ℝ) :
  width = 45 →
  flow_rate = 5 →
  volume_per_minute = 7500 →
  (volume_per_minute / (width * (flow_rate * 1000 / 60))) = 2 := by
  sorry


end NUMINAMATH_CALUDE_river_depth_l3734_373489


namespace NUMINAMATH_CALUDE_polynomial_equality_l3734_373424

/-- Given two polynomials p(x) = 2x^2 + 5x - 2 and q(x) = 2x^2 + 5x + 4,
    prove that the polynomial r(x) = 10x + 6 satisfies p(x) + r(x) = q(x) for all x. -/
theorem polynomial_equality (x : ℝ) :
  (2 * x^2 + 5 * x - 2) + (10 * x + 6) = 2 * x^2 + 5 * x + 4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l3734_373424


namespace NUMINAMATH_CALUDE_dog_fur_objects_l3734_373417

theorem dog_fur_objects (burrs : ℕ) (ticks : ℕ) : 
  burrs = 12 → ticks = 6 * burrs → burrs + ticks = 84 := by
  sorry

end NUMINAMATH_CALUDE_dog_fur_objects_l3734_373417


namespace NUMINAMATH_CALUDE_nested_arithmetic_expression_l3734_373493

theorem nested_arithmetic_expression : 1 - (2 - (3 - 4 - (5 - 6))) = -1 := by
  sorry

end NUMINAMATH_CALUDE_nested_arithmetic_expression_l3734_373493


namespace NUMINAMATH_CALUDE_alpha_value_l3734_373452

theorem alpha_value (α : Real) (h1 : α ∈ Set.Ioo (π/2) π) 
  (h2 : ∀ x, (Real.sin α) ^ (x^2 - 2*x + 3) ≤ 1/4) : α = 5*π/6 := by
  sorry

end NUMINAMATH_CALUDE_alpha_value_l3734_373452


namespace NUMINAMATH_CALUDE_sin_cos_identity_l3734_373442

theorem sin_cos_identity : 
  Real.sin (75 * π / 180) * Real.sin (15 * π / 180) + 
  Real.cos (75 * π / 180) * Real.cos (15 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l3734_373442


namespace NUMINAMATH_CALUDE_ali_seashells_to_friends_l3734_373476

/-- The number of seashells Ali gave to his friends -/
def seashells_to_friends (initial : ℕ) (to_brothers : ℕ) (remaining : ℕ) : ℕ :=
  initial - to_brothers - 2 * remaining

theorem ali_seashells_to_friends :
  seashells_to_friends 180 30 55 = 40 := by
  sorry

end NUMINAMATH_CALUDE_ali_seashells_to_friends_l3734_373476


namespace NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l3734_373471

theorem tan_alpha_plus_pi_fourth (α : ℝ) 
  (h : 2 * Real.sin (2 * α) = 1 + Real.cos (2 * α)) : 
  Real.tan (α + π / 4) = -1 ∨ Real.tan (α + π / 4) = 3 := by
sorry

end NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l3734_373471


namespace NUMINAMATH_CALUDE_coordinate_system_proofs_l3734_373415

-- Define the points A, B, and C
def A (a : ℝ) : ℝ × ℝ := (-2, a + 1)
def B (a : ℝ) : ℝ × ℝ := (a - 1, 4)
def C (b : ℝ) : ℝ × ℝ := (b - 2, b)

-- Define the conditions and prove the statements
theorem coordinate_system_proofs :
  -- 1. When C is on the x-axis, its coordinates are (-2,0)
  (∃ b : ℝ, C b = (-2, 0) ∧ (C b).2 = 0) ∧
  -- 2. When C is on the y-axis, its coordinates are (0,2)
  (∃ b : ℝ, C b = (0, 2) ∧ (C b).1 = 0) ∧
  -- 3. When AB is parallel to the x-axis, the distance between A and B is 4
  (∃ a : ℝ, (A a).2 = (B a).2 ∧ Real.sqrt ((A a).1 - (B a).1)^2 = 4) ∧
  -- 4. When CD is perpendicular to the x-axis at point D and CD=1, 
  --    the coordinates of C are either (-1,1) or (-3,-1)
  (∃ b d : ℝ, (C b).1 = d ∧ Real.sqrt ((C b).1 - d)^2 + (C b).2^2 = 1 ∧
    ((C b = (-1, 1)) ∨ (C b = (-3, -1)))) :=
by sorry

end NUMINAMATH_CALUDE_coordinate_system_proofs_l3734_373415


namespace NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_coefficient_x_squared_is_three_l3734_373483

theorem coefficient_x_squared_in_expansion (x : ℝ) : 
  (Finset.range 4).sum (λ k => (Nat.choose 3 k) * x^k) = 1 + 3*x + 3*x^2 + x^3 := by
  sorry

theorem coefficient_x_squared_is_three : 
  (Finset.range 4).sum (λ k => (Nat.choose 3 k) * (if k = 2 then 1 else 0)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_coefficient_x_squared_is_three_l3734_373483


namespace NUMINAMATH_CALUDE_sqrt_25_l3734_373453

theorem sqrt_25 : Real.sqrt 25 = 5 ∨ Real.sqrt 25 = -5 := by sorry

end NUMINAMATH_CALUDE_sqrt_25_l3734_373453


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l3734_373430

theorem product_of_three_numbers (a b c : ℝ) 
  (sum_eq_one : a + b + c = 1)
  (sum_sq_eq_one : a^2 + b^2 + c^2 = 1)
  (sum_cube_eq_one : a^3 + b^3 + c^3 = 1) : 
  a * b * c = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l3734_373430


namespace NUMINAMATH_CALUDE_remainder_eight_power_2010_mod_100_l3734_373403

theorem remainder_eight_power_2010_mod_100 : 8^2010 % 100 = 24 := by
  sorry

end NUMINAMATH_CALUDE_remainder_eight_power_2010_mod_100_l3734_373403


namespace NUMINAMATH_CALUDE_bianca_extra_flowers_l3734_373472

/-- The number of extra flowers Bianca picked -/
def extra_flowers (tulips roses used : ℕ) : ℕ :=
  tulips + roses - used

/-- Proof that Bianca picked 7 extra flowers -/
theorem bianca_extra_flowers :
  extra_flowers 39 49 81 = 7 := by
  sorry

end NUMINAMATH_CALUDE_bianca_extra_flowers_l3734_373472


namespace NUMINAMATH_CALUDE_emily_toys_sale_l3734_373482

def sell_toys (initial : ℕ) (percent1 : ℕ) (percent2 : ℕ) : ℕ :=
  let remaining_after_day1 := initial - (initial * percent1 / 100)
  let remaining_after_day2 := remaining_after_day1 - (remaining_after_day1 * percent2 / 100)
  remaining_after_day2

theorem emily_toys_sale :
  sell_toys 35 50 60 = 8 := by
  sorry

end NUMINAMATH_CALUDE_emily_toys_sale_l3734_373482


namespace NUMINAMATH_CALUDE_intersection_A_B_l3734_373498

def A : Set ℝ := {x | ∃ n : ℤ, x = 2 * n - 1}
def B : Set ℝ := {x | x^2 - 4*x < 0}

theorem intersection_A_B : A ∩ B = {1, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3734_373498


namespace NUMINAMATH_CALUDE_min_value_quadratic_l3734_373448

theorem min_value_quadratic (y : ℝ) : 
  (5 * y^2 + 5 * y + 4 = 9) → 
  (∀ z : ℝ, 5 * z^2 + 5 * z + 4 = 9 → y ≤ z) → 
  y = (-1 - Real.sqrt 5) / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l3734_373448


namespace NUMINAMATH_CALUDE_prob_all_same_color_is_34_455_l3734_373445

def red_marbles : ℕ := 4
def white_marbles : ℕ := 5
def blue_marbles : ℕ := 6
def total_marbles : ℕ := red_marbles + white_marbles + blue_marbles

def prob_all_same_color : ℚ :=
  (red_marbles * (red_marbles - 1) * (red_marbles - 2) +
   white_marbles * (white_marbles - 1) * (white_marbles - 2) +
   blue_marbles * (blue_marbles - 1) * (blue_marbles - 2)) /
  (total_marbles * (total_marbles - 1) * (total_marbles - 2))

theorem prob_all_same_color_is_34_455 : prob_all_same_color = 34 / 455 := by
  sorry

end NUMINAMATH_CALUDE_prob_all_same_color_is_34_455_l3734_373445


namespace NUMINAMATH_CALUDE_least_common_multiple_problem_l3734_373455

def is_divisible_by_all (n : ℕ) : Prop :=
  n % 24 = 0 ∧ n % 32 = 0 ∧ n % 36 = 0 ∧ n % 54 = 0

theorem least_common_multiple_problem : 
  ∃! x : ℕ, (is_divisible_by_all (856 + x) ∧ 
    ∀ y : ℕ, y < x → ¬is_divisible_by_all (856 + y)) ∧ 
  x = 8 := by
  sorry

end NUMINAMATH_CALUDE_least_common_multiple_problem_l3734_373455


namespace NUMINAMATH_CALUDE_nancy_and_rose_bracelets_l3734_373404

/-- The number of beads required for each bracelet -/
def beads_per_bracelet : ℕ := 8

/-- The number of metal beads Nancy has -/
def nancy_metal_beads : ℕ := 40

/-- The number of pearl beads Nancy has -/
def nancy_pearl_beads : ℕ := nancy_metal_beads + 20

/-- The number of crystal beads Rose has -/
def rose_crystal_beads : ℕ := 20

/-- The number of stone beads Rose has -/
def rose_stone_beads : ℕ := 2 * rose_crystal_beads

/-- The total number of beads Nancy and Rose have -/
def total_beads : ℕ := nancy_metal_beads + nancy_pearl_beads + rose_crystal_beads + rose_stone_beads

/-- The number of bracelets Nancy and Rose can make -/
def bracelets_made : ℕ := total_beads / beads_per_bracelet

theorem nancy_and_rose_bracelets : bracelets_made = 20 := by
  sorry

end NUMINAMATH_CALUDE_nancy_and_rose_bracelets_l3734_373404


namespace NUMINAMATH_CALUDE_cube_face_sum_l3734_373460

theorem cube_face_sum (a b c d e f : ℕ+) :
  (a * b * c + a * e * c + a * b * f + a * e * f +
   d * b * c + d * e * c + d * b * f + d * e * f) = 1491 →
  (a + b + c + d + e + f : ℕ) = 41 := by
sorry

end NUMINAMATH_CALUDE_cube_face_sum_l3734_373460


namespace NUMINAMATH_CALUDE_equation_proof_l3734_373480

theorem equation_proof : Real.sqrt (3^2 + 4^2) / Real.sqrt (25 - 1) = 5 * Real.sqrt 6 / 12 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l3734_373480


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3734_373487

theorem complex_equation_solution (z : ℂ) : z * (1 - Complex.I) = 2 → z = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3734_373487


namespace NUMINAMATH_CALUDE_quadratic_root_property_l3734_373464

theorem quadratic_root_property (n : ℝ) (x₁ x₂ : ℝ) : 
  (x₁^2 - 3*x₁ + n = 0) → 
  (x₂^2 - 3*x₂ + n = 0) → 
  (x₁ + x₂ - 2 = x₁ * x₂) → 
  n = 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_property_l3734_373464


namespace NUMINAMATH_CALUDE_triangle_side_ratio_l3734_373481

theorem triangle_side_ratio (a b c : ℝ) (A : ℝ) (h1 : A = 2 * Real.pi / 3) 
  (h2 : a^2 = 2*b*c + 3*c^2) : c/b = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_ratio_l3734_373481


namespace NUMINAMATH_CALUDE_divisibility_property_l3734_373491

theorem divisibility_property (y : ℕ) (h : y ≠ 0) :
  (y - 1) ∣ (y^(y^2) - 2*y^(y+1) + 1) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_property_l3734_373491


namespace NUMINAMATH_CALUDE_hyperbolas_same_asymptotes_l3734_373434

-- Define the two hyperbolas
def hyperbola1 (x y : ℝ) : Prop := x^2/16 - y^2/9 = 1
def hyperbola2 (x y : ℝ) : Prop := y^2/9 - x^2/16 = 1

-- Define the asymptotes for a hyperbola
def asymptotes (a b : ℝ) (x y : ℝ) : Prop := y = (b/a)*x ∨ y = -(b/a)*x

-- Theorem stating that both hyperbolas have the same asymptotes
theorem hyperbolas_same_asymptotes :
  ∃ (a b : ℝ), ∀ (x y : ℝ),
    (hyperbola1 x y ∨ hyperbola2 x y) → asymptotes a b x y :=
sorry

end NUMINAMATH_CALUDE_hyperbolas_same_asymptotes_l3734_373434


namespace NUMINAMATH_CALUDE_football_team_size_l3734_373435

/-- Represents the number of players on a football team -/
def total_players : ℕ := 70

/-- Represents the number of throwers on the team -/
def throwers : ℕ := 52

/-- Represents the total number of right-handed players -/
def right_handed_players : ℕ := 64

/-- States that one third of non-throwers are left-handed -/
axiom one_third_non_throwers_left_handed :
  (total_players - throwers) / 3 = (total_players - throwers - (right_handed_players - throwers))

/-- All throwers are right-handed -/
axiom all_throwers_right_handed :
  throwers ≤ right_handed_players

/-- Theorem stating that the total number of players is 70 -/
theorem football_team_size :
  total_players = 70 :=
sorry

end NUMINAMATH_CALUDE_football_team_size_l3734_373435


namespace NUMINAMATH_CALUDE_negative_root_implies_a_less_than_negative_three_l3734_373432

theorem negative_root_implies_a_less_than_negative_three (a : ℝ) :
  (∃ x : ℝ, x < 0 ∧ 5^x = (a + 3) / (a - 3)) → a < -3 := by
  sorry

end NUMINAMATH_CALUDE_negative_root_implies_a_less_than_negative_three_l3734_373432


namespace NUMINAMATH_CALUDE_demand_exceeds_15000_only_in_7_and_8_l3734_373444

def S (n : ℕ) : ℚ := (n : ℚ) / 90 * (21 * n - n^2 - 5)

def a (n : ℕ) : ℚ := S n - S (n-1)

theorem demand_exceeds_15000_only_in_7_and_8 :
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 12 →
    (a n > (3/2) ↔ n = 7 ∨ n = 8) :=
sorry

end NUMINAMATH_CALUDE_demand_exceeds_15000_only_in_7_and_8_l3734_373444


namespace NUMINAMATH_CALUDE_campers_in_two_classes_l3734_373486

/-- Represents the number of campers in a single class -/
def class_size : ℕ := 20

/-- Represents the number of campers in all three classes -/
def in_all_classes : ℕ := 4

/-- Represents the number of campers in exactly one class -/
def in_one_class : ℕ := 24

/-- Represents the total number of campers -/
def total_campers : ℕ := class_size * 3 - 2 * in_all_classes

theorem campers_in_two_classes : 
  ∃ (x : ℕ), x = total_campers - in_one_class - in_all_classes ∧ x = 12 := by
  sorry

end NUMINAMATH_CALUDE_campers_in_two_classes_l3734_373486


namespace NUMINAMATH_CALUDE_discount_percentage_proof_l3734_373441

theorem discount_percentage_proof (jacket_price shirt_price shoes_price : ℝ)
  (jacket_discount shirt_discount shoes_discount : ℝ) :
  jacket_price = 120 ∧ shirt_price = 60 ∧ shoes_price = 90 ∧
  jacket_discount = 0.30 ∧ shirt_discount = 0.50 ∧ shoes_discount = 0.25 →
  (jacket_price * jacket_discount + shirt_price * shirt_discount + shoes_price * shoes_discount) /
  (jacket_price + shirt_price + shoes_price) = 0.328 := by
  sorry

end NUMINAMATH_CALUDE_discount_percentage_proof_l3734_373441


namespace NUMINAMATH_CALUDE_arithmetic_segments_form_quadrilateral_l3734_373495

/-- Four segments in an arithmetic sequence with total length 3 can form a quadrilateral -/
theorem arithmetic_segments_form_quadrilateral :
  ∀ (a d : ℝ),
  a > 0 ∧ d > 0 →
  a + (a + d) + (a + 2*d) + (a + 3*d) = 3 →
  (a + (a + d) + (a + 2*d) > a + 3*d) ∧
  (a + (a + d) + (a + 3*d) > a + 2*d) ∧
  (a + (a + 2*d) + (a + 3*d) > a + d) ∧
  ((a + d) + (a + 2*d) + (a + 3*d) > a) :=
by
  sorry


end NUMINAMATH_CALUDE_arithmetic_segments_form_quadrilateral_l3734_373495


namespace NUMINAMATH_CALUDE_complex_multiplication_opposites_l3734_373465

theorem complex_multiplication_opposites (a : ℝ) (i : ℂ) (h1 : a > 0) (h2 : i * i = -1) :
  (Complex.re (a * i * (a + i)) = -Complex.im (a * i * (a + i))) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_opposites_l3734_373465


namespace NUMINAMATH_CALUDE_restaurant_group_composition_l3734_373440

/-- Proves that in a group of 11 people, where adult meals cost $8 each and kids eat free,
    if the total cost is $72, then the number of kids in the group is 2. -/
theorem restaurant_group_composition (total_people : ℕ) (adult_meal_cost : ℕ) (total_cost : ℕ) 
  (h1 : total_people = 11)
  (h2 : adult_meal_cost = 8)
  (h3 : total_cost = 72) :
  total_people - (total_cost / adult_meal_cost) = 2 :=
by sorry

end NUMINAMATH_CALUDE_restaurant_group_composition_l3734_373440


namespace NUMINAMATH_CALUDE_no_prime_factor_6k_plus_5_l3734_373461

theorem no_prime_factor_6k_plus_5 (n k : ℕ+) (p : ℕ) (h_prime : Nat.Prime p) 
  (h_factor : p ∣ n^2 - n + 1) : p ≠ 6 * k + 5 := by
  sorry

end NUMINAMATH_CALUDE_no_prime_factor_6k_plus_5_l3734_373461


namespace NUMINAMATH_CALUDE_largest_n_for_seq_containment_l3734_373426

/-- A bi-infinite sequence of natural numbers -/
def BiInfiniteSeq := ℤ → ℕ

/-- A sequence is periodic with period p if it repeats every p elements -/
def IsPeriodic (s : BiInfiniteSeq) (p : ℕ) : Prop :=
  ∀ i : ℤ, s i = s (i + p)

/-- A subsequence of length n starting at index i is contained in another sequence -/
def SubseqContained (sub main : BiInfiniteSeq) (n : ℕ) (i : ℤ) : Prop :=
  ∀ k : ℕ, k < n → ∃ j : ℤ, sub (i + k) = main j

/-- The main theorem stating the largest possible n -/
theorem largest_n_for_seq_containment :
  ∃ (n : ℕ) (A B : BiInfiniteSeq),
    IsPeriodic A 1995 ∧
    ¬ IsPeriodic B 1995 ∧
    (∀ i : ℤ, SubseqContained B A n i) ∧
    (∀ m : ℕ, m > n →
      ¬ ∃ (C D : BiInfiniteSeq),
        IsPeriodic C 1995 ∧
        ¬ IsPeriodic D 1995 ∧
        (∀ i : ℤ, SubseqContained D C m i)) ∧
    n = 1995 :=
  sorry

end NUMINAMATH_CALUDE_largest_n_for_seq_containment_l3734_373426


namespace NUMINAMATH_CALUDE_opposite_numbers_not_on_hyperbola_l3734_373429

theorem opposite_numbers_not_on_hyperbola (x y : ℝ) : 
  y = 1 / x → x ≠ -y := by
  sorry

end NUMINAMATH_CALUDE_opposite_numbers_not_on_hyperbola_l3734_373429


namespace NUMINAMATH_CALUDE_triangle_inscription_exists_l3734_373470

-- Define the triangle type
structure Triangle :=
  (A B C : Point)

-- Define congruence between triangles
def congruent (t1 t2 : Triangle) : Prop := sorry

-- Define inscribed triangle
def inscribed (inner outer : Triangle) : Prop := sorry

-- Theorem statement
theorem triangle_inscription_exists (ABC : Triangle) :
  ∃ (PQR : Triangle), ∃ (XYZ : Triangle),
    congruent XYZ ABC ∧ inscribed XYZ PQR := by sorry

end NUMINAMATH_CALUDE_triangle_inscription_exists_l3734_373470


namespace NUMINAMATH_CALUDE_probability_six_consecutive_heads_l3734_373485

/-- A sequence of coin flips represented as a list of booleans, where true represents heads and false represents tails. -/
def CoinFlips := List Bool

/-- The number of coin flips. -/
def numFlips : Nat := 10

/-- A function that checks if a list of coin flips contains at least 6 consecutive heads. -/
def hasAtLeastSixConsecutiveHeads (flips : CoinFlips) : Bool :=
  sorry

/-- The total number of possible outcomes for 10 coin flips. -/
def totalOutcomes : Nat := 2^numFlips

/-- The number of favorable outcomes (sequences with at least 6 consecutive heads). -/
def favorableOutcomes : Nat := 129

/-- The probability of getting at least 6 consecutive heads in 10 flips of a fair coin. -/
theorem probability_six_consecutive_heads :
  (favorableOutcomes : ℚ) / totalOutcomes = 129 / 1024 :=
sorry

end NUMINAMATH_CALUDE_probability_six_consecutive_heads_l3734_373485


namespace NUMINAMATH_CALUDE_factory_cost_l3734_373409

/-- Calculates the total cost of employing all workers for one day --/
def total_cost (total_employees : ℕ) 
  (group1_count : ℕ) (group1_rate : ℚ) (group1_regular_hours : ℕ)
  (group2_count : ℕ) (group2_rate : ℚ) (group2_regular_hours : ℕ)
  (group3_count : ℕ) (group3_rate : ℚ) (group3_regular_hours : ℕ) (group3_flat_rate : ℚ)
  (total_hours : ℕ) : ℚ :=
  let group1_cost := group1_count * (
    group1_rate * group1_regular_hours +
    group1_rate * 1.5 * (total_hours - group1_regular_hours)
  )
  let group2_cost := group2_count * (
    group2_rate * group2_regular_hours +
    group2_rate * 2 * (total_hours - group2_regular_hours)
  )
  let group3_cost := group3_count * (
    group3_rate * group3_regular_hours + group3_flat_rate
  )
  group1_cost + group2_cost + group3_cost

/-- Theorem stating the total cost for the given problem --/
theorem factory_cost : 
  total_cost 500 300 15 8 100 18 10 100 20 8 50 12 = 109200 := by
  sorry

end NUMINAMATH_CALUDE_factory_cost_l3734_373409


namespace NUMINAMATH_CALUDE_hyperbola_condition_ellipse_y_focus_condition_l3734_373488

-- Define the curve C
def C (t : ℝ) := {(x, y) : ℝ × ℝ | x^2 / (5 - t) + y^2 / (t - 1) = 1}

-- Define what it means for C to be a hyperbola
def is_hyperbola (t : ℝ) := (5 - t) * (t - 1) < 0

-- Define what it means for C to be an ellipse with focus on y-axis
def is_ellipse_y_focus (t : ℝ) := t - 1 > 5 - t ∧ t - 1 > 0 ∧ 5 - t > 0

-- Statement 1
theorem hyperbola_condition (t : ℝ) : 
  t < 1 → is_hyperbola t :=
sorry

-- Statement 2
theorem ellipse_y_focus_condition (t : ℝ) :
  is_ellipse_y_focus t → 3 < t ∧ t < 5 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_condition_ellipse_y_focus_condition_l3734_373488


namespace NUMINAMATH_CALUDE_function_composition_fraction_l3734_373418

def f (x : ℝ) : ℝ := 3 * x + 2

def g (x : ℝ) : ℝ := 2 * x - 3

theorem function_composition_fraction :
  f (g (f 3)) / g (f (g 3)) = 59 / 19 := by sorry

end NUMINAMATH_CALUDE_function_composition_fraction_l3734_373418


namespace NUMINAMATH_CALUDE_rectangle_horizontal_length_l3734_373427

/-- Given a square with side length 80 cm and a rectangle with vertical length 100 cm,
    if their perimeters are equal, then the horizontal length of the rectangle is 60 cm. -/
theorem rectangle_horizontal_length (square_side : ℝ) (rect_vertical : ℝ) (rect_horizontal : ℝ) :
  square_side = 80 ∧ rect_vertical = 100 ∧ 4 * square_side = 2 * (rect_vertical + rect_horizontal) →
  rect_horizontal = 60 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_horizontal_length_l3734_373427


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l3734_373411

/-- Given two points A and B symmetric with respect to the y-axis, prove that the sum of their x-coordinates is the negative of the difference of their y-coordinates. -/
theorem symmetric_points_sum (m n : ℝ) : 
  (∃ (A B : ℝ × ℝ), A = (m, -3) ∧ B = (2, n) ∧ 
   (A.1 = -B.1) ∧ (A.2 = B.2)) → 
  m + n = -5 := by
sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l3734_373411


namespace NUMINAMATH_CALUDE_carnival_activity_order_l3734_373420

/- Define the activities -/
inductive Activity
| Dodgeball
| MagicShow
| SingingContest

/- Define the popularity of each activity -/
def popularity : Activity → Rat
| Activity.Dodgeball => 9/24
| Activity.MagicShow => 4/12
| Activity.SingingContest => 1/3

/- Define the ordering of activities based on popularity -/
def more_popular (a b : Activity) : Prop :=
  popularity a > popularity b

/- Theorem statement -/
theorem carnival_activity_order :
  (more_popular Activity.Dodgeball Activity.MagicShow) ∧
  (more_popular Activity.MagicShow Activity.SingingContest) ∨
  (popularity Activity.MagicShow = popularity Activity.SingingContest) :=
sorry

end NUMINAMATH_CALUDE_carnival_activity_order_l3734_373420


namespace NUMINAMATH_CALUDE_lcm_ratio_sum_l3734_373437

theorem lcm_ratio_sum (a b c : ℕ+) : 
  a.val * 3 = b.val * 2 →
  b.val * 7 = c.val * 3 →
  Nat.lcm a.val (Nat.lcm b.val c.val) = 126 →
  a.val + b.val + c.val = 216 := by
sorry

end NUMINAMATH_CALUDE_lcm_ratio_sum_l3734_373437


namespace NUMINAMATH_CALUDE_trig_identity_l3734_373492

theorem trig_identity : 4 * Real.cos (50 * π / 180) - Real.tan (40 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l3734_373492


namespace NUMINAMATH_CALUDE_exists_vertex_with_positive_product_l3734_373431

-- Define a polyhedron type
structure Polyhedron where
  vertices : Finset Nat
  edges : Finset (Nat × Nat)
  marks : (Nat × Nat) → Int
  vertex_count : vertices.card = 101
  edge_marks : ∀ e ∈ edges, marks e = 1 ∨ marks e = -1

-- Define the product of marks at a vertex
def product_at_vertex (p : Polyhedron) (v : Nat) : Int :=
  (p.edges.filter (λ e => e.1 = v ∨ e.2 = v)).prod p.marks

-- Theorem statement
theorem exists_vertex_with_positive_product (p : Polyhedron) :
  ∃ v ∈ p.vertices, product_at_vertex p v = 1 := by
  sorry

end NUMINAMATH_CALUDE_exists_vertex_with_positive_product_l3734_373431


namespace NUMINAMATH_CALUDE_coin_division_problem_l3734_373412

theorem coin_division_problem : ∃ (n : ℕ), 
  (n > 0) ∧ 
  (n % 8 = 6) ∧ 
  (n % 7 = 5) ∧ 
  (n % 9 = 0) ∧ 
  (∀ m : ℕ, m > 0 ∧ m % 8 = 6 ∧ m % 7 = 5 → m ≥ n) :=
by sorry

end NUMINAMATH_CALUDE_coin_division_problem_l3734_373412


namespace NUMINAMATH_CALUDE_quadratic_roots_product_l3734_373469

theorem quadratic_roots_product (x₁ x₂ : ℝ) : 
  x₁^2 - 3*x₁ + 2 = 0 → 
  x₂^2 - 3*x₂ + 2 = 0 → 
  (x₁ + 1) * (x₂ + 1) = 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_product_l3734_373469


namespace NUMINAMATH_CALUDE_distance_between_trees_l3734_373401

/-- Given a yard of length 325 meters with 26 trees planted at equal distances,
    including one tree at each end, the distance between consecutive trees is 13 meters. -/
theorem distance_between_trees (yard_length : ℝ) (num_trees : ℕ) : 
  yard_length = 325 → num_trees = 26 → (yard_length / (num_trees - 1 : ℝ)) = 13 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_trees_l3734_373401


namespace NUMINAMATH_CALUDE_product_of_decimals_l3734_373450

theorem product_of_decimals : (0.7 : ℝ) * 0.3 = 0.21 := by
  sorry

end NUMINAMATH_CALUDE_product_of_decimals_l3734_373450


namespace NUMINAMATH_CALUDE_mary_age_proof_l3734_373407

/-- Mary's age today -/
def mary_age : ℕ := 12

/-- Mary's father's age today -/
def father_age : ℕ := 4 * mary_age

theorem mary_age_proof :
  (father_age = 4 * mary_age) ∧
  (father_age - 3 = 5 * (mary_age - 3)) →
  mary_age = 12 :=
by sorry

end NUMINAMATH_CALUDE_mary_age_proof_l3734_373407


namespace NUMINAMATH_CALUDE_A_intersect_Z_l3734_373494

def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 1}

theorem A_intersect_Z : A ∩ (Set.range (Int.cast : ℤ → ℝ)) = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_A_intersect_Z_l3734_373494


namespace NUMINAMATH_CALUDE_abs_x_squared_minus_x_lt_two_l3734_373419

theorem abs_x_squared_minus_x_lt_two (x : ℝ) :
  |x^2 - x| < 2 ↔ -1 < x ∧ x < 2 := by sorry

end NUMINAMATH_CALUDE_abs_x_squared_minus_x_lt_two_l3734_373419


namespace NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l3734_373414

theorem coefficient_x_squared_in_expansion (x : ℝ) : 
  (Finset.range 6).sum (fun k => Nat.choose 5 k * (2 * x) ^ k * 1 ^ (5 - k)) = 
  1 + 10 * x + 40 * x^2 + 80 * x^3 + 80 * x^4 + 32 * x^5 :=
by sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l3734_373414


namespace NUMINAMATH_CALUDE_school_play_attendance_l3734_373405

theorem school_play_attendance : 
  let num_girls : ℕ := 10
  let num_boys : ℕ := 12
  let family_members_per_kid : ℕ := 3
  let kids_with_stepparent : ℕ := 5
  let kids_with_grandparents : ℕ := 3
  let additional_grandparents_per_kid : ℕ := 2

  (num_girls + num_boys) * family_members_per_kid + 
  kids_with_stepparent + 
  kids_with_grandparents * additional_grandparents_per_kid = 77 :=
by sorry

end NUMINAMATH_CALUDE_school_play_attendance_l3734_373405


namespace NUMINAMATH_CALUDE_ezekiel_shoe_pairs_l3734_373438

/-- The number of shoes in a pair -/
def shoes_per_pair : ℕ := 2

/-- The total number of new shoes Ezekiel has -/
def total_shoes : ℕ := 6

/-- The number of pairs of shoes Ezekiel bought -/
def pairs_bought : ℕ := total_shoes / shoes_per_pair

theorem ezekiel_shoe_pairs : pairs_bought = 3 := by
  sorry

end NUMINAMATH_CALUDE_ezekiel_shoe_pairs_l3734_373438


namespace NUMINAMATH_CALUDE_alex_calculation_l3734_373454

theorem alex_calculation (x : ℝ) : 
  (x / 9 - 21 = 24) → (x * 9 + 21 = 3666) := by
  sorry

end NUMINAMATH_CALUDE_alex_calculation_l3734_373454


namespace NUMINAMATH_CALUDE_custom_mult_identity_l3734_373490

/-- Custom multiplication operation -/
noncomputable def customMult (a b c : ℝ) (x y : ℝ) : ℝ := a * x + b * y + c * x * y

theorem custom_mult_identity {a b c : ℝ} (h1 : customMult a b c 1 2 = 4) (h2 : customMult a b c 2 3 = 6) :
  ∃ m : ℝ, m ≠ 0 ∧ (∀ x : ℝ, customMult a b c x m = x) → m = 5 := by
  sorry


end NUMINAMATH_CALUDE_custom_mult_identity_l3734_373490


namespace NUMINAMATH_CALUDE_derivative_equality_l3734_373410

-- Define the function f
noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x + c

-- Define the derivative of f
noncomputable def f' (a b : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + b

-- Theorem statement
theorem derivative_equality (a b c : ℝ) :
  (f' a b 2 = 2) → (f' a b (-2) = 2) := by
  sorry

end NUMINAMATH_CALUDE_derivative_equality_l3734_373410


namespace NUMINAMATH_CALUDE_correct_calculation_l3734_373459

-- Define variables
variable (x y : ℝ)

-- Theorem statement
theorem correct_calculation : -4 * x * y + 3 * x * y = -x * y := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l3734_373459


namespace NUMINAMATH_CALUDE_sin_cos_identity_l3734_373436

theorem sin_cos_identity : 
  Real.sin (15 * π / 180) * Real.sin (105 * π / 180) - 
  Real.cos (15 * π / 180) * Real.cos (105 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l3734_373436


namespace NUMINAMATH_CALUDE_square_area_to_side_ratio_l3734_373497

theorem square_area_to_side_ratio :
  ∀ (s1 s2 : ℝ), s1 > 0 → s2 > 0 →
  (s1^2 / s2^2 = 243 / 75) →
  ∃ (a b c : ℕ), 
    (s1 / s2 = a * Real.sqrt b / c) ∧
    (a = 9 ∧ b = 1 ∧ c = 5) ∧
    (a + b + c = 15) := by
  sorry

end NUMINAMATH_CALUDE_square_area_to_side_ratio_l3734_373497


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l3734_373449

theorem simplify_trig_expression (α : Real) (h : π / 2 < α ∧ α < π) :
  Real.sqrt ((1 + Real.sin α) / (1 - Real.sin α)) - Real.sqrt ((1 - Real.sin α) / (1 + Real.sin α)) = -2 * Real.tan α := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l3734_373449


namespace NUMINAMATH_CALUDE_range_of_k_l3734_373475

/-- Given a function f(x) = (x^2 + x + 1) / (kx^2 + kx + 1) with domain R,
    the range of k is [0, 4) -/
theorem range_of_k (f : ℝ → ℝ) (k : ℝ) : 
  (∀ x, f x = (x^2 + x + 1) / (k*x^2 + k*x + 1)) → 
  (∀ x, k*x^2 + k*x + 1 ≠ 0) →
  (0 ≤ k ∧ k < 4) :=
sorry

end NUMINAMATH_CALUDE_range_of_k_l3734_373475


namespace NUMINAMATH_CALUDE_total_wheels_is_119_l3734_373421

/-- The total number of wheels in Liam's three garages --/
def total_wheels : ℕ :=
  let first_garage := 
    (3 * 2) + 2 + (6 * 3) + (9 * 1) + (3 * 4)
  let second_garage := 
    (2 * 2) + (1 * 3) + (3 * 1) + (4 * 4) + (1 * 5) + 2
  let third_garage := 
    (3 * 2) + (4 * 3) + 1 + 1 + (2 * 4) + (1 * 5) + 7 - 1
  first_garage + second_garage + third_garage

/-- Theorem stating that the total number of wheels in Liam's three garages is 119 --/
theorem total_wheels_is_119 : total_wheels = 119 := by
  sorry

end NUMINAMATH_CALUDE_total_wheels_is_119_l3734_373421


namespace NUMINAMATH_CALUDE_problem_solution_l3734_373467

def p (m : ℝ) : Prop := ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a ≠ b ∧ 
  ∀ x y : ℝ, x^2 / (4 - m) + y^2 / m = 1 ↔ (x / a)^2 + (y / b)^2 = 1

def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*m*x + 1 > 0

def S (m : ℝ) : Prop := ∃ x : ℝ, m*x^2 + 2*m*x + 2 - m = 0

theorem problem_solution :
  (∀ m : ℝ, S m → (m < 0 ∨ m ≥ 1)) ∧
  (∀ m : ℝ, (p m ∨ q m) ∧ ¬(q m) → 1 ≤ m ∧ m < 2) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l3734_373467


namespace NUMINAMATH_CALUDE_geometric_sequence_iff_square_middle_l3734_373478

/-- A sequence (a, b, c) is geometric if there exists a common ratio r such that b = ar and c = br. -/
def IsGeometricSequence (a b c : ℝ) : Prop :=
  ∃ r : ℝ, b = a * r ∧ c = b * r

/-- The main theorem stating that b² = ac is necessary and sufficient for (a, b, c) to form a geometric sequence. -/
theorem geometric_sequence_iff_square_middle (a b c : ℝ) :
  IsGeometricSequence a b c ↔ b^2 = a * c :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_iff_square_middle_l3734_373478


namespace NUMINAMATH_CALUDE_tabs_remaining_l3734_373496

theorem tabs_remaining (initial_tabs : ℕ) : 
  initial_tabs = 400 → 
  (initial_tabs - (initial_tabs / 4) - 
   ((initial_tabs - (initial_tabs / 4)) * 2 / 5) -
   ((initial_tabs - (initial_tabs / 4) - 
     ((initial_tabs - (initial_tabs / 4)) * 2 / 5)) / 2)) = 90 := by
  sorry

end NUMINAMATH_CALUDE_tabs_remaining_l3734_373496


namespace NUMINAMATH_CALUDE_g_equals_zero_at_negative_one_l3734_373422

/-- The function g(x) as defined in the problem -/
def g (x r : ℝ) : ℝ := 3 * x^3 - 2 * x^2 + 4 * x - 5 + r

/-- Theorem stating that g(-1) = 0 when r = 14 -/
theorem g_equals_zero_at_negative_one (r : ℝ) : g (-1) r = 0 ↔ r = 14 := by sorry

end NUMINAMATH_CALUDE_g_equals_zero_at_negative_one_l3734_373422


namespace NUMINAMATH_CALUDE_least_time_six_horses_meet_l3734_373425

def horse_lap_time (k : ℕ) : ℕ := k + 1

def is_at_start (t : ℕ) (k : ℕ) : Prop :=
  t % (horse_lap_time k) = 0

def at_least_six_at_start (t : ℕ) : Prop :=
  ∃ (s : Finset ℕ), s.card ≥ 6 ∧ s ⊆ Finset.range 8 ∧ ∀ k ∈ s, is_at_start t k

theorem least_time_six_horses_meet :
  ∃ (T : ℕ), T > 0 ∧ at_least_six_at_start T ∧
  ∀ (t : ℕ), t > 0 ∧ t < T → ¬(at_least_six_at_start t) ∧
  T = 420 :=
sorry

end NUMINAMATH_CALUDE_least_time_six_horses_meet_l3734_373425
