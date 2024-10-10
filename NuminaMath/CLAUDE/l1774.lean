import Mathlib

namespace quadratic_minimum_minimizing_x_value_l1774_177498

/-- The quadratic function f(x) = x^2 - 10x + 24 attains its minimum value when x = 5. -/
theorem quadratic_minimum : ∀ x : ℝ, (x^2 - 10*x + 24) ≥ (5^2 - 10*5 + 24) := by
  sorry

/-- The value of x that minimizes the quadratic function f(x) = x^2 - 10x + 24 is 5. -/
theorem minimizing_x_value : ∃! x : ℝ, ∀ y : ℝ, (x^2 - 10*x + 24) ≤ (y^2 - 10*y + 24) := by
  sorry

end quadratic_minimum_minimizing_x_value_l1774_177498


namespace inequality_and_equality_conditions_l1774_177405

/-- A function f that represents the inequality to be proven -/
noncomputable def f (a b : ℝ) : ℝ := sorry

/-- Theorem stating the inequality and equality conditions -/
theorem inequality_and_equality_conditions (a b : ℝ) (ha : a ≥ 3) (hb : b ≥ 3) :
  f a b ≥ 0 ∧ (f a b = 0 ↔ (a = 3 ∧ b ≥ 3) ∨ (b = 3 ∧ a ≥ 3)) := by sorry

end inequality_and_equality_conditions_l1774_177405


namespace angle_triple_complement_l1774_177401

theorem angle_triple_complement (x : ℝ) : x = 3 * (90 - x) → x = 67.5 := by
  sorry

end angle_triple_complement_l1774_177401


namespace inverse_variation_problem_l1774_177477

theorem inverse_variation_problem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : ∃ k : ℝ, k > 0 ∧ ∀ x y, x^3 * y = k) 
  (h4 : 2^3 * 8 = x^3 * 512) : x = 1/2 := by
sorry

end inverse_variation_problem_l1774_177477


namespace quadratic_solution_l1774_177460

theorem quadratic_solution (b : ℝ) : 
  ((-9 : ℝ)^2 + b * (-9 : ℝ) - 36 = 0) → b = 13 := by
  sorry

end quadratic_solution_l1774_177460


namespace course_selection_schemes_l1774_177413

def number_of_courses : ℕ := 7
def courses_to_choose : ℕ := 4

def total_combinations : ℕ := Nat.choose number_of_courses courses_to_choose

def forbidden_combinations : ℕ := Nat.choose (number_of_courses - 2) (courses_to_choose - 2)

theorem course_selection_schemes :
  total_combinations - forbidden_combinations = 25 := by sorry

end course_selection_schemes_l1774_177413


namespace loss_percentage_calculation_l1774_177432

theorem loss_percentage_calculation (cost_price selling_price : ℝ) : 
  cost_price = 1500 →
  selling_price = 1290 →
  (cost_price - selling_price) / cost_price * 100 = 14 := by
sorry

end loss_percentage_calculation_l1774_177432


namespace smallest_n_for_euler_totient_equation_l1774_177475

def euler_totient (n : ℕ) : ℕ := sorry

theorem smallest_n_for_euler_totient_equation : 
  ∀ n : ℕ, n > 0 → euler_totient n = (2^5 * n) / 47 → n ≥ 59895 :=
sorry

end smallest_n_for_euler_totient_equation_l1774_177475


namespace smallest_solution_abs_equation_l1774_177414

theorem smallest_solution_abs_equation :
  ∃ (x : ℝ), x * |x| = 3 * x + 2 ∧
  ∀ (y : ℝ), y * |y| = 3 * y + 2 → x ≤ y :=
by
  use (-2)
  sorry

end smallest_solution_abs_equation_l1774_177414


namespace faucet_fill_time_l1774_177416

/-- Proves that given four faucets can fill a 120-gallon tub in 8 minutes, 
    eight faucets will fill a 30-gallon tub in 60 seconds. -/
theorem faucet_fill_time : 
  ∀ (faucets_1 faucets_2 : ℕ) 
    (tub_1 tub_2 : ℝ) 
    (time_1 : ℝ) 
    (time_2 : ℝ),
  faucets_1 = 4 →
  faucets_2 = 8 →
  tub_1 = 120 →
  tub_2 = 30 →
  time_1 = 8 →
  (faucets_1 : ℝ) * tub_2 * time_1 = faucets_2 * tub_1 * time_2 →
  time_2 = 1 :=
by
  sorry

end faucet_fill_time_l1774_177416


namespace bullet_speed_difference_l1774_177481

/-- The speed difference of a bullet fired from a moving horse with wind assistance -/
theorem bullet_speed_difference
  (horse_speed : ℝ) 
  (bullet_speed : ℝ)
  (wind_speed : ℝ)
  (h1 : horse_speed = 20)
  (h2 : bullet_speed = 400)
  (h3 : wind_speed = 10) :
  (bullet_speed + horse_speed + wind_speed) - (bullet_speed - horse_speed - wind_speed) = 60 := by
  sorry


end bullet_speed_difference_l1774_177481


namespace sophomore_allocation_l1774_177454

theorem sophomore_allocation (total_students : ℕ) (sophomores : ℕ) (total_spots : ℕ) :
  total_students = 800 →
  sophomores = 260 →
  total_spots = 40 →
  (sophomores : ℚ) / total_students * total_spots = 13 := by
  sorry

end sophomore_allocation_l1774_177454


namespace quadratic_always_positive_l1774_177487

theorem quadratic_always_positive 
  (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hseq : b / a = c / b) : 
  ∀ x : ℝ, a * x^2 + b * x + c > 0 := by
  sorry

end quadratic_always_positive_l1774_177487


namespace fraction_of_fraction_of_fraction_quarter_of_fifth_of_sixth_of_120_l1774_177407

theorem fraction_of_fraction_of_fraction (a b c d : ℚ) :
  a * (b * (c * d)) = (a * b * c) * d :=
by sorry

theorem quarter_of_fifth_of_sixth_of_120 :
  (1 / 4 : ℚ) * ((1 / 5 : ℚ) * ((1 / 6 : ℚ) * 120)) = 1 :=
by sorry

end fraction_of_fraction_of_fraction_quarter_of_fifth_of_sixth_of_120_l1774_177407


namespace collinear_points_t_value_l1774_177493

/-- Given three points A, B, and C in a 2D plane, this function checks if they are collinear --/
def are_collinear (A B C : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  let (x₃, y₃) := C
  (y₂ - y₁) * (x₃ - x₁) = (y₃ - y₁) * (x₂ - x₁)

/-- Theorem stating that if points A(1, 2), B(-3, 4), and C(2, t) are collinear, then t = 3/2 --/
theorem collinear_points_t_value :
  ∀ t : ℝ, are_collinear (1, 2) (-3, 4) (2, t) → t = 3/2 :=
by
  sorry

end collinear_points_t_value_l1774_177493


namespace paving_stone_length_l1774_177495

/-- Given a rectangular courtyard and paving stones with specific properties,
    prove that the length of each paving stone is 4 meters. -/
theorem paving_stone_length
  (courtyard_length : ℝ)
  (courtyard_width : ℝ)
  (num_stones : ℕ)
  (stone_width : ℝ)
  (h1 : courtyard_length = 40)
  (h2 : courtyard_width = 20)
  (h3 : num_stones = 100)
  (h4 : stone_width = 2)
  : ∃ (stone_length : ℝ), stone_length = 4 ∧
    courtyard_length * courtyard_width = ↑num_stones * stone_length * stone_width :=
by
  sorry


end paving_stone_length_l1774_177495


namespace triangle_properties_l1774_177441

/-- Given a triangle with sides 8, 15, and 17, prove it's a right triangle
    and find the longest side of a similar triangle with perimeter 160 -/
theorem triangle_properties (a b c : ℝ) (h1 : a = 8) (h2 : b = 15) (h3 : c = 17) :
  (a^2 + b^2 = c^2) ∧ 
  (∃ (x : ℝ), x * (a + b + c) = 160 ∧ x * c = 68) :=
by sorry

end triangle_properties_l1774_177441


namespace expression_equality_1_expression_equality_2_l1774_177452

-- Part 1
theorem expression_equality_1 : 
  2 * Real.sin (45 * π / 180) - (π - Real.sqrt 5) ^ 0 + (1/2)⁻¹ + |Real.sqrt 2 - 1| = 2 * Real.sqrt 2 := by
  sorry

-- Part 2
theorem expression_equality_2 (a b : ℝ) : 
  (2*a + 3*b) * (3*a - 2*b) = 6*a^2 + 5*a*b - 6*b^2 := by
  sorry

end expression_equality_1_expression_equality_2_l1774_177452


namespace circle_tangent_condition_l1774_177447

-- Define a circle using its equation coefficients
structure Circle where
  D : ℝ
  E : ℝ
  F : ℝ

-- Define what it means for a circle to be tangent to the x-axis at the origin
def tangent_to_x_axis_at_origin (c : Circle) : Prop :=
  ∃ (y : ℝ), y ≠ 0 ∧ 0^2 + y^2 + c.D*0 + c.E*y + c.F = 0 ∧
  ∀ (x : ℝ), x ≠ 0 → (∀ (y : ℝ), x^2 + y^2 + c.D*x + c.E*y + c.F ≠ 0)

-- The main theorem
theorem circle_tangent_condition (c : Circle) :
  tangent_to_x_axis_at_origin c ↔ c.E ≠ 0 ∧ c.D = 0 ∧ c.F = 0 := by
  sorry

end circle_tangent_condition_l1774_177447


namespace vehicle_value_last_year_l1774_177404

theorem vehicle_value_last_year 
  (value_this_year : ℝ) 
  (value_ratio : ℝ) 
  (h1 : value_this_year = 16000)
  (h2 : value_ratio = 0.8)
  (h3 : value_this_year = value_ratio * value_last_year) :
  value_last_year = 20000 :=
by
  sorry

end vehicle_value_last_year_l1774_177404


namespace inverse_value_of_symmetrical_function_l1774_177436

-- Define a function that is symmetrical about a point
def SymmetricalAboutPoint (f : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  ∀ x y, f x = y ↔ f (2 * p.1 - x) = 2 * p.2 - y

-- Define the existence of an inverse function
def HasInverse (f : ℝ → ℝ) : Prop :=
  ∃ g : ℝ → ℝ, (∀ x, g (f x) = x) ∧ (∀ y, f (g y) = y)

theorem inverse_value_of_symmetrical_function
  (f : ℝ → ℝ)
  (h_sym : SymmetricalAboutPoint f (1, 2))
  (h_inv : HasInverse f)
  (h_f4 : f 4 = 0) :
  ∃ f_inv : ℝ → ℝ, HasInverse f ∧ f_inv 4 = -2 := by
  sorry

end inverse_value_of_symmetrical_function_l1774_177436


namespace stratified_sampling_admin_count_l1774_177417

theorem stratified_sampling_admin_count 
  (total_employees : ℕ) 
  (admin_employees : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_employees = 120) 
  (h2 : admin_employees = 40) 
  (h3 : sample_size = 24) : 
  ℕ :=
  by
    sorry

#check stratified_sampling_admin_count

end stratified_sampling_admin_count_l1774_177417


namespace bandages_left_in_box_l1774_177412

/-- The number of bandages in a box before use -/
def initial_bandages : ℕ := 24 - 8

/-- The number of bandages used on the left knee -/
def left_knee_bandages : ℕ := 2

/-- The number of bandages used on the right knee -/
def right_knee_bandages : ℕ := 3

/-- The total number of bandages used -/
def total_used_bandages : ℕ := left_knee_bandages + right_knee_bandages

theorem bandages_left_in_box : initial_bandages - total_used_bandages = 11 := by
  sorry

end bandages_left_in_box_l1774_177412


namespace function_properties_l1774_177472

/-- Given a function f with the specified properties, prove its simplified form and extrema -/
theorem function_properties (ω : ℝ) (h_ω_pos : ω > 0) :
  let f : ℝ → ℝ := λ x ↦ 2 * Real.sqrt 3 * Real.cos (ω * x) * Real.sin (ω * x) - 2 * (Real.cos (ω * x))^2 + 1
  (∀ x, f (x + π) = f x) →  -- smallest positive period is π
  (∃ g : ℝ → ℝ, ∀ x, f x = 2 * Real.sin (2 * x - π / 6)) ∧
  (∀ x ∈ Set.Icc 0 (π / 2), f x ≤ 2) ∧
  (∀ x ∈ Set.Icc 0 (π / 2), f x ≥ -1) ∧
  (f 0 = -1) ∧
  (f (π / 3) = 2) := by
  sorry

end function_properties_l1774_177472


namespace sum_of_xyz_l1774_177400

theorem sum_of_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hxy : x * y = 30) (hxz : x * z = 60) (hyz : y * z = 90) :
  x + y + z = 11 * Real.sqrt 5 := by
  sorry

end sum_of_xyz_l1774_177400


namespace simplest_quadratic_radical_sqrt_8_simplification_sqrt_1_3_simplification_sqrt_4_simplification_l1774_177470

-- Define what it means for a quadratic radical to be simplest
def is_simplest_quadratic_radical (x : ℝ) : Prop :=
  ∀ y : ℝ, y > 0 → y ≠ x → (∃ n : ℕ, x = Real.sqrt n) → 
    ¬(∃ a b : ℝ, a > 0 ∧ b > 0 ∧ x = a * Real.sqrt b ∧ b < y)

-- State the theorem
theorem simplest_quadratic_radical :
  is_simplest_quadratic_radical (Real.sqrt 6) ∧
  ¬is_simplest_quadratic_radical (Real.sqrt 8) ∧
  ¬is_simplest_quadratic_radical (Real.sqrt (1/3)) ∧
  ¬is_simplest_quadratic_radical (Real.sqrt 4) :=
by sorry

-- Define the simplification rules
theorem sqrt_8_simplification : Real.sqrt 8 = 2 * Real.sqrt 2 := by sorry
theorem sqrt_1_3_simplification : Real.sqrt (1/3) = Real.sqrt 3 / 3 := by sorry
theorem sqrt_4_simplification : Real.sqrt 4 = 2 := by sorry

end simplest_quadratic_radical_sqrt_8_simplification_sqrt_1_3_simplification_sqrt_4_simplification_l1774_177470


namespace min_distance_theorem_l1774_177433

theorem min_distance_theorem (a b x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  x + y + Real.sqrt ((a - x)^2 + (b - y)^2) ≥ Real.sqrt (a^2 + b^2) := by
  sorry

end min_distance_theorem_l1774_177433


namespace complement_union_theorem_l1774_177445

-- Define the universal set U as the real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x > 1}

-- Define set B
def B : Set ℝ := {x | 0 < x ∧ x < 2}

-- State the theorem
theorem complement_union_theorem : (U \ A) ∪ B = {x : ℝ | x < 2} := by sorry

end complement_union_theorem_l1774_177445


namespace ellipse_to_hyperbola_l1774_177420

/-- Given an ellipse with equation x²/8 + y²/5 = 1 where its foci are its vertices,
    prove that the equation of the hyperbola with foci at the vertices of the ellipse
    is x²/3 - y²/5 = 1 -/
theorem ellipse_to_hyperbola (x y : ℝ) :
  (∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧
    (x^2 / 8 + y^2 / 5 = 1) ∧
    (c^2 = a^2 + b^2) ∧
    (c = 2 * a)) →
  (∃ a' b' c' : ℝ, a' > 0 ∧ b' > 0 ∧ c' > 0 ∧
    (x^2 / 3 - y^2 / 5 = 1) ∧
    (c'^2 = a'^2 + b'^2) ∧
    (c' = 2 * Real.sqrt 2)) :=
by sorry

end ellipse_to_hyperbola_l1774_177420


namespace olivia_picked_16_pieces_l1774_177440

/-- The number of pieces of paper Olivia picked up -/
def olivia_pieces : ℕ := 19 - 3

/-- The number of pieces of paper Edward picked up -/
def edward_pieces : ℕ := 3

/-- The total number of pieces of paper picked up -/
def total_pieces : ℕ := 19

theorem olivia_picked_16_pieces :
  olivia_pieces = 16 ∧ olivia_pieces + edward_pieces = total_pieces :=
sorry

end olivia_picked_16_pieces_l1774_177440


namespace triangle_two_solutions_l1774_177483

theorem triangle_two_solutions (a b : ℝ) (A : ℝ) (ha : a = Real.sqrt 3) (hb : b = 3) (hA : A = π / 6) :
  (b * Real.sin A < a) ∧ (a < b) → ∃ (B C : ℝ), 0 < B ∧ 0 < C ∧ A + B + C = π ∧
  a = b * Real.sin C / Real.sin A ∧ 
  b = a * Real.sin B / Real.sin A :=
sorry

end triangle_two_solutions_l1774_177483


namespace age_puzzle_l1774_177461

/-- The age of a person satisfying a specific age-related equation --/
theorem age_puzzle : ∃ A : ℕ, 5 * (A + 5) - 5 * (A - 5) = A ∧ A = 50 := by
  sorry

end age_puzzle_l1774_177461


namespace hyperbola_condition_l1774_177409

-- Define the equation
def is_hyperbola (k : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (3 - k) + y^2 / (k - 1) = 1

-- Theorem statement
theorem hyperbola_condition (k : ℝ) :
  k > 3 → is_hyperbola k ∧ ¬(∀ k', is_hyperbola k' → k' > 3) :=
by sorry

end hyperbola_condition_l1774_177409


namespace new_men_average_age_l1774_177499

/-- Given a group of 12 men, where replacing two men aged 21 and 23 with two new men
    increases the average age by 1 year, prove that the average age of the two new men is 28 years. -/
theorem new_men_average_age
  (n : ℕ) -- number of men
  (old_age1 old_age2 : ℕ) -- ages of the two replaced men
  (avg_increase : ℚ) -- increase in average age
  (h1 : n = 12)
  (h2 : old_age1 = 21)
  (h3 : old_age2 = 23)
  (h4 : avg_increase = 1) :
  (old_age1 + old_age2 + n * avg_increase) / 2 = 28 :=
by sorry

end new_men_average_age_l1774_177499


namespace average_of_first_n_naturals_l1774_177489

theorem average_of_first_n_naturals (n : ℕ) : 
  (n * (n + 1)) / (2 * n) = 10 → n = 19 := by
  sorry

end average_of_first_n_naturals_l1774_177489


namespace horner_v₂_eq_40_l1774_177465

/-- Horner's method for a polynomial of degree 6 -/
def horner (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) (x : ℝ) : ℝ :=
  a₀ + x * (a₁ + x * (a₂ + x * (a₃ + x * (a₄ + x * (a₅ + x * a₆)))))

/-- The second Horner value for a polynomial of degree 6 -/
def v₂ (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) (x : ℝ) : ℝ :=
  a₂ + x * (a₃ + x * (a₄ + x * (a₅ + x * a₆))) - 
  x * (a₁ + x * (a₂ + x * (a₃ + x * (a₄ + x * (a₅ + x * a₆)))))

theorem horner_v₂_eq_40 :
  v₂ 64 (-192) 240 (-160) 60 (-12) 1 2 = 40 :=
by sorry

end horner_v₂_eq_40_l1774_177465


namespace local_max_at_two_l1774_177488

/-- The function f(x) = x(x-c)² has a local maximum at x=2 if and only if c = 6 -/
theorem local_max_at_two (c : ℝ) : 
  (∃ δ > 0, ∀ x ∈ Set.Ioo (2 - δ) (2 + δ), x * (x - c)^2 ≤ 2 * (2 - c)^2) ↔ c = 6 := by
  sorry

end local_max_at_two_l1774_177488


namespace parabola_coefficients_l1774_177438

/-- A parabola with coefficients a, b, and c in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℚ
  b : ℚ
  c : ℚ

/-- The vertex of a parabola -/
def vertex (p : Parabola) : ℚ × ℚ := sorry

/-- Check if a point lies on the parabola -/
def lies_on (p : Parabola) (x y : ℚ) : Prop :=
  y = p.a * x^2 + p.b * x + p.c

/-- Check if the parabola has a vertical axis of symmetry -/
def has_vertical_axis (p : Parabola) : Prop := sorry

theorem parabola_coefficients :
  ∀ p : Parabola,
    vertex p = (5, -3) →
    has_vertical_axis p →
    lies_on p 2 4 →
    p.a = 7/9 ∧ p.b = -70/9 ∧ p.c = 140/9 := by sorry

end parabola_coefficients_l1774_177438


namespace equation_solution_l1774_177437

theorem equation_solution :
  ∃! y : ℚ, 7 * (4 * y + 3) - 3 = -3 * (2 - 5 * y) ∧ y = -24 / 13 := by
  sorry

end equation_solution_l1774_177437


namespace vegan_meal_clients_l1774_177408

theorem vegan_meal_clients (total : ℕ) (kosher : ℕ) (both : ℕ) (neither : ℕ) :
  total = 30 ∧ kosher = 8 ∧ both = 3 ∧ neither = 18 →
  ∃ vegan : ℕ, vegan = 10 ∧ vegan + (kosher - both) + neither = total :=
by sorry

end vegan_meal_clients_l1774_177408


namespace pet_food_difference_l1774_177425

theorem pet_food_difference (dog_food : ℕ) (cat_food : ℕ) 
  (h1 : dog_food = 600) (h2 : cat_food = 327) : 
  dog_food - cat_food = 273 := by
  sorry

end pet_food_difference_l1774_177425


namespace function_shift_and_overlap_l1774_177455

theorem function_shift_and_overlap (f : ℝ → ℝ) :
  (∀ x, f (x - π / 12) = Real.cos (π / 2 - 2 * x)) →
  (∀ x, f x = Real.sin (2 * x - π / 6)) :=
by sorry

end function_shift_and_overlap_l1774_177455


namespace point_coordinates_l1774_177484

def fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

theorem point_coordinates :
  ∀ (x y : ℝ),
  fourth_quadrant x y →
  |x| = 3 →
  |y| = 5 →
  (x = 3 ∧ y = -5) :=
by sorry

end point_coordinates_l1774_177484


namespace three_digit_numbers_from_five_cards_l1774_177422

theorem three_digit_numbers_from_five_cards : 
  let n : ℕ := 5  -- number of cards
  let r : ℕ := 3  -- number of digits in the formed number
  Nat.factorial n / Nat.factorial (n - r) = 60 := by
  sorry

end three_digit_numbers_from_five_cards_l1774_177422


namespace teapot_teacup_discount_l1774_177459

/-- Represents the payment amount for a purchase of teapots and teacups under different discount methods -/
def payment_amount (x : ℝ) : Prop :=
  let teapot_price : ℝ := 20
  let teacup_price : ℝ := 5
  let num_teapots : ℝ := 4
  let discount_rate : ℝ := 0.92
  let y1 : ℝ := teapot_price * num_teapots + teacup_price * (x - num_teapots)
  let y2 : ℝ := (teapot_price * num_teapots + teacup_price * x) * discount_rate
  (4 ≤ x ∧ x < 34 → y1 < y2) ∧
  (x = 34 → y1 = y2) ∧
  (x > 34 → y1 > y2)

theorem teapot_teacup_discount (x : ℝ) (h : x ≥ 4) : payment_amount x := by
  sorry

end teapot_teacup_discount_l1774_177459


namespace quadratic_equation_solution_l1774_177428

theorem quadratic_equation_solution :
  ∀ x : ℝ, x * (x - 3) = 0 ↔ x = 0 ∨ x = 3 := by
sorry

end quadratic_equation_solution_l1774_177428


namespace seven_digit_divisible_by_13_l1774_177453

def is_valid_number (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ n = 3000000 + 100000 * a + 100 * b + 3

def is_divisible_by_13 (n : ℕ) : Prop :=
  ∃ (k : ℕ), n = 13 * k

theorem seven_digit_divisible_by_13 :
  {n : ℕ | is_valid_number n ∧ is_divisible_by_13 n} =
  {3000803, 3020303, 3030703, 3050203, 3060603, 3080103, 3090503} :=
sorry

end seven_digit_divisible_by_13_l1774_177453


namespace complex_modulus_problem_l1774_177411

theorem complex_modulus_problem (z : ℂ) : 2 + z * Complex.I = z - 2 * Complex.I → Complex.abs z = 2 := by
  sorry

end complex_modulus_problem_l1774_177411


namespace polygon_has_five_sides_l1774_177497

/-- The set T of points (x, y) satisfying the given conditions -/
def T (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let x := p.1; let y := p.2;
    a / 3 ≤ x ∧ x ≤ 5 * a / 2 ∧
    a / 3 ≤ y ∧ y ≤ 5 * a / 2 ∧
    x + y ≥ 3 * a / 2 ∧
    x + 2 * a ≥ 2 * y ∧
    2 * y + 2 * a ≥ 3 * x}

/-- The theorem stating that the polygon formed by T has 5 sides -/
theorem polygon_has_five_sides (a : ℝ) (ha : a > 0) :
  ∃ (vertices : Finset (ℝ × ℝ)), vertices.card = 5 ∧
  (∀ p ∈ T a, p ∈ convexHull ℝ (↑vertices : Set (ℝ × ℝ))) ∧
  (∀ v ∈ vertices, v ∈ T a) :=
sorry

end polygon_has_five_sides_l1774_177497


namespace civilisation_meaning_l1774_177462

/-- The meaning of a word -/
def word_meaning (word : String) : String :=
  sorry

/-- Theorem: The meaning of "civilisation (n.)" is "civilization" -/
theorem civilisation_meaning : word_meaning "civilisation (n.)" = "civilization" :=
  sorry

end civilisation_meaning_l1774_177462


namespace expression_evaluation_l1774_177476

theorem expression_evaluation : -3 * 5 - (-4 * -2) + (-15 * -3) / 3 = -8 := by
  sorry

end expression_evaluation_l1774_177476


namespace min_value_sum_reciprocals_l1774_177450

theorem min_value_sum_reciprocals (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_sum : a + b + c = 4) :
  (1 / a + 4 / b + 9 / c) ≥ 9 ∧ ∃ (a₀ b₀ c₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ 0 < c₀ ∧ a₀ + b₀ + c₀ = 4 ∧ 1 / a₀ + 4 / b₀ + 9 / c₀ = 9 := by
  sorry

end min_value_sum_reciprocals_l1774_177450


namespace inequality_solution_set_l1774_177473

-- Define the inequality
def inequality (x : ℝ) : Prop := (5*x + 3)/(x - 1) ≤ 3

-- Define the solution set
def solution_set : Set ℝ := {x | -3 ≤ x ∧ x < 1}

-- Theorem stating that the solution set is correct
theorem inequality_solution_set : 
  ∀ x : ℝ, x ∈ solution_set ↔ inequality x ∧ x ≠ 1 :=
sorry

end inequality_solution_set_l1774_177473


namespace no_solution_exists_l1774_177491

theorem no_solution_exists : ¬ ∃ (a b c d : ℤ), a^4 + b^4 + c^4 + 2016 = 10*d := by
  sorry

end no_solution_exists_l1774_177491


namespace largest_common_term_l1774_177492

def first_sequence (n : ℕ) : ℕ := 3 + 8 * n

def second_sequence (m : ℕ) : ℕ := 5 + 9 * m

theorem largest_common_term :
  ∃ (n m : ℕ),
    first_sequence n = second_sequence m ∧
    first_sequence n = 131 ∧
    first_sequence n ≤ 150 ∧
    ∀ (k l : ℕ), first_sequence k = second_sequence l → first_sequence k ≤ 150 → first_sequence k ≤ 131 :=
by sorry

end largest_common_term_l1774_177492


namespace car_dealership_ratio_l1774_177410

/-- Given a car dealership with economy cars, luxury cars, and sport utility vehicles,
    where the ratio of economy to luxury cars is 3:2 and the ratio of economy cars
    to sport utility vehicles is 4:1, prove that the ratio of luxury cars to sport
    utility vehicles is 8:3. -/
theorem car_dealership_ratio (E L S : ℚ) 
    (h1 : E / L = 3 / 2)
    (h2 : E / S = 4 / 1) :
    L / S = 8 / 3 := by
  sorry

end car_dealership_ratio_l1774_177410


namespace graduation_chairs_l1774_177446

/-- Calculates the total number of chairs needed for a graduation ceremony. -/
def chairs_needed (graduates : ℕ) (parents_per_graduate : ℕ) (teachers : ℕ) : ℕ :=
  graduates + (graduates * parents_per_graduate) + teachers + (teachers / 2)

/-- Proves that 180 chairs are needed for the given graduation ceremony. -/
theorem graduation_chairs : chairs_needed 50 2 20 = 180 := by
  sorry

end graduation_chairs_l1774_177446


namespace officials_selection_count_l1774_177478

/-- Represents the number of ways to choose officials from a club --/
def choose_officials (total_members : ℕ) (boys : ℕ) (girls : ℕ) : ℕ :=
  girls * boys * (boys - 1)

/-- Theorem: The number of ways to choose officials under given conditions is 1716 --/
theorem officials_selection_count :
  choose_officials 25 12 13 = 1716 := by
  sorry

end officials_selection_count_l1774_177478


namespace smaller_number_l1774_177427

theorem smaller_number (a b d x y : ℝ) : 
  0 < a → a < b → x > 0 → y > 0 → x / y = 2 * a / (3 * b) → x + 2 * y = d →
  min x y = a * d / (2 * a + 3 * b) := by
sorry

end smaller_number_l1774_177427


namespace real_roots_iff_k_le_2_m_eq_3_and_other_root_4_l1774_177485

-- Define the quadratic equation
def quadratic (k : ℝ) (x : ℝ) : Prop := x^2 - 4*x + 2*k = 0

-- Define the condition for real roots
def has_real_roots (k : ℝ) : Prop := ∃ x : ℝ, quadratic k x

-- Define the second quadratic equation
def quadratic2 (m : ℝ) (x : ℝ) : Prop := x^2 - 2*m*x + 3*m - 1 = 0

-- Theorem for part 1
theorem real_roots_iff_k_le_2 :
  ∀ k : ℝ, has_real_roots k ↔ k ≤ 2 :=
sorry

-- Theorem for part 2
theorem m_eq_3_and_other_root_4 :
  ∃ x : ℝ, quadratic 2 x ∧ quadratic2 3 x ∧ quadratic2 3 4 :=
sorry

end real_roots_iff_k_le_2_m_eq_3_and_other_root_4_l1774_177485


namespace circles_tangent_line_parallel_l1774_177419

-- Define the types for points, lines, and circles
variable (Point Line Circle : Type)

-- Define the necessary relations and operations
variable (tangent_circles : Circle → Circle → Prop)
variable (tangent_circle_line : Circle → Line → Point → Prop)
variable (tangent_circles_at : Circle → Circle → Point → Prop)
variable (on_line : Point → Line → Prop)
variable (between : Point → Point → Point → Prop)
variable (intersection : Line → Line → Point)
variable (line_through : Point → Point → Line)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem circles_tangent_line_parallel 
  (Γ Γ₁ Γ₂ : Circle) (l : Line) 
  (A A₁ A₂ B₁ B₂ C D₁ D₂ : Point) :
  tangent_circles Γ Γ₁ →
  tangent_circles Γ Γ₂ →
  tangent_circles Γ₁ Γ₂ →
  tangent_circle_line Γ l A →
  tangent_circle_line Γ₁ l A₁ →
  tangent_circle_line Γ₂ l A₂ →
  tangent_circles_at Γ Γ₁ B₁ →
  tangent_circles_at Γ Γ₂ B₂ →
  tangent_circles_at Γ₁ Γ₂ C →
  between A₁ A A₂ →
  D₁ = intersection (line_through A₁ C) (line_through A₂ B₂) →
  D₂ = intersection (line_through A₂ C) (line_through A₁ B₁) →
  parallel (line_through D₁ D₂) l :=
by sorry

end circles_tangent_line_parallel_l1774_177419


namespace modulus_of_z_l1774_177486

theorem modulus_of_z (z : ℂ) (h : z * (1 - Complex.I) = 2 + 4 * Complex.I) : 
  Complex.abs z = Real.sqrt 10 := by
  sorry

end modulus_of_z_l1774_177486


namespace sum_of_coefficients_l1774_177426

theorem sum_of_coefficients (a b c : ℕ+) : 
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ 
    (Real.sqrt 3 + 1 / Real.sqrt 3 + Real.sqrt 11 + 1 / Real.sqrt 11 = (a * Real.sqrt 3 + b * Real.sqrt 11) / c) ∧
    (∀ (d e f : ℕ+), (Real.sqrt 3 + 1 / Real.sqrt 3 + Real.sqrt 11 + 1 / Real.sqrt 11 = (d * Real.sqrt 3 + e * Real.sqrt 11) / f) → c ≤ f)) →
  a + b + c = 113 := by
sorry

end sum_of_coefficients_l1774_177426


namespace quiz_competition_participants_l1774_177469

theorem quiz_competition_participants (initial_participants : ℕ) : 
  (initial_participants : ℝ) * 0.4 * 0.25 = 15 → initial_participants = 150 :=
by
  sorry

end quiz_competition_participants_l1774_177469


namespace divisibility_by_twelve_l1774_177444

theorem divisibility_by_twelve (n : Nat) : n < 10 → (3150 + n) % 12 = 0 ↔ n = 6 := by
  sorry

end divisibility_by_twelve_l1774_177444


namespace complex_quotient_real_l1774_177494

/-- Given complex numbers Z₁ and Z₂, where Z₁ = a + 2i and Z₂ = 3 - 4i,
    if Z₁/Z₂ is a real number, then a = -3/2 -/
theorem complex_quotient_real (a : ℝ) :
  let Z₁ : ℂ := a + 2*I
  let Z₂ : ℂ := 3 - 4*I
  (∃ (r : ℝ), Z₁ / Z₂ = r) → a = -3/2 := by
  sorry

end complex_quotient_real_l1774_177494


namespace magazine_subscription_l1774_177448

theorem magazine_subscription (total_students : ℕ) 
  (boys_first_half : ℕ) (girls_first_half : ℕ)
  (boys_second_half : ℕ) (girls_second_half : ℕ)
  (boys_whole_year : ℕ) 
  (h1 : total_students = 56)
  (h2 : boys_first_half = 25)
  (h3 : girls_first_half = 15)
  (h4 : boys_second_half = 26)
  (h5 : girls_second_half = 25)
  (h6 : boys_whole_year = 23) :
  girls_first_half - (girls_first_half + girls_second_half - (total_students - (boys_first_half + boys_second_half - boys_whole_year))) = 3 := by
  sorry

end magazine_subscription_l1774_177448


namespace min_intersection_size_l1774_177434

theorem min_intersection_size (total students_green_eyes students_own_lunch : ℕ)
  (h_total : total = 25)
  (h_green : students_green_eyes = 15)
  (h_lunch : students_own_lunch = 18)
  : ∃ (intersection : ℕ), 
    intersection ≤ students_green_eyes ∧ 
    intersection ≤ students_own_lunch ∧
    intersection ≥ students_green_eyes + students_own_lunch - total ∧
    intersection = 8 := by
  sorry

end min_intersection_size_l1774_177434


namespace daria_concert_money_l1774_177451

theorem daria_concert_money (total_tickets : ℕ) (ticket_cost : ℕ) (current_savings : ℕ) :
  total_tickets = 4 →
  ticket_cost = 90 →
  current_savings = 189 →
  total_tickets * ticket_cost - current_savings = 171 :=
by sorry

end daria_concert_money_l1774_177451


namespace p_and_q_true_iff_not_p_or_not_q_false_l1774_177466

theorem p_and_q_true_iff_not_p_or_not_q_false (p q : Prop) :
  (p ∧ q) ↔ ¬(¬p ∨ ¬q) := by
  sorry

end p_and_q_true_iff_not_p_or_not_q_false_l1774_177466


namespace quadratic_inequality_solution_l1774_177418

theorem quadratic_inequality_solution (x : ℝ) :
  3 * x^2 + 9 * x + 6 ≤ 0 ↔ -2 ≤ x ∧ x ≤ -1 := by
  sorry

end quadratic_inequality_solution_l1774_177418


namespace other_root_is_one_l1774_177443

theorem other_root_is_one (a : ℝ) : 
  (2^2 - a*2 + 2 = 0) → 
  ∃ x, x ≠ 2 ∧ x^2 - a*x + 2 = 0 ∧ x = 1 := by
sorry

end other_root_is_one_l1774_177443


namespace prime_sum_and_squares_l1774_177435

theorem prime_sum_and_squares (p q r s : ℕ) : 
  p.Prime ∧ q.Prime ∧ r.Prime ∧ s.Prime ∧  -- p, q, r, s are prime
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s ∧  -- p, q, r, s are distinct
  (p + q + r + s).Prime ∧  -- their sum is prime
  ∃ a, p^2 + q*s = a^2 ∧  -- p² + qs is a perfect square
  ∃ b, p^2 + q*r = b^2  -- p² + qr is a perfect square
  →
  ((p = 2 ∧ q = 7 ∧ r = 11 ∧ s = 3) ∨ (p = 2 ∧ q = 7 ∧ r = 3 ∧ s = 11)) :=
by sorry

end prime_sum_and_squares_l1774_177435


namespace product_equality_l1774_177415

theorem product_equality : 100 * 29.98 * 2.998 * 1000 = (2998 : ℝ) ^ 2 := by
  sorry

end product_equality_l1774_177415


namespace intersection_point_on_lines_unique_intersection_l1774_177482

/-- The intersection point of two lines -/
def intersection_point : ℚ × ℚ := (39/17, 53/17)

/-- First line equation: 8x - 3y = 9 -/
def line1 (x y : ℚ) : Prop := 8*x - 3*y = 9

/-- Second line equation: 6x + 2y = 20 -/
def line2 (x y : ℚ) : Prop := 6*x + 2*y = 20

/-- Theorem stating that the intersection_point satisfies both line equations -/
theorem intersection_point_on_lines : 
  let (x, y) := intersection_point
  line1 x y ∧ line2 x y :=
by sorry

/-- Theorem stating that the intersection_point is the unique solution -/
theorem unique_intersection :
  ∀ (x y : ℚ), line1 x y ∧ line2 x y → (x, y) = intersection_point :=
by sorry

end intersection_point_on_lines_unique_intersection_l1774_177482


namespace root_sum_ratio_l1774_177471

theorem root_sum_ratio (k₁ k₂ : ℝ) : 
  (∃ a b : ℝ, (k₁ * (a^2 - a) + a + 7 = 0 ∧ k₂ * (b^2 - b) + b + 7 = 0) ∧
              (a / b + b / a = 5 / 6)) →
  k₁ / k₂ + k₂ / k₁ = 433 / 36 := by
sorry

end root_sum_ratio_l1774_177471


namespace paradise_park_ferris_wheel_large_seat_capacity_l1774_177474

/-- Represents a Ferris wheel with small and large seats -/
structure FerrisWheel where
  small_seats : Nat
  large_seats : Nat
  small_seat_capacity : Nat
  large_seat_capacity : Nat

/-- Calculates the total number of people who can ride on large seats -/
def large_seat_capacity (fw : FerrisWheel) : Nat :=
  fw.large_seats * fw.large_seat_capacity

/-- Theorem stating that the capacity of large seats in the given Ferris wheel is 84 -/
theorem paradise_park_ferris_wheel_large_seat_capacity :
  let fw := FerrisWheel.mk 3 7 16 12
  large_seat_capacity fw = 84 := by
  sorry

end paradise_park_ferris_wheel_large_seat_capacity_l1774_177474


namespace log_equation_solution_l1774_177430

theorem log_equation_solution (y : ℝ) (h : y > 0) :
  Real.log y / Real.log 3 + Real.log y / Real.log 9 = 5 → y = 3^(10/3) := by
  sorry

end log_equation_solution_l1774_177430


namespace value_added_to_reach_new_average_l1774_177490

theorem value_added_to_reach_new_average (n : ℕ) (initial_avg final_avg : ℝ) (h1 : n = 15) (h2 : initial_avg = 40) (h3 : final_avg = 55) :
  ∃ x : ℝ, (n : ℝ) * initial_avg + n * x = n * final_avg ∧ x = 15 :=
by sorry

end value_added_to_reach_new_average_l1774_177490


namespace initial_crayons_l1774_177458

theorem initial_crayons (C : ℕ) : 
  (C : ℚ) * (3/4) * (1/2) = 18 → C = 48 := by
  sorry

end initial_crayons_l1774_177458


namespace problem_solution_l1774_177456

theorem problem_solution (x : ℝ) : (400 * 7000 : ℝ) = 28000 * (100 ^ x) → x = 1 := by
  sorry

end problem_solution_l1774_177456


namespace monica_cookies_problem_l1774_177406

/-- The number of cookies Monica's father ate -/
def father_cookies : ℕ := 6

/-- The total number of cookies Monica made -/
def total_cookies : ℕ := 30

/-- The number of cookies Monica has left -/
def remaining_cookies : ℕ := 8

theorem monica_cookies_problem :
  (∃ (f : ℕ),
    f = father_cookies ∧
    total_cookies = f + (f / 2) + (f / 2 + 2) + remaining_cookies) :=
by
  sorry

end monica_cookies_problem_l1774_177406


namespace workday_meeting_percentage_l1774_177429

-- Define the workday duration in minutes
def workday_minutes : ℕ := 10 * 60

-- Define the duration of the first meeting
def first_meeting_duration : ℕ := 30

-- Define the duration of the second meeting
def second_meeting_duration : ℕ := 2 * first_meeting_duration

-- Define the duration of the third meeting
def third_meeting_duration : ℕ := first_meeting_duration + second_meeting_duration

-- Define the total time spent in meetings
def total_meeting_time : ℕ := first_meeting_duration + second_meeting_duration + third_meeting_duration

-- Theorem to prove
theorem workday_meeting_percentage : 
  (total_meeting_time : ℚ) / workday_minutes * 100 = 30 := by sorry

end workday_meeting_percentage_l1774_177429


namespace planes_intersect_necessary_not_sufficient_for_skew_lines_l1774_177464

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- Define the intersection relation between two planes
variable (intersect : Plane → Plane → Prop)

-- Define the skew relation between two lines
variable (skew : Line → Line → Prop)

-- Define the theorem
theorem planes_intersect_necessary_not_sufficient_for_skew_lines
  (α β : Plane) (m n : Line)
  (distinct_planes : α ≠ β)
  (m_perp_α : perpendicular m α)
  (n_perp_β : perpendicular n β) :
  (∀ m n, skew m n → intersect α β) ∧
  ¬(∀ m n, intersect α β → skew m n) :=
sorry

end planes_intersect_necessary_not_sufficient_for_skew_lines_l1774_177464


namespace optimal_allocation_l1774_177423

/-- Represents the farming problem with given conditions -/
structure FarmingProblem where
  totalLand : ℝ
  riceYield : ℝ
  peanutYield : ℝ
  riceCost : ℝ
  peanutCost : ℝ
  ricePrice : ℝ
  peanutPrice : ℝ
  availableInvestment : ℝ

/-- Calculates the profit for a given allocation of land -/
def profit (p : FarmingProblem) (riceLand : ℝ) (peanutLand : ℝ) : ℝ :=
  (p.ricePrice * p.riceYield - p.riceCost) * riceLand +
  (p.peanutPrice * p.peanutYield - p.peanutCost) * peanutLand

/-- Checks if a land allocation is valid according to the problem constraints -/
def isValidAllocation (p : FarmingProblem) (riceLand : ℝ) (peanutLand : ℝ) : Prop :=
  riceLand ≥ 0 ∧ peanutLand ≥ 0 ∧
  riceLand + peanutLand ≤ p.totalLand ∧
  p.riceCost * riceLand + p.peanutCost * peanutLand ≤ p.availableInvestment

/-- The main theorem stating that the given allocation maximizes profit -/
theorem optimal_allocation (p : FarmingProblem) 
  (h : p = {
    totalLand := 2,
    riceYield := 6000,
    peanutYield := 1500,
    riceCost := 3600,
    peanutCost := 1200,
    ricePrice := 3,
    peanutPrice := 5,
    availableInvestment := 6000
  }) :
  ∀ x y, isValidAllocation p x y → profit p x y ≤ profit p (3/2) (1/2) :=
sorry


end optimal_allocation_l1774_177423


namespace kathleen_bottle_caps_l1774_177479

/-- The number of times Kathleen went to the store last month -/
def store_visits : ℕ := 5

/-- The number of bottle caps Kathleen buys each time she goes to the store -/
def bottle_caps_per_visit : ℕ := 5

/-- The total number of bottle caps Kathleen bought last month -/
def total_bottle_caps : ℕ := store_visits * bottle_caps_per_visit

theorem kathleen_bottle_caps : total_bottle_caps = 25 := by
  sorry

end kathleen_bottle_caps_l1774_177479


namespace student_number_problem_l1774_177480

theorem student_number_problem : ∃ x : ℤ, 2 * x - 138 = 110 ∧ x = 124 := by
  sorry

end student_number_problem_l1774_177480


namespace problem_1_problem_2_problem_3_l1774_177442

/-- Sequence sum -/
def S (n : ℕ) : ℕ := n^2

/-- Main sequence -/
def a (n : ℕ) : ℝ := 2*n - 1

/-- Arithmetic subsequence -/
def isArithmeticSubsequence (f : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, f (n + 1) - f n = d

/-- Geometric subsequence -/
def isGeometricSubsequence (f : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, f (n + 1) = q * f n

/-- Arithmetic sequence -/
def isArithmeticSequence (f : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, f (n + 1) - f n = d

/-- Geometric sequence -/
def isGeometricSequence (f : ℕ → ℝ) (q : ℝ) : Prop :=
  q ≠ 1 ∧ ∀ n : ℕ, f (n + 1) = q * f n

theorem problem_1 : isArithmeticSubsequence (λ n => a (3*n)) := by sorry

theorem problem_2 (a : ℕ → ℤ) (d : ℤ) (h1 : d ≠ 0) (h2 : isArithmeticSequence a d) 
  (h3 : a 5 = 6) (h4 : isGeometricSubsequence (λ n => a (2*n + 1))) :
  ∃ n1 : ℕ, n1 ∈ ({6, 8, 11} : Set ℕ) := by sorry

theorem problem_3 (a : ℕ → ℝ) (q : ℝ) (h1 : isGeometricSequence a q) 
  (h2 : ∃ f : ℕ → ℕ, Infinite {n : ℕ | n ∈ Set.range f} ∧ isArithmeticSubsequence (λ n => a (f n))) :
  q = -1 := by sorry

end problem_1_problem_2_problem_3_l1774_177442


namespace even_function_symmetric_about_y_axis_l1774_177421

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

theorem even_function_symmetric_about_y_axis (f : ℝ → ℝ) (h : even_function f) :
  ∀ x y : ℝ, f x = y ↔ f (-x) = y :=
by sorry

end even_function_symmetric_about_y_axis_l1774_177421


namespace wendy_albums_l1774_177463

theorem wendy_albums (total_pictures : ℕ) (pictures_in_one_album : ℕ) (pictures_per_album : ℕ) 
  (h1 : total_pictures = 45)
  (h2 : pictures_in_one_album = 27)
  (h3 : pictures_per_album = 2) :
  (total_pictures - pictures_in_one_album) / pictures_per_album = 9 := by
  sorry

end wendy_albums_l1774_177463


namespace factorization_sum_l1774_177457

theorem factorization_sum (a b c d e f g : ℤ) :
  (∀ x y : ℝ, 16 * x^4 - 81 * y^4 = (a * x + b * y) * (c * x^2 + d * x * y + e * y^2) * (f * x + g * y)) →
  a + b + c + d + e + f + g = 17 := by
  sorry

end factorization_sum_l1774_177457


namespace sand_weight_l1774_177496

/-- Given the total weight of materials and the weight of gravel, 
    calculate the weight of sand -/
theorem sand_weight (total_weight gravel_weight : ℝ) 
  (h1 : total_weight = 14.02)
  (h2 : gravel_weight = 5.91) : 
  total_weight - gravel_weight = 8.11 := by
  sorry

end sand_weight_l1774_177496


namespace mars_inhabitable_area_l1774_177439

/-- The fraction of Mars' surface that is not covered by water -/
def mars_land_fraction : ℚ := 3/5

/-- The fraction of Mars' land that is inhabitable -/
def mars_inhabitable_land_fraction : ℚ := 2/3

/-- The fraction of Mars' surface that Martians can inhabit -/
def mars_inhabitable_fraction : ℚ := mars_land_fraction * mars_inhabitable_land_fraction

theorem mars_inhabitable_area :
  mars_inhabitable_fraction = 2/5 := by
  sorry

end mars_inhabitable_area_l1774_177439


namespace lyle_notebook_cost_l1774_177467

/-- The cost of a pen in dollars -/
def pen_cost : ℝ := 1.50

/-- The cost of a notebook in dollars -/
def notebook_cost : ℝ := 3 * pen_cost

/-- The number of notebooks Lyle wants to buy -/
def num_notebooks : ℕ := 4

/-- The total cost of notebooks Lyle will pay -/
def total_cost : ℝ := num_notebooks * notebook_cost

theorem lyle_notebook_cost : total_cost = 18 := by
  sorry

end lyle_notebook_cost_l1774_177467


namespace unique_solution_condition_l1774_177402

theorem unique_solution_condition (k : ℝ) : 
  (∃! x : ℝ, (3*x + 5)*(x - 6) = -53 + k*x + x^2) ↔ (k = -1 ∨ k = -25) := by
  sorry

end unique_solution_condition_l1774_177402


namespace rectangular_field_fence_l1774_177449

theorem rectangular_field_fence (L W : ℝ) : 
  L > 0 ∧ W > 0 →  -- Positive dimensions
  L * W = 210 →    -- Area condition
  L + 2 * W = 41 → -- Fencing condition
  L = 21 :=        -- Conclusion: uncovered side length
by sorry

end rectangular_field_fence_l1774_177449


namespace wedding_attendance_l1774_177431

theorem wedding_attendance (actual_attendance : ℕ) (show_up_rate : ℚ) : 
  actual_attendance = 209 → show_up_rate = 95/100 → 
  ∃ expected_attendance : ℕ, expected_attendance = 220 ∧ 
  (↑actual_attendance : ℚ) = show_up_rate * expected_attendance := by
sorry

end wedding_attendance_l1774_177431


namespace derivative_f_at_one_l1774_177424

noncomputable def f (x : ℝ) : ℝ := x^3 + Real.log x

theorem derivative_f_at_one :
  deriv f 1 = 4 := by sorry

end derivative_f_at_one_l1774_177424


namespace arrangements_count_l1774_177468

/-- Represents the number of teachers -/
def num_teachers : ℕ := 5

/-- Represents the number of days -/
def num_days : ℕ := 4

/-- Represents the number of teachers required on Monday -/
def teachers_on_monday : ℕ := 2

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

/-- Calculates the number of arrangements for the given scenario -/
def num_arrangements : ℕ := 
  (choose num_teachers teachers_on_monday) * (Nat.factorial (num_teachers - teachers_on_monday))

/-- Theorem stating that the number of arrangements is 60 -/
theorem arrangements_count : num_arrangements = 60 := by
  sorry

end arrangements_count_l1774_177468


namespace circular_arcs_in_regular_ngon_l1774_177403

/-- A regular n-gon -/
structure RegularNGon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ
  is_regular : sorry

/-- A point inside a regular n-gon -/
def PointInside (E : RegularNGon n) (P : ℝ × ℝ) : Prop := sorry

/-- A circular arc inside a regular n-gon -/
def CircularArcInside (E : RegularNGon n) (arc : ℝ × ℝ → ℝ × ℝ → Prop) : Prop := sorry

/-- The angle between two circular arcs at their intersection point -/
def AngleBetweenArcs (arc1 arc2 : ℝ × ℝ → ℝ × ℝ → Prop) (P : ℝ × ℝ) : ℝ := sorry

theorem circular_arcs_in_regular_ngon (n : ℕ) (E : RegularNGon n) (P₁ P₂ : ℝ × ℝ) 
  (h₁ : PointInside E P₁) (h₂ : PointInside E P₂) :
  ∃ (arc1 arc2 : ℝ × ℝ → ℝ × ℝ → Prop),
    CircularArcInside E arc1 ∧ 
    CircularArcInside E arc2 ∧
    arc1 P₁ P₂ ∧ 
    arc2 P₁ P₂ ∧
    AngleBetweenArcs arc1 arc2 P₁ ≥ (1 - 2 / n) * π ∧
    AngleBetweenArcs arc1 arc2 P₂ ≥ (1 - 2 / n) * π :=
sorry

end circular_arcs_in_regular_ngon_l1774_177403
