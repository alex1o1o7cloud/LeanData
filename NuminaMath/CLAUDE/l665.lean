import Mathlib

namespace cylinder_surface_area_l665_66531

/-- A cylinder with a square axial section of area 4 has a surface area of 6π -/
theorem cylinder_surface_area (r h : Real) : 
  r * h = 2 →  -- axial section is a square
  r * r = 1 →  -- area of square is 4
  2 * Real.pi * r * r + 2 * Real.pi * r * h = 6 * Real.pi := by
  sorry


end cylinder_surface_area_l665_66531


namespace chips_per_console_is_five_l665_66542

/-- The number of computer chips created per day -/
def chips_per_day : ℕ := 467

/-- The number of video game consoles created per day -/
def consoles_per_day : ℕ := 93

/-- The number of computer chips needed per console -/
def chips_per_console : ℕ := chips_per_day / consoles_per_day

theorem chips_per_console_is_five : chips_per_console = 5 := by
  sorry

end chips_per_console_is_five_l665_66542


namespace circle_line_intersection_l665_66538

-- Define the circle equation
def circle_eq (x y a : ℝ) : Prop := x^2 + y^2 + 2*x - 2*y + 2*a = 0

-- Define the line equation
def line_eq (x y : ℝ) : Prop := x + y + 2 = 0

-- Define the intersection of the circle and the line
def intersection (a : ℝ) : Prop := ∃ x y : ℝ, circle_eq x y a ∧ line_eq x y

-- Define the chord length
def chord_length (a : ℝ) : ℝ := 4

-- Theorem statement
theorem circle_line_intersection (a : ℝ) : 
  intersection a ∧ chord_length a = 4 → a = -2 := by
  sorry

end circle_line_intersection_l665_66538


namespace correlation_relationships_l665_66503

/-- Represents a relationship between variables -/
inductive Relationship
  | CubeVolumeEdge
  | PointOnCurve
  | AppleProductionClimate
  | TreeDiameterHeight

/-- Defines whether a relationship is a correlation relationship -/
def isCorrelationRelationship (r : Relationship) : Prop :=
  match r with
  | Relationship.AppleProductionClimate => True
  | Relationship.TreeDiameterHeight => True
  | _ => False

/-- Theorem stating that only AppleProductionClimate and TreeDiameterHeight are correlation relationships -/
theorem correlation_relationships :
  ∀ r : Relationship,
    isCorrelationRelationship r ↔ (r = Relationship.AppleProductionClimate ∨ r = Relationship.TreeDiameterHeight) :=
by sorry

end correlation_relationships_l665_66503


namespace cindy_marbles_problem_l665_66570

def friends_given_marbles (initial_marbles : ℕ) (marbles_per_friend : ℕ) (remaining_marbles_multiplier : ℕ) (final_multiplied_marbles : ℕ) : ℕ :=
  (initial_marbles - final_multiplied_marbles / remaining_marbles_multiplier) / marbles_per_friend

theorem cindy_marbles_problem :
  friends_given_marbles 500 80 4 720 = 4 := by
  sorry

end cindy_marbles_problem_l665_66570


namespace bucket_capacity_l665_66533

theorem bucket_capacity (tank_capacity : ℕ) : ∃ (x : ℕ),
  (18 * x = tank_capacity) ∧
  (216 * 5 = tank_capacity) ∧
  (x = 60) := by
  sorry

end bucket_capacity_l665_66533


namespace rectangle_ellipse_perimeter_l665_66515

/-- Given a rectangle and an ellipse with specific properties, prove that the rectangle's perimeter is 8√1003 -/
theorem rectangle_ellipse_perimeter :
  ∀ (x y a b : ℝ),
    -- Rectangle properties
    x > 0 ∧ y > 0 ∧
    x * y = 2006 ∧
    -- Ellipse properties
    a > 0 ∧ b > 0 ∧
    x + y = 2 * a ∧
    x^2 + y^2 = 4 * (a^2 - b^2) ∧
    π * a * b = 2006 * π →
    -- Conclusion: Perimeter of the rectangle
    2 * (x + y) = 8 * Real.sqrt 1003 := by
  sorry

end rectangle_ellipse_perimeter_l665_66515


namespace M_intersect_N_l665_66523

/-- The set M defined by the condition √x < 4 -/
def M : Set ℝ := {x | Real.sqrt x < 4}

/-- The set N defined by the condition 3x ≥ 1 -/
def N : Set ℝ := {x | 3 * x ≥ 1}

/-- The intersection of sets M and N -/
theorem M_intersect_N : M ∩ N = {x : ℝ | 1/3 ≤ x ∧ x < 16} := by sorry

end M_intersect_N_l665_66523


namespace average_weight_BCDE_l665_66567

/-- Given the weights of individuals A, B, C, D, and E, prove that the average weight of B, C, D, and E is 51 kg. -/
theorem average_weight_BCDE (w_A w_B w_C w_D w_E : ℝ) : 
  (w_A + w_B + w_C) / 3 = 50 →
  (w_A + w_B + w_C + w_D) / 4 = 53 →
  w_E = w_D + 3 →
  w_A = 73 →
  (w_B + w_C + w_D + w_E) / 4 = 51 := by
sorry

end average_weight_BCDE_l665_66567


namespace parabola_directrix_l665_66525

/-- Given a parabola with equation y = 4x^2, its directrix has equation y = -1/16 -/
theorem parabola_directrix (x y : ℝ) : 
  (y = 4 * x^2) → (∃ p : ℝ, p > 0 ∧ y = (1 / (4 * p)) * x^2 ∧ -1 / (4 * p) = -1/16) :=
by sorry

end parabola_directrix_l665_66525


namespace base_representation_of_500_l665_66544

theorem base_representation_of_500 :
  ∃! b : ℕ, 
    b > 1 ∧ 
    (∃ (a₁ a₂ a₃ a₄ a₅ : ℕ), 
      a₁ < b ∧ a₂ < b ∧ a₃ < b ∧ a₄ < b ∧ a₅ < b ∧
      500 = a₁ * b^4 + a₂ * b^3 + a₃ * b^2 + a₄ * b + a₅) ∧
    b^4 ≤ 500 ∧ 
    500 < b^5 :=
by sorry

end base_representation_of_500_l665_66544


namespace stars_per_jar_l665_66507

theorem stars_per_jar (stars_made : ℕ) (bottles_to_fill : ℕ) (stars_to_make : ℕ) : 
  stars_made = 33 →
  bottles_to_fill = 4 →
  stars_to_make = 307 →
  (stars_made + stars_to_make) / bottles_to_fill = 85 :=
by
  sorry

end stars_per_jar_l665_66507


namespace triangle_side_value_l665_66510

-- Define a triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Define the conditions of the problem
def satisfiesConditions (t : Triangle) : Prop :=
  t.b = Real.sqrt 13 ∧
  t.a + t.c = 4 ∧
  (Real.cos t.B) / (Real.cos t.C) = -t.b / (2 * t.a + t.c)

-- State the theorem
theorem triangle_side_value (t : Triangle) (h : satisfiesConditions t) :
  t.a = 1 ∨ t.a = 3 := by
  sorry

end triangle_side_value_l665_66510


namespace semicircle_limit_l665_66562

/-- The limit of the sum of areas of n semicircles constructed on equal parts 
    of a circle's diameter approaches 0 as n approaches infinity. -/
theorem semicircle_limit (D : ℝ) (h : D > 0) : 
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, (π * D^2) / (8 * n) < ε := by
sorry

end semicircle_limit_l665_66562


namespace zhang_hua_cards_l665_66539

-- Define the variables
variable (x y z : ℕ)

-- State the theorem
theorem zhang_hua_cards :
  (Nat.lcm (Nat.lcm x y) z = 60) →
  (Nat.gcd x y = 4) →
  (Nat.gcd y z = 3) →
  (x = 4 ∨ x = 20) :=
by
  sorry

end zhang_hua_cards_l665_66539


namespace typing_time_proportional_l665_66566

/-- Given that 450 characters can be typed in 9 minutes, 
    prove that 1800 characters can be typed in 36 minutes. -/
theorem typing_time_proportional 
  (chars_per_9min : ℕ) 
  (h_chars : chars_per_9min = 450) :
  (1800 : ℝ) / (36 : ℝ) = (chars_per_9min : ℝ) / 9 :=
by sorry

end typing_time_proportional_l665_66566


namespace ellipse_min_major_axis_l665_66555

/-- Given an ellipse where the maximum area of a triangle formed by a point on the ellipse and its two foci is 1, 
    the minimum length of the major axis is 2√2. -/
theorem ellipse_min_major_axis (a b c : ℝ) : 
  (a > 0) → (b > 0) → (c > 0) → 
  (b * c = 1) →  -- maximum triangle area condition
  (a^2 = b^2 + c^2) →  -- ellipse equation
  (2 * a ≥ 2 * Real.sqrt 2) ∧ 
  (∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ b₀ * c₀ = 1 ∧ a₀^2 = b₀^2 + c₀^2 ∧ 2 * a₀ = 2 * Real.sqrt 2) :=
by sorry

end ellipse_min_major_axis_l665_66555


namespace money_spending_l665_66551

theorem money_spending (M : ℚ) : 
  (2 / 7 : ℚ) * M = 500 →
  M = 1750 := by
  sorry

end money_spending_l665_66551


namespace polynomial_division_l665_66532

-- Define the polynomial that can be divided by x^2 + 3x - 4
def is_divisible (a b c : ℝ) : Prop :=
  ∃ (q : ℝ → ℝ), ∀ x, x^3 + a*x^2 + b*x + c = (x^2 + 3*x - 4) * q x

-- Main theorem
theorem polynomial_division (a b c : ℝ) 
  (h : is_divisible a b c) : 
  (4*a + c = 12) ∧ 
  (2*a - 2*b - c = 14) ∧ 
  (∀ (a' b' c' : ℤ), (is_divisible (a' : ℝ) (b' : ℝ) (c' : ℝ)) → 
    c' ≥ a' ∧ a' > 1 → a' = 2 ∧ b' = -7 ∧ c' = 4) :=
by sorry

end polynomial_division_l665_66532


namespace circle_intersection_m_range_l665_66545

theorem circle_intersection_m_range (m : ℝ) : 
  (∃ x y : ℝ, x^2 + y^2 - 2*x + 10*y + 1 = 0 ∧ x^2 + y^2 - 2*x + 2*y - m = 0) →
  -1 < m ∧ m < 79 := by
sorry

end circle_intersection_m_range_l665_66545


namespace smallest_n_for_unique_k_l665_66559

theorem smallest_n_for_unique_k : ∃ (k : ℤ), (9:ℚ)/16 < (1:ℚ)/(1+k) ∧ (1:ℚ)/(1+k) < 7/12 ∧
  ∀ (n : ℕ), n > 0 → n < 1 →
    ¬(∃! (k : ℤ), (9:ℚ)/16 < (n:ℚ)/(n+k) ∧ (n:ℚ)/(n+k) < 7/12) :=
by sorry

end smallest_n_for_unique_k_l665_66559


namespace vector_BC_l665_66595

/-- Given points A(0,1), B(3,2), and vector AC(-4,-3), prove that vector BC is (-7,-4) -/
theorem vector_BC (A B C : ℝ × ℝ) : 
  A = (0, 1) → 
  B = (3, 2) → 
  C.1 - A.1 = -4 → 
  C.2 - A.2 = -3 → 
  (C.1 - B.1, C.2 - B.2) = (-7, -4) := by
sorry


end vector_BC_l665_66595


namespace expression_evaluation_l665_66509

theorem expression_evaluation : (24 * 2 - 6) / ((6 - 2) * 2) = 5.25 := by
  sorry

end expression_evaluation_l665_66509


namespace integer_list_mean_mode_relation_l665_66513

theorem integer_list_mean_mode_relation : 
  ∀ x : ℕ, 
  x ≤ 100 → 
  x > 0 →
  let list := [20, x, x, x, x]
  let mean := (20 + 4 * x) / 5
  let mode := x
  mean = 2 * mode → 
  x = 10 := by
sorry

end integer_list_mean_mode_relation_l665_66513


namespace smallest_c_for_g_range_five_l665_66514

/-- The function g(x) defined in the problem -/
def g (c : ℝ) (x : ℝ) : ℝ := 2 * x^2 - 4 * x + c

/-- Theorem stating that 7 is the smallest value of c such that 5 is in the range of g(x) -/
theorem smallest_c_for_g_range_five :
  ∀ c : ℝ, (∃ x : ℝ, g c x = 5) ↔ c ≥ 7 :=
by sorry

end smallest_c_for_g_range_five_l665_66514


namespace cube_root_equation_solution_l665_66561

theorem cube_root_equation_solution :
  ∃! x : ℝ, (5 + x / 3) ^ (1/3 : ℝ) = -4 :=
by
  use -207
  sorry

end cube_root_equation_solution_l665_66561


namespace complex_real_condition_l665_66556

/-- If z = (2+mi)/(1+i) is a real number and m is a real number, then m = 2 -/
theorem complex_real_condition (m : ℝ) : 
  let z : ℂ := (2 + m * Complex.I) / (1 + Complex.I)
  (z.im = 0) → m = 2 := by
  sorry

end complex_real_condition_l665_66556


namespace fraction_simplification_l665_66550

theorem fraction_simplification (x y z : ℝ) (h : x + y + z ≠ 0) :
  (x^2 + y^2 - z^2 + 2*x*y) / (x^2 + z^2 - y^2 + 2*x*z) = (x + y - z) / (x + z - y) := by
  sorry

end fraction_simplification_l665_66550


namespace triangle_properties_l665_66506

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a = 5 ∧
  t.b^2 + t.c^2 - Real.sqrt 2 * t.b * t.c = 25 ∧
  Real.cos t.B = 3/5

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h : triangle_conditions t) : 
  t.A = Real.pi/4 ∧ t.c = 7 := by
  sorry


end triangle_properties_l665_66506


namespace series_sum_equals_three_l665_66584

theorem series_sum_equals_three (k : ℝ) (h1 : k > 1) 
  (h2 : ∑' n, (5 * n - 1) / k^n = 13/4) : k = 3 := by
  sorry

end series_sum_equals_three_l665_66584


namespace simplify_sqrt_450_l665_66589

theorem simplify_sqrt_450 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end simplify_sqrt_450_l665_66589


namespace perpendicular_planes_parallel_perpendicular_planes_line_l665_66541

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (subset : Line → Plane → Prop)
variable (plane_parallel : Plane → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)
variable (line_perpendicular_plane : Line → Plane → Prop)
variable (intersection : Plane → Plane → Line)

-- Theorem 1
theorem perpendicular_planes_parallel 
  (m n l : Line) (α β : Plane) :
  line_perpendicular_plane l α →
  line_perpendicular_plane m β →
  parallel l m →
  plane_parallel α β := by sorry

-- Theorem 2
theorem perpendicular_planes_line 
  (m n : Line) (α β : Plane) :
  plane_perpendicular α β →
  intersection α β = m →
  subset n β →
  perpendicular n m →
  line_perpendicular_plane n α := by sorry

end perpendicular_planes_parallel_perpendicular_planes_line_l665_66541


namespace line_y_intercept_l665_66518

/-- A straight line in the xy-plane with slope 2 passing through (239, 480) has y-intercept 2 -/
theorem line_y_intercept (m : ℝ) (x₀ y₀ b : ℝ) : 
  m = 2 → 
  x₀ = 239 → 
  y₀ = 480 → 
  y₀ = m * x₀ + b → 
  b = 2 := by sorry

end line_y_intercept_l665_66518


namespace horner_method_f_at_3_l665_66593

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = x^5 + 2x^3 + 3x^2 + x + 1 -/
def f (x : ℝ) : ℝ := x^5 + 2*x^3 + 3*x^2 + x + 1

/-- Coefficients of f(x) in descending order of degree -/
def f_coeffs : List ℝ := [1, 0, 2, 3, 1, 1]

theorem horner_method_f_at_3 :
  horner f_coeffs 3 = 36 := by
  sorry

#eval horner f_coeffs 3  -- This should output 36
#eval f 3  -- This should also output 36

end horner_method_f_at_3_l665_66593


namespace banana_arrangement_count_l665_66587

/-- The number of unique arrangements of the letters in "BANANA" -/
def banana_arrangements : ℕ := 60

/-- The total number of letters in "BANANA" -/
def total_letters : ℕ := 6

/-- The number of A's in "BANANA" -/
def num_a : ℕ := 3

/-- The number of N's in "BANANA" -/
def num_n : ℕ := 2

/-- The number of B's in "BANANA" -/
def num_b : ℕ := 1

/-- Theorem stating that the number of unique arrangements of the letters in "BANANA" is 60 -/
theorem banana_arrangement_count :
  banana_arrangements = (Nat.factorial total_letters) / ((Nat.factorial num_a) * (Nat.factorial num_n)) :=
sorry

end banana_arrangement_count_l665_66587


namespace hidden_dots_on_three_dice_l665_66553

def total_dots_on_die : ℕ := 21

def total_dots_on_three_dice : ℕ := 3 * total_dots_on_die

def visible_faces : List ℕ := [1, 2, 2, 3, 5, 4, 5, 6]

def sum_visible_faces : ℕ := visible_faces.sum

theorem hidden_dots_on_three_dice : 
  total_dots_on_three_dice - sum_visible_faces = 35 := by
  sorry

end hidden_dots_on_three_dice_l665_66553


namespace composition_result_l665_66530

-- Define the functions f and g
def f (x : ℝ) : ℝ := 2 * x + 5
def g (x : ℝ) : ℝ := 3 * x + 4

-- State the theorem
theorem composition_result : f (g (f 3)) = 79 := by
  sorry

end composition_result_l665_66530


namespace five_double_prime_l665_66527

-- Define the prime operation
def prime (q : ℝ) : ℝ := 3 * q - 3

-- Theorem statement
theorem five_double_prime : prime (prime 5) = 33 := by
  sorry

end five_double_prime_l665_66527


namespace salary_increase_percentage_l665_66577

theorem salary_increase_percentage (original_salary : ℝ) (h : original_salary > 0) : 
  let decreased_salary := 0.5 * original_salary
  let final_salary := 0.75 * original_salary
  ∃ P : ℝ, decreased_salary * (1 + P) = final_salary ∧ P = 0.5 :=
by sorry

end salary_increase_percentage_l665_66577


namespace geometric_sequence_problem_l665_66564

theorem geometric_sequence_problem (a : ℝ) : 
  a > 0 ∧ 
  (∃ r : ℝ, 140 * r = a ∧ a * r = 45 / 28) →
  a = 15 := by
sorry

end geometric_sequence_problem_l665_66564


namespace junk_mail_distribution_l665_66598

theorem junk_mail_distribution (blocks : ℕ) (pieces_per_block : ℕ) (h1 : blocks = 4) (h2 : pieces_per_block = 48) :
  blocks * pieces_per_block = 192 := by
  sorry

end junk_mail_distribution_l665_66598


namespace quadratic_factorization_l665_66535

theorem quadratic_factorization (C D : ℤ) :
  (∀ y, 20 * y^2 - 122 * y + 72 = (C * y - 8) * (D * y - 9)) →
  C * D + C = 25 := by
sorry

end quadratic_factorization_l665_66535


namespace binomial_sum_condition_l665_66580

theorem binomial_sum_condition (n : ℕ) (hn : n ≥ 2) :
  (∀ i j : ℕ, 0 ≤ i ∧ i ≤ n ∧ 0 ≤ j ∧ j ≤ n →
    (i + j) % 2 = (Nat.choose n i + Nat.choose n j) % 2) ↔
  ∃ k : ℕ, k ≥ 2 ∧ n = 2^k - 2 :=
by sorry

end binomial_sum_condition_l665_66580


namespace coefficient_x_squared_in_expansion_l665_66504

theorem coefficient_x_squared_in_expansion : 
  ∀ (a₀ a₁ a₂ a₃ a₄ : ℝ), 
  (∀ x : ℝ, (x - 1)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) → 
  a₂ = 6 := by
sorry

end coefficient_x_squared_in_expansion_l665_66504


namespace not_divisible_1998_pow_minus_1_l665_66558

theorem not_divisible_1998_pow_minus_1 (m : ℕ) : ¬(1000^m - 1 ∣ 1998^m - 1) := by
  sorry

end not_divisible_1998_pow_minus_1_l665_66558


namespace rain_probability_weekend_l665_66501

/-- Probability of rain on at least one day during a weekend given specific conditions --/
theorem rain_probability_weekend (p_rain_sat : ℝ) (p_rain_sun_given_rain_sat : ℝ) (p_rain_sun_given_no_rain_sat : ℝ)
  (h1 : p_rain_sat = 0.3)
  (h2 : p_rain_sun_given_rain_sat = 0.7)
  (h3 : p_rain_sun_given_no_rain_sat = 0.6) :
  1 - (1 - p_rain_sat) * (1 - p_rain_sun_given_no_rain_sat) = 0.72 := by
  sorry

#check rain_probability_weekend

end rain_probability_weekend_l665_66501


namespace correct_balloons_given_to_fred_l665_66594

/-- The number of balloons Sam gave to Fred -/
def balloons_given_to_fred (sam_initial : ℝ) (dan : ℝ) (total_after : ℝ) : ℝ :=
  sam_initial - (total_after - dan)

theorem correct_balloons_given_to_fred :
  balloons_given_to_fred 46.0 16.0 52 = 10.0 := by
  sorry

end correct_balloons_given_to_fred_l665_66594


namespace piggy_bank_coins_l665_66520

theorem piggy_bank_coins (nickels : ℕ) (dimes : ℕ) (quarters : ℕ) : 
  dimes = 2 * nickels →
  quarters = dimes / 2 →
  5 * nickels + 10 * dimes + 25 * quarters = 1950 →
  nickels = 39 := by
sorry

end piggy_bank_coins_l665_66520


namespace library_visitors_l665_66502

theorem library_visitors (month_days : ℕ) (non_sunday_visitors : ℕ) (avg_visitors : ℕ) :
  month_days = 30 →
  non_sunday_visitors = 700 →
  avg_visitors = 750 →
  ∃ (sunday_visitors : ℕ),
    (5 * sunday_visitors + 25 * non_sunday_visitors) / month_days = avg_visitors ∧
    sunday_visitors = 1000 :=
by sorry

end library_visitors_l665_66502


namespace regression_y_intercept_l665_66508

/-- Empirical regression equation for height prediction -/
def height_prediction (x : ℝ) (a : ℝ) : ℝ := 3 * x + a

/-- Average height of the 50 classmates -/
def average_height : ℝ := 170

/-- Average shoe size of the 50 classmates -/
def average_shoe_size : ℝ := 40

/-- Theorem stating that the y-intercept (a) of the regression line is 50 -/
theorem regression_y_intercept :
  ∃ (a : ℝ), height_prediction average_shoe_size a = average_height ∧ a = 50 := by
  sorry

end regression_y_intercept_l665_66508


namespace limit_of_exponential_sine_l665_66519

theorem limit_of_exponential_sine (ε : ℝ) (hε : ε > 0) :
  ∃ δ : ℝ, δ > 0 ∧
  ∀ x : ℝ, 0 < |x - 3| ∧ |x - 3| < δ →
    |(2 - x / 3)^(Real.sin (π * x)) - 1| < ε :=
by sorry

end limit_of_exponential_sine_l665_66519


namespace parametric_line_unique_constants_l665_66547

/-- A line passing through two points with given parametric equations -/
structure ParametricLine where
  a : ℝ
  b : ℝ
  passes_through_P : 0 = 0 + a ∧ 2 = (b/2) * 0 + 1
  passes_through_Q : 1 = 1 + a ∧ 3 = (b/2) * 1 + 1

/-- Theorem stating the unique values of a and b for the given line -/
theorem parametric_line_unique_constants (l : ParametricLine) : l.a = -1 ∧ l.b = 2 := by
  sorry

end parametric_line_unique_constants_l665_66547


namespace solution_to_equation_l665_66537

theorem solution_to_equation : 
  {x : ℝ | Real.sqrt ((3 + Real.sqrt 8) ^ x) + Real.sqrt ((3 - Real.sqrt 8) ^ x) = 6} = {2, -2} := by
sorry

end solution_to_equation_l665_66537


namespace bobby_candy_consumption_l665_66590

/-- The number of candies Bobby eats per day during weekdays -/
def weekday_candies : ℕ := 2

/-- The number of candies Bobby eats per day during weekends -/
def weekend_candies : ℕ := 1

/-- The number of weekdays in a week -/
def weekdays : ℕ := 5

/-- The number of weekend days in a week -/
def weekend_days : ℕ := 2

/-- The number of weeks it takes Bobby to finish the packets -/
def weeks : ℕ := 3

/-- The number of packets Bobby buys -/
def packets : ℕ := 2

/-- The number of candies in a packet -/
def candies_per_packet : ℕ := 18

theorem bobby_candy_consumption :
  weekday_candies * weekdays * weeks +
  weekend_candies * weekend_days * weeks =
  candies_per_packet * packets := by
  sorry

end bobby_candy_consumption_l665_66590


namespace min_sum_m_n_l665_66505

theorem min_sum_m_n (m n : ℕ+) (h : 135 * m = n^3) : 
  ∀ (k l : ℕ+), 135 * k = l^3 → m + n ≤ k + l :=
by sorry

end min_sum_m_n_l665_66505


namespace man_walking_time_l665_66569

/-- The man's usual time to cover the distance -/
def usual_time : ℝ := 72

/-- The man's usual speed -/
def usual_speed : ℝ := 1

/-- The factor by which the man's speed is reduced -/
def speed_reduction_factor : ℝ := 0.75

/-- The additional time taken when walking at reduced speed -/
def additional_time : ℝ := 24

theorem man_walking_time :
  (usual_speed * usual_time = speed_reduction_factor * usual_speed * (usual_time + additional_time)) →
  usual_time = 72 := by
  sorry

end man_walking_time_l665_66569


namespace simplify_fraction_l665_66536

theorem simplify_fraction (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 0) :
  (1 - 1 / (x - 1)) / ((x^2 - 2*x) / (x^2 - 1)) = (x + 1) / x :=
by sorry

end simplify_fraction_l665_66536


namespace max_value_of_fraction_l665_66573

theorem max_value_of_fraction (x : ℝ) : (4 * x^2 + 8 * x + 19) / (4 * x^2 + 8 * x + 9) ≤ 3 := by
  sorry

end max_value_of_fraction_l665_66573


namespace veggie_servings_per_week_l665_66548

/-- The number of veggie servings eaten in one week -/
def veggieServingsPerWeek (dailyServings : ℕ) (daysInWeek : ℕ) : ℕ :=
  dailyServings * daysInWeek

/-- Theorem: Given 3 servings daily and 7 days in a week, the total veggie servings per week is 21 -/
theorem veggie_servings_per_week :
  veggieServingsPerWeek 3 7 = 21 := by
  sorry

end veggie_servings_per_week_l665_66548


namespace simplify_expression_1_simplify_expression_2_l665_66549

variable (x y a : ℝ)

theorem simplify_expression_1 : (x + y)^2 + y * (3 * x - y) = x^2 + 5 * x * y := by sorry

theorem simplify_expression_2 (h1 : a ≠ 1) (h2 : a ≠ 4) (h3 : a ≠ -4) :
  ((4 - a^2) / (a - 1) + a) / ((a^2 - 16) / (a - 1)) = -1 / (a + 4) := by sorry

end simplify_expression_1_simplify_expression_2_l665_66549


namespace parallelogram_area_is_288_l665_66521

/-- Represents a parallelogram ABCD -/
structure Parallelogram where
  AB : ℝ
  BC : ℝ
  height : ℝ

/-- Calculates the area of a parallelogram -/
def area (p : Parallelogram) : ℝ := p.AB * p.height

theorem parallelogram_area_is_288 (p : Parallelogram) 
  (h1 : p.AB = 24)
  (h2 : p.BC = 30)
  (h3 : p.height = 12) : 
  area p = 288 := by
  sorry

end parallelogram_area_is_288_l665_66521


namespace greatest_divisor_four_consecutive_integers_l665_66554

theorem greatest_divisor_four_consecutive_integers :
  ∀ n : ℕ, n > 0 →
  ∃ m : ℕ, m > 0 ∧ m ∣ (n * (n + 1) * (n + 2) * (n + 3)) ∧
  ∀ k : ℕ, k > m → ¬(∀ i : ℕ, i > 0 → k ∣ (i * (i + 1) * (i + 2) * (i + 3))) →
  m = 24 :=
by sorry

end greatest_divisor_four_consecutive_integers_l665_66554


namespace white_square_area_l665_66528

theorem white_square_area (cube_edge : ℝ) (blue_paint_area : ℝ) : 
  cube_edge = 12 → 
  blue_paint_area = 432 → 
  (cube_edge ^ 2 * 6 - blue_paint_area) / 6 = 72 := by
  sorry

end white_square_area_l665_66528


namespace gcd_48576_34650_l665_66581

theorem gcd_48576_34650 : Nat.gcd 48576 34650 = 1 := by
  sorry

end gcd_48576_34650_l665_66581


namespace inequality_proof_l665_66529

theorem inequality_proof (a b c : ℝ) (h1 : c > b) (h2 : b > a) :
  a^2*b + b^2*c + c^2*a < a*b^2 + b*c^2 + c*a^2 := by
  sorry

end inequality_proof_l665_66529


namespace hash_difference_l665_66575

-- Define the # operation
def hash (x y : ℤ) : ℤ := x * y - x - 2 * y

-- State the theorem
theorem hash_difference : (hash 6 4) - (hash 4 6) = 2 := by sorry

end hash_difference_l665_66575


namespace division_problem_l665_66522

theorem division_problem (remainder quotient divisor dividend : ℕ) : 
  remainder = 6 →
  divisor = 5 * quotient →
  divisor = 3 * remainder + 2 →
  dividend = divisor * quotient + remainder →
  dividend = 86 := by
sorry

end division_problem_l665_66522


namespace scientific_notation_of_0_0008_l665_66582

theorem scientific_notation_of_0_0008 : ∃ (a : ℝ) (n : ℤ), 
  0.0008 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 8 ∧ n = -4 := by
  sorry

end scientific_notation_of_0_0008_l665_66582


namespace pencil_count_l665_66565

theorem pencil_count (pens pencils : ℕ) : 
  (pens : ℚ) / pencils = 5 / 6 →
  pencils = pens + 8 →
  pencils = 48 :=
by sorry

end pencil_count_l665_66565


namespace largest_divisor_of_cube_divisible_by_127_l665_66597

theorem largest_divisor_of_cube_divisible_by_127 (n : ℕ+) 
  (h : 127 ∣ n^3) : 
  ∀ m : ℕ+, m ∣ n → m ≤ 127 := by
sorry

end largest_divisor_of_cube_divisible_by_127_l665_66597


namespace estimate_excellent_scores_result_l665_66543

/-- Estimates the number of excellent scores in a population based on a sample. -/
def estimate_excellent_scores (total_population : ℕ) (sample_size : ℕ) (excellent_in_sample : ℕ) : ℕ :=
  (total_population * excellent_in_sample) / sample_size

/-- Theorem stating that the estimated number of excellent scores is 152 given the problem conditions. -/
theorem estimate_excellent_scores_result :
  estimate_excellent_scores 380 50 20 = 152 := by
  sorry

end estimate_excellent_scores_result_l665_66543


namespace distribute_5_3_l665_66574

/-- The number of ways to distribute n volunteers to k schools, 
    with each school receiving at least one volunteer -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 5 volunteers to 3 schools, 
    with each school receiving at least one volunteer, is 150 -/
theorem distribute_5_3 : distribute 5 3 = 150 := by sorry

end distribute_5_3_l665_66574


namespace two_number_difference_l665_66512

theorem two_number_difference (a b : ℕ) (h1 : a = 10 * b) (h2 : a + b = 17402) : 
  a - b = 14238 := by
sorry

end two_number_difference_l665_66512


namespace expression_value_l665_66526

theorem expression_value (x y : ℝ) (hx : x = 2) (hy : y = 1) : 2 * x - 3 * y = 1 := by
  sorry

end expression_value_l665_66526


namespace shopping_cart_fruit_ratio_l665_66578

theorem shopping_cart_fruit_ratio :
  ∀ (apples oranges pears : ℕ),
    oranges = 3 * apples →
    apples = (pears : ℚ) * (83333333333333333 : ℚ) / (1000000000000000000 : ℚ) →
    (pears : ℚ) / (oranges : ℚ) = 4 :=
by
  sorry

end shopping_cart_fruit_ratio_l665_66578


namespace marble_theorem_l665_66546

def marble_problem (adam mary greg john sarah peter emily : ℚ) : Prop :=
  adam = 29 ∧
  mary = adam - 11 ∧
  greg = adam + 14 ∧
  john = 2 * mary ∧
  sarah = greg - 7 ∧
  peter = 3 * adam ∧
  emily = (mary + greg) / 2 ∧
  peter + john + sarah - (adam + mary + greg + emily) = 38.5

theorem marble_theorem :
  ∀ adam mary greg john sarah peter emily : ℚ,
  marble_problem adam mary greg john sarah peter emily :=
by
  sorry

end marble_theorem_l665_66546


namespace als_original_portion_l665_66540

theorem als_original_portion (total_initial : ℝ) (total_final : ℝ) (al_loss : ℝ) 
  (h1 : total_initial = 1200)
  (h2 : total_final = 1800)
  (h3 : al_loss = 200) :
  ∃ (al betty clare : ℝ),
    al + betty + clare = total_initial ∧
    al - al_loss + 3 * betty + 3 * clare = total_final ∧
    al = 800 := by
  sorry

end als_original_portion_l665_66540


namespace trigonometric_identities_l665_66511

theorem trigonometric_identities :
  (2 * Real.sin (75 * π / 180) * Real.cos (75 * π / 180) = 1 / 2) ∧
  (Real.cos (45 * π / 180) * Real.cos (15 * π / 180) - Real.sin (45 * π / 180) * Real.sin (15 * π / 180) = 1 / 2) ∧
  ((Real.tan (77 * π / 180) - Real.tan (32 * π / 180)) / (2 * (1 + Real.tan (77 * π / 180) * Real.tan (32 * π / 180))) = 1 / 2) :=
by sorry


end trigonometric_identities_l665_66511


namespace triangle_max_area_l665_66568

theorem triangle_max_area (A B C : ℝ) (a b c : ℝ) : 
  A = (2 * π) / 3 →
  b + 2 * c = 8 →
  0 < a ∧ 0 < b ∧ 0 < c →
  (∀ b' c' : ℝ, b' + 2 * c' = 8 → 
    b' * c' * Real.sin A ≤ b * c * Real.sin A) →
  a^2 = b^2 + c^2 - 2 * b * c * Real.cos A →
  a = 2 * Real.sqrt 7 := by
sorry

end triangle_max_area_l665_66568


namespace base_10_to_7_395_l665_66591

/-- Converts a natural number from base 10 to base 7 -/
def toBase7 (n : ℕ) : List ℕ :=
  sorry

theorem base_10_to_7_395 :
  toBase7 395 = [1, 1, 0, 3] :=
sorry

end base_10_to_7_395_l665_66591


namespace tonys_fever_l665_66576

theorem tonys_fever (normal_temp : ℝ) (temp_increase : ℝ) (fever_threshold : ℝ)
  (h1 : normal_temp = 95)
  (h2 : temp_increase = 10)
  (h3 : fever_threshold = 100) :
  normal_temp + temp_increase - fever_threshold = 5 :=
by sorry

end tonys_fever_l665_66576


namespace intersection_A_complement_B_l665_66500

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}

-- Define set B
def B : Set ℝ := {x | x > 1}

-- Theorem statement
theorem intersection_A_complement_B :
  A ∩ Bᶜ = {x : ℝ | -1 ≤ x ∧ x ≤ 1} := by sorry

end intersection_A_complement_B_l665_66500


namespace probability_at_least_one_defective_l665_66599

/-- The probability of selecting at least one defective item from a set of products -/
theorem probability_at_least_one_defective 
  (total : ℕ) 
  (defective : ℕ) 
  (selected : ℕ) 
  (h1 : total = 10) 
  (h2 : defective = 3) 
  (h3 : selected = 3) :
  (1 : ℚ) - (Nat.choose (total - defective) selected : ℚ) / (Nat.choose total selected : ℚ) = 17/24 := by
  sorry

#check probability_at_least_one_defective

end probability_at_least_one_defective_l665_66599


namespace tablet_savings_l665_66592

/-- The amount saved when buying a tablet from the cheaper store --/
theorem tablet_savings (list_price : ℝ) (tech_discount_percent : ℝ) (electro_discount : ℝ) :
  list_price = 120 →
  tech_discount_percent = 15 →
  electro_discount = 20 →
  list_price * (1 - tech_discount_percent / 100) - (list_price - electro_discount) = 2 :=
by sorry

end tablet_savings_l665_66592


namespace yuna_has_most_apples_l665_66552

def jungkook_apples : ℚ := 6 / 3
def yoongi_apples : ℕ := 4
def yuna_apples : ℕ := 5

theorem yuna_has_most_apples : 
  (jungkook_apples : ℝ) < yuna_apples ∧ yoongi_apples < yuna_apples :=
by sorry

end yuna_has_most_apples_l665_66552


namespace quadratic_inequality_solution_l665_66596

theorem quadratic_inequality_solution (a : ℝ) : 
  (∀ x : ℝ, ax^2 + 8*a*x + 21 < 0 ↔ -7 < x ∧ x < -1) → a = 3 := by
  sorry

end quadratic_inequality_solution_l665_66596


namespace gcd_of_numbers_l665_66588

theorem gcd_of_numbers : Nat.gcd 128 (Nat.gcd 144 (Nat.gcd 480 450)) = 6 := by
  sorry

end gcd_of_numbers_l665_66588


namespace sum_x_y_equals_three_l665_66583

def A (x : ℝ) : Set ℝ := {2, x}
def B (x y : ℝ) : Set ℝ := {x*y, 1}

theorem sum_x_y_equals_three (x y : ℝ) : A x = B x y → x + y = 3 := by
  sorry

end sum_x_y_equals_three_l665_66583


namespace shirt_price_theorem_l665_66524

/-- Represents the problem of determining shirt prices and profits --/
structure ShirtProblem where
  first_batch_cost : ℝ
  second_batch_cost : ℝ
  quantity_ratio : ℝ
  price_difference : ℝ
  discount_quantity : ℕ
  discount_rate : ℝ
  min_profit : ℝ

/-- Calculates the unit price of the first batch --/
def first_batch_unit_price (p : ShirtProblem) : ℝ := 80

/-- Calculates the minimum selling price per shirt --/
def min_selling_price (p : ShirtProblem) : ℝ := 120

/-- Theorem stating the correctness of the calculated prices --/
theorem shirt_price_theorem (p : ShirtProblem) 
  (h1 : p.first_batch_cost = 3200)
  (h2 : p.second_batch_cost = 7200)
  (h3 : p.quantity_ratio = 2)
  (h4 : p.price_difference = 10)
  (h5 : p.discount_quantity = 20)
  (h6 : p.discount_rate = 0.2)
  (h7 : p.min_profit = 3520) :
  first_batch_unit_price p = 80 ∧ 
  min_selling_price p = 120 ∧
  min_selling_price p ≥ (p.min_profit + p.first_batch_cost + p.second_batch_cost) / 
    (p.first_batch_cost / first_batch_unit_price p + 
     p.second_batch_cost / (first_batch_unit_price p + p.price_difference) + 
     p.discount_quantity * (1 - p.discount_rate)) := by
  sorry


end shirt_price_theorem_l665_66524


namespace parabola_shift_through_origin_l665_66571

-- Define the parabola function
def parabola (x : ℝ) : ℝ := (x + 3)^2 - 1

-- Define the shifted parabola function
def shifted_parabola (h : ℝ) (x : ℝ) : ℝ := parabola (x - h)

-- Theorem statement
theorem parabola_shift_through_origin :
  ∀ h : ℝ, shifted_parabola h 0 = 0 ↔ h = 2 ∨ h = 4 := by
  sorry

end parabola_shift_through_origin_l665_66571


namespace boys_count_l665_66517

theorem boys_count (total_students : ℕ) (girls_ratio boys_ratio : ℕ) (h1 : total_students = 30) 
  (h2 : girls_ratio = 1) (h3 : boys_ratio = 2) : 
  (total_students * boys_ratio) / (girls_ratio + boys_ratio) = 20 := by
  sorry

end boys_count_l665_66517


namespace prime_factorization_2020_2021_l665_66563

theorem prime_factorization_2020_2021 : 
  (2020 = 2^2 * 5 * 101) ∧ (2021 = 43 * 47) := by
  sorry

end prime_factorization_2020_2021_l665_66563


namespace quadratic_minimum_l665_66516

theorem quadratic_minimum (x : ℝ) : ∃ m : ℝ, m = 1337 ∧ ∀ x, 5*x^2 - 20*x + 1357 ≥ m := by
  sorry

end quadratic_minimum_l665_66516


namespace no_natural_n_for_sum_of_squares_l665_66579

theorem no_natural_n_for_sum_of_squares : 
  ¬ ∃ (n : ℕ), ∃ (x y : ℕ+), 
    2 * n * (n + 1) * (n + 2) * (n + 3) + 12 = x^2 + y^2 := by
  sorry

end no_natural_n_for_sum_of_squares_l665_66579


namespace intersection_when_m_3_range_of_m_when_subset_l665_66534

-- Define sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.sqrt (3 - 2*x - x^2)}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*x + 1 - m^2 ≤ 0}

-- Theorem for part 1
theorem intersection_when_m_3 :
  A ∩ B 3 = {x | -2 ≤ x ∧ x ≤ 1} := by sorry

-- Theorem for part 2
theorem range_of_m_when_subset (m : ℝ) :
  m > 0 → A ⊆ B m → m ≥ 4 := by sorry

end intersection_when_m_3_range_of_m_when_subset_l665_66534


namespace minimum_order_amount_correct_l665_66572

/-- The minimum order amount to get a discount at Silvia's bakery -/
def minimum_order_amount : ℝ := 60

/-- The discount percentage offered by the bakery -/
def discount_percentage : ℝ := 0.10

/-- The total cost of Silvia's order before discount -/
def order_cost : ℝ := 2 * 15 + 6 * 3 + 6 * 2

/-- The total cost of Silvia's order after discount -/
def discounted_cost : ℝ := 54

/-- Theorem stating that the minimum order amount to get the discount is correct -/
theorem minimum_order_amount_correct :
  minimum_order_amount = order_cost ∧
  discounted_cost = order_cost * (1 - discount_percentage) :=
sorry

end minimum_order_amount_correct_l665_66572


namespace negation_of_rectangle_diagonals_equal_l665_66585

theorem negation_of_rectangle_diagonals_equal :
  let p := "The diagonals of a rectangle are equal"
  ¬p = "The diagonals of a rectangle are not equal" := by
  sorry

end negation_of_rectangle_diagonals_equal_l665_66585


namespace painting_club_teams_l665_66557

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * Nat.factorial (n - k))

theorem painting_club_teams (n : ℕ) (h : n = 7) : 
  choose n 4 * choose (n - 4) 2 = 105 :=
by sorry

end painting_club_teams_l665_66557


namespace solve_xy_l665_66560

def A : Nat := 89252525 -- ... (200-digit number)

def B (x y : Nat) : Nat := 444 * x * 100000 + 18 * 1000 + y * 10 + 27

def digit_from_right (n : Nat) (pos : Nat) : Nat :=
  (n / (10 ^ (pos - 1))) % 10

theorem solve_xy :
  ∀ x y : Nat,
    x < 10 → y < 10 →
    digit_from_right (A * B x y) 53 = 1 →
    digit_from_right (A * B x y) 54 = 0 →
    x = 4 ∧ y = 6 :=
by sorry

end solve_xy_l665_66560


namespace apple_weight_probability_l665_66586

theorem apple_weight_probability (p_less_200 p_more_300 : ℝ) 
  (h1 : p_less_200 = 0.10)
  (h2 : p_more_300 = 0.12) :
  1 - p_less_200 - p_more_300 = 0.78 := by
  sorry

end apple_weight_probability_l665_66586
