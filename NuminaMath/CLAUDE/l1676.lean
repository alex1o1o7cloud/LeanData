import Mathlib

namespace quadratic_root_difference_l1676_167665

theorem quadratic_root_difference : 
  let a : ℝ := 5 + 3 * Real.sqrt 2
  let b : ℝ := 2 + Real.sqrt 2
  let c : ℝ := -1
  let discriminant := b^2 - 4*a*c
  let root_difference := (2 * Real.sqrt discriminant) / (2 * a)
  root_difference = (2 * Real.sqrt (24 * Real.sqrt 2 + 180)) / 7 := by
sorry

end quadratic_root_difference_l1676_167665


namespace leak_drains_in_26_hours_l1676_167632

/-- Represents the time it takes for a leak to drain a tank, given the fill times with and without the leak -/
def leak_drain_time (pump_fill_time leak_fill_time : ℚ) : ℚ :=
  let pump_rate := 1 / pump_fill_time
  let combined_rate := 1 / leak_fill_time
  let leak_rate := pump_rate - combined_rate
  1 / leak_rate

/-- Theorem stating that given the specific fill times, the leak drains the tank in 26 hours -/
theorem leak_drains_in_26_hours :
  leak_drain_time 2 (13/6) = 26 := by sorry

end leak_drains_in_26_hours_l1676_167632


namespace matrix_A_properties_l1676_167663

/-- The line l: 2x - y = 3 -/
def line_l (x y : ℝ) : Prop := 2 * x - y = 3

/-- The transformation matrix A -/
def matrix_A : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![-1, 1],
    ![-4, 3]]

/-- The inverse of matrix A -/
def matrix_A_inv : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![3, -1],
    ![4, -1]]

/-- The transformation σ maps the line l onto itself -/
def transformation_preserves_line (A : Matrix (Fin 2) (Fin 2) ℝ) : Prop :=
  ∀ x y : ℝ, line_l x y → line_l (A 0 0 * x + A 0 1 * y) (A 1 0 * x + A 1 1 * y)

theorem matrix_A_properties :
  transformation_preserves_line matrix_A ∧
  matrix_A * matrix_A_inv = 1 ∧
  matrix_A_inv * matrix_A = 1 := by
  sorry

end matrix_A_properties_l1676_167663


namespace min_perimeter_isosceles_triangle_l1676_167625

/-- Represents a triangle with integer side lengths where two sides are equal -/
structure IsoscelesTriangle where
  a : ℕ  -- length of BC
  b : ℕ  -- length of AB and AC
  ab_eq_ac : b = b  -- AB = AC

/-- Represents the geometric configuration described in the problem -/
structure GeometricConfiguration (t : IsoscelesTriangle) where
  ω_center_is_incenter : Bool
  excircle_bc_internal : Bool
  excircle_ab_external : Bool
  excircle_ac_not_tangent : Bool

/-- The theorem statement -/
theorem min_perimeter_isosceles_triangle 
  (t : IsoscelesTriangle) 
  (config : GeometricConfiguration t) : 
  2 * t.b + t.a ≥ 20 := by
  sorry

#check min_perimeter_isosceles_triangle

end min_perimeter_isosceles_triangle_l1676_167625


namespace fraction_multiplication_l1676_167613

theorem fraction_multiplication : (1 : ℚ) / 2 * 3 / 5 * 7 / 11 = 21 / 110 := by sorry

end fraction_multiplication_l1676_167613


namespace original_proposition_converse_is_false_inverse_is_false_contrapositive_is_true_l1676_167656

-- Original proposition
theorem original_proposition (a b : ℝ) : a = b → a^2 = b^2 := by sorry

-- Converse is false
theorem converse_is_false : ¬ (∀ a b : ℝ, a^2 = b^2 → a = b) := by sorry

-- Inverse is false
theorem inverse_is_false : ¬ (∀ a b : ℝ, a ≠ b → a^2 ≠ b^2) := by sorry

-- Contrapositive is true
theorem contrapositive_is_true : ∀ a b : ℝ, a^2 ≠ b^2 → a ≠ b := by sorry

end original_proposition_converse_is_false_inverse_is_false_contrapositive_is_true_l1676_167656


namespace stating_equation_satisfied_l1676_167684

/-- 
Represents a number system with base X.
-/
structure BaseX where
  X : ℕ
  X_ge_two : X ≥ 2

/-- 
Represents a digit in the number system with base X.
-/
def Digit (b : BaseX) := {d : ℕ // d < b.X}

/--
Converts a number represented by digits in base X to its decimal value.
-/
def to_decimal (b : BaseX) (digits : List (Digit b)) : ℕ :=
  digits.foldr (fun d acc => acc * b.X + d.val) 0

/--
Theorem stating that the equation ABBC * CCA = CCCCAC is satisfied
in any base X ≥ 2 when A = 1, B = 0, and C = X - 1 or C = 1.
-/
theorem equation_satisfied (b : BaseX) :
  let A : Digit b := ⟨1, by sorry⟩
  let B : Digit b := ⟨0, by sorry⟩
  let C₁ : Digit b := ⟨b.X - 1, by sorry⟩
  let C₂ : Digit b := ⟨1, by sorry⟩
  (to_decimal b [A, B, B, C₁] * to_decimal b [C₁, C₁, A] = to_decimal b [C₁, C₁, C₁, C₁, A, C₁]) ∧
  (to_decimal b [A, B, B, C₂] * to_decimal b [C₂, C₂, A] = to_decimal b [C₂, C₂, C₂, C₂, A, C₂]) :=
by
  sorry


end stating_equation_satisfied_l1676_167684


namespace student_representatives_distribution_l1676_167693

theorem student_representatives_distribution (n m : ℕ) : 
  n = 6 ∧ m = 4 → (Nat.choose (n + m - 2) (m - 1) = Nat.choose 5 3) := by
  sorry

end student_representatives_distribution_l1676_167693


namespace second_chapter_pages_l1676_167678

/-- A book with three chapters -/
structure Book where
  chapter1 : ℕ
  chapter2 : ℕ
  chapter3 : ℕ

/-- The book satisfies the given conditions -/
def satisfiesConditions (b : Book) : Prop :=
  b.chapter1 = 35 ∧ b.chapter3 = 3 ∧ b.chapter2 = b.chapter3 + 15

theorem second_chapter_pages (b : Book) (h : satisfiesConditions b) : b.chapter2 = 18 := by
  sorry

end second_chapter_pages_l1676_167678


namespace trigonometric_relation_and_triangle_property_l1676_167648

theorem trigonometric_relation_and_triangle_property (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (x * Real.sin (π/5) + y * Real.cos (π/5)) / (x * Real.cos (π/5) - y * Real.sin (π/5)) = Real.tan (9*π/20)) :
  ∃ (A B C : ℝ), 
    (y / x = (Real.tan (9*π/20) * Real.cos (π/5) - Real.sin (π/5)) / (Real.cos (π/5) + Real.tan (9*π/20) * Real.sin (π/5))) ∧
    (Real.tan C = y / x) ∧
    (∀ A' B' : ℝ, Real.sin (2*A') + 2 * Real.cos B' ≤ B) :=
by sorry

end trigonometric_relation_and_triangle_property_l1676_167648


namespace log_sqrt10_1000sqrt10_eq_7_l1676_167624

theorem log_sqrt10_1000sqrt10_eq_7 : Real.log (1000 * Real.sqrt 10) / Real.log (Real.sqrt 10) = 7 := by
  sorry

end log_sqrt10_1000sqrt10_eq_7_l1676_167624


namespace rectangle_to_circle_area_l1676_167620

/-- Given a rectangle with area 200 and length twice its width, 
    the area of the largest circle that can be formed from a string 
    equal to the rectangle's perimeter is 900/π. -/
theorem rectangle_to_circle_area (w : ℝ) (h1 : w > 0) : 
  let l := 2 * w
  let area_rect := w * l
  let perimeter := 2 * (w + l)
  area_rect = 200 → (perimeter^2) / (4 * π) = 900 / π := by
  sorry

end rectangle_to_circle_area_l1676_167620


namespace largest_n_for_rational_sum_of_roots_l1676_167695

theorem largest_n_for_rational_sum_of_roots : 
  ∀ n : ℕ, n > 2501 → ¬(∃ (q : ℚ), q = Real.sqrt (n - 100) + Real.sqrt (n + 100)) := by
  sorry

end largest_n_for_rational_sum_of_roots_l1676_167695


namespace black_card_fraction_l1676_167680

theorem black_card_fraction (total : ℕ) (red_fraction : ℚ) (green : ℕ) : 
  total = 120 → 
  red_fraction = 2 / 5 → 
  green = 32 → 
  (5 : ℚ) / 9 = (total - (red_fraction * total) - green) / (total - (red_fraction * total)) := by
  sorry

end black_card_fraction_l1676_167680


namespace arithmetic_progression_reciprocal_l1676_167683

/-- If a, b, and c form an arithmetic progression, and their reciprocals also form an arithmetic progression, then a = b = c. -/
theorem arithmetic_progression_reciprocal (a b c : ℝ) 
  (h1 : b - a = c - b)  -- a, b, c form an arithmetic progression
  (h2 : 1/b - 1/a = 1/c - 1/b)  -- reciprocals form an arithmetic progression
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) : a = b ∧ b = c :=
by sorry

end arithmetic_progression_reciprocal_l1676_167683


namespace quadratic_inequality_range_l1676_167697

theorem quadratic_inequality_range (m : ℝ) : 
  (∀ x : ℝ, x > 1 ∧ x < 2 → x^2 + m*x + 4 < 0) ↔ m ≤ -5 :=
by sorry

end quadratic_inequality_range_l1676_167697


namespace average_price_is_52_cents_l1676_167638

/-- Represents the fruit selection problem --/
structure FruitSelection where
  apple_price : ℚ
  orange_price : ℚ
  total_fruits : ℕ
  initial_avg_price : ℚ
  oranges_returned : ℕ

/-- Calculates the average price of fruits kept --/
def average_price_kept (fs : FruitSelection) : ℚ :=
  let apples := fs.total_fruits - (fs.initial_avg_price * fs.total_fruits - fs.apple_price * fs.total_fruits) / (fs.orange_price - fs.apple_price)
  let oranges := fs.total_fruits - apples
  let kept_oranges := oranges - fs.oranges_returned
  let total_kept := apples + kept_oranges
  (fs.apple_price * apples + fs.orange_price * kept_oranges) / total_kept

/-- Theorem stating that the average price of fruits kept is 52 cents --/
theorem average_price_is_52_cents (fs : FruitSelection) 
    (h1 : fs.apple_price = 40/100)
    (h2 : fs.orange_price = 60/100)
    (h3 : fs.total_fruits = 30)
    (h4 : fs.initial_avg_price = 56/100)
    (h5 : fs.oranges_returned = 15) :
  average_price_kept fs = 52/100 := by
  sorry

#eval average_price_kept {
  apple_price := 40/100,
  orange_price := 60/100,
  total_fruits := 30,
  initial_avg_price := 56/100,
  oranges_returned := 15
}

end average_price_is_52_cents_l1676_167638


namespace ellipse_and_tangent_circle_l1676_167643

/-- Definition of the ellipse C -/
def ellipse_C (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

/-- Definition of the circle tangent to line l -/
def tangent_circle (x y : ℝ) : Prop :=
  (x + 1)^2 + y^2 = 4 / 3

/-- Theorem statement -/
theorem ellipse_and_tangent_circle :
  ∀ (x y : ℝ),
  -- Conditions
  (∃ (a b : ℝ), a > b ∧ b > 0 ∧ a^2 - b^2 = 1) →  -- Ellipse properties
  (1^2 / 4 + (3/2)^2 / 3 = 1) →  -- Point (1, 3/2) lies on C
  (∃ (m : ℝ), m^2 = 2) →  -- Slope of line l
  -- Conclusions
  (ellipse_C x y ↔ x^2 / 4 + y^2 / 3 = 1) ∧
  (tangent_circle x y ↔ (x + 1)^2 + y^2 = 4 / 3) :=
sorry

end ellipse_and_tangent_circle_l1676_167643


namespace green_or_yellow_marble_probability_l1676_167647

/-- The probability of drawing a green or yellow marble from a bag -/
theorem green_or_yellow_marble_probability
  (green : ℕ) (yellow : ℕ) (white : ℕ)
  (h_green : green = 4)
  (h_yellow : yellow = 3)
  (h_white : white = 6) :
  (green + yellow) / (green + yellow + white) = 7 / 13 :=
by sorry

end green_or_yellow_marble_probability_l1676_167647


namespace complex_square_value_l1676_167667

theorem complex_square_value (m n : ℝ) (h : m * (1 + Complex.I) = 1 + n * Complex.I) :
  ((m + n * Complex.I) / (m - n * Complex.I)) ^ 2 = -1 := by
  sorry

end complex_square_value_l1676_167667


namespace paper_length_is_correct_l1676_167616

/-- The length of a rectangular sheet of paper satisfying given conditions -/
def paper_length : ℚ :=
  let width : ℚ := 9
  let second_sheet_length : ℚ := 11
  let second_sheet_width : ℚ := 9/2
  let area_difference : ℚ := 100
  (2 * second_sheet_length * second_sheet_width + area_difference) / (2 * width)

theorem paper_length_is_correct :
  let width : ℚ := 9
  let second_sheet_length : ℚ := 11
  let second_sheet_width : ℚ := 9/2
  let area_difference : ℚ := 100
  2 * paper_length * width = 2 * second_sheet_length * second_sheet_width + area_difference :=
by
  sorry

#eval paper_length

end paper_length_is_correct_l1676_167616


namespace jason_arm_tattoos_count_l1676_167608

-- Define the number of tattoos Jason has on each arm
def jason_arm_tattoos : ℕ := sorry

-- Define the number of tattoos Jason has on each leg
def jason_leg_tattoos : ℕ := 3

-- Define the total number of tattoos Jason has
def jason_total_tattoos : ℕ := 2 * jason_arm_tattoos + 2 * jason_leg_tattoos

-- Define the number of tattoos Adam has
def adam_tattoos : ℕ := 23

-- Theorem to prove
theorem jason_arm_tattoos_count :
  jason_arm_tattoos = 2 ∧
  adam_tattoos = 2 * jason_total_tattoos + 3 :=
by sorry

end jason_arm_tattoos_count_l1676_167608


namespace car_distance_problem_l1676_167602

/-- Represents the distance traveled by a car given its speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

theorem car_distance_problem (speed_x speed_y : ℝ) (initial_time : ℝ) :
  speed_x = 35 →
  speed_y = 65 →
  initial_time = 72 / 60 →
  ∃ t : ℝ, 
    distance speed_y t = distance speed_x initial_time + distance speed_x t ∧
    distance speed_x t = 49 := by
  sorry

#check car_distance_problem

end car_distance_problem_l1676_167602


namespace train_length_l1676_167677

/-- Given a train with speed 72 km/hr crossing a 250 m long platform in 26 seconds,
    prove that the length of the train is 270 meters. -/
theorem train_length (train_speed : ℝ) (platform_length : ℝ) (crossing_time : ℝ) :
  train_speed = 72 * (1000 / 3600) → 
  platform_length = 250 →
  crossing_time = 26 →
  (train_speed * crossing_time) - platform_length = 270 := by
  sorry

#eval (72 * (1000 / 3600) * 26) - 250  -- Should output 270

end train_length_l1676_167677


namespace matt_twice_james_age_l1676_167685

/-- 
Given:
- James turned 27 three years ago
- Matt is now 65 years old

Prove that in 5 years, Matt will be twice James' age.
-/
theorem matt_twice_james_age (james_age_three_years_ago : ℕ) (matt_current_age : ℕ) :
  james_age_three_years_ago = 27 →
  matt_current_age = 65 →
  ∃ (years_from_now : ℕ), 
    years_from_now = 5 ∧
    matt_current_age + years_from_now = 2 * (james_age_three_years_ago + 3 + years_from_now) :=
by sorry

end matt_twice_james_age_l1676_167685


namespace unique_modulo_congruence_l1676_167669

theorem unique_modulo_congruence : ∃! n : ℤ, 0 ≤ n ∧ n ≤ 10 ∧ n ≡ 99999 [ZMOD 11] ∧ n = 9 := by
  sorry

end unique_modulo_congruence_l1676_167669


namespace wiper_line_to_surface_l1676_167628

/-- A car wiper blade modeled as a line -/
structure WiperBlade :=
  (length : ℝ)

/-- A windshield modeled as a surface -/
structure Windshield :=
  (width : ℝ)
  (height : ℝ)

/-- The area swept by a wiper blade on a windshield -/
def swept_area (blade : WiperBlade) (shield : Windshield) : ℝ :=
  blade.length * shield.width

/-- Theorem stating that a car wiper on a windshield represents a line moving into a surface -/
theorem wiper_line_to_surface (blade : WiperBlade) (shield : Windshield) :
  ∃ (area : ℝ), area = swept_area blade shield ∧ area > 0 :=
sorry

end wiper_line_to_surface_l1676_167628


namespace production_problem_l1676_167674

/-- Calculates the production for today given the number of past days, 
    past average production, and new average including today's production. -/
def todaysProduction (n : ℕ) (pastAvg newAvg : ℚ) : ℚ :=
  (n + 1) * newAvg - n * pastAvg

/-- Proves that given the conditions, today's production is 105 units. -/
theorem production_problem (n : ℕ) (pastAvg newAvg : ℚ) 
  (h1 : n = 10)
  (h2 : pastAvg = 50)
  (h3 : newAvg = 55) :
  todaysProduction n pastAvg newAvg = 105 := by
  sorry

#eval todaysProduction 10 50 55

end production_problem_l1676_167674


namespace rotation_of_point_A_l1676_167682

/-- Represents a 2D point or vector -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Rotates a point clockwise by π/2 around the origin -/
def rotate_clockwise_90 (p : Point2D) : Point2D :=
  ⟨p.y, -p.x⟩

theorem rotation_of_point_A : 
  let A : Point2D := ⟨2, 1⟩
  let B : Point2D := rotate_clockwise_90 A
  B.x = 1 ∧ B.y = -2 := by
  sorry

end rotation_of_point_A_l1676_167682


namespace largest_binomial_equality_l1676_167614

theorem largest_binomial_equality : ∃ n : ℕ, (n ≤ 11 ∧ Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 n ∧ ∀ m : ℕ, m ≤ 11 → Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 m → m ≤ n) :=
by sorry

end largest_binomial_equality_l1676_167614


namespace minimal_ratio_S₁_S₂_l1676_167605

noncomputable def S₁ (α : Real) : Real :=
  4 - (2 * Real.sqrt 2 / Real.cos α)

noncomputable def S₂ (α : Real) : Real :=
  ((Real.sqrt 2 * (Real.sin α + Real.cos α) - 1)^2) / (2 * Real.sin α * Real.cos α)

theorem minimal_ratio_S₁_S₂ :
  ∃ (α₁ α₂ : Real), 
    0 ≤ α₁ ∧ α₁ ≤ Real.pi/12 ∧
    Real.pi/12 ≤ α₂ ∧ α₂ ≤ 5*Real.pi/12 ∧
    S₁ α₁ / (8 - S₁ α₁) = 1/7 ∧
    S₂ α₂ / (8 - S₂ α₂) = 1/7 ∧
    ∀ (β γ : Real), 
      (0 ≤ β ∧ β ≤ Real.pi/12 → S₁ β / (8 - S₁ β) ≥ 1/7) ∧
      (Real.pi/12 ≤ γ ∧ γ ≤ 5*Real.pi/12 → S₂ γ / (8 - S₂ γ) ≥ 1/7) :=
by sorry

end minimal_ratio_S₁_S₂_l1676_167605


namespace marbles_distribution_l1676_167634

/-- Given a total number of marbles and a number of groups, 
    calculates the number of marbles in each group -/
def marbles_per_group (total_marbles : ℕ) (num_groups : ℕ) : ℕ :=
  total_marbles / num_groups

/-- Proves that given 20 marbles and 5 groups, there are 4 marbles in each group -/
theorem marbles_distribution :
  marbles_per_group 20 5 = 4 := by
  sorry

end marbles_distribution_l1676_167634


namespace evaluate_expression_l1676_167696

theorem evaluate_expression (x y z : ℚ) 
  (hx : x = 1/4) (hy : y = 4/5) (hz : z = -2) : 
  x^3 * y^2 * z^2 = 1/25 := by sorry

end evaluate_expression_l1676_167696


namespace parabola_coefficient_l1676_167668

/-- Given a parabola y = ax^2 + bx + c with vertex (h, k) and passing through (0, -k) where k ≠ 0,
    prove that b = 4k/h -/
theorem parabola_coefficient (a b c h k : ℝ) (hk : k ≠ 0) :
  (∀ x, a * x^2 + b * x + c = a * (x - h)^2 + k) →
  a * 0^2 + b * 0 + c = -k →
  b = 4 * k / h := by sorry

end parabola_coefficient_l1676_167668


namespace geometric_sequence_general_term_l1676_167642

/-- A geometric sequence with its sum and common ratio -/
structure GeometricSequence where
  a : ℕ+ → ℝ
  S : ℕ+ → ℝ
  q : ℝ
  sum_formula : ∀ n : ℕ+, S n = (a 1) * (1 - q^n.val) / (1 - q)
  term_formula : ∀ n : ℕ+, a n = (a 1) * q^(n.val - 1)

/-- The theorem stating the general term of the specific geometric sequence -/
theorem geometric_sequence_general_term 
  (seq : GeometricSequence) 
  (h1 : seq.S 3 = 14) 
  (h2 : seq.q = 2) :
  ∀ n : ℕ+, seq.a n = 2^n.val :=
sorry

end geometric_sequence_general_term_l1676_167642


namespace sequence_gcd_property_l1676_167609

theorem sequence_gcd_property (a : ℕ → ℕ) :
  (∀ i j : ℕ, i ≠ j → Nat.gcd (a i) (a j) = Nat.gcd i j) →
  ∀ i : ℕ, a i = i := by
  sorry

end sequence_gcd_property_l1676_167609


namespace largest_square_tile_l1676_167654

theorem largest_square_tile (wall_width wall_length : ℕ) 
  (h1 : wall_width = 120) (h2 : wall_length = 96) : 
  Nat.gcd wall_width wall_length = 24 := by
  sorry

end largest_square_tile_l1676_167654


namespace solve_equation_one_solve_equation_two_l1676_167659

-- First equation: 3x = 2x + 12
theorem solve_equation_one : ∃ x : ℝ, 3 * x = 2 * x + 12 ∧ x = 12 := by sorry

-- Second equation: x/2 - 3 = 5
theorem solve_equation_two : ∃ x : ℝ, x / 2 - 3 = 5 ∧ x = 16 := by sorry

end solve_equation_one_solve_equation_two_l1676_167659


namespace janice_throw_ratio_l1676_167681

/-- The height of Christine's first throw in feet -/
def christine_first : ℕ := 20

/-- The height of Janice's first throw in feet -/
def janice_first : ℕ := christine_first - 4

/-- The height of Christine's second throw in feet -/
def christine_second : ℕ := christine_first + 10

/-- The height of Christine's third throw in feet -/
def christine_third : ℕ := christine_second + 4

/-- The height of Janice's third throw in feet -/
def janice_third : ℕ := christine_first + 17

/-- The height of the highest throw in feet -/
def highest_throw : ℕ := 37

/-- The height of Janice's second throw in feet -/
def janice_second : ℕ := 2 * janice_first

theorem janice_throw_ratio :
  janice_second = 2 * janice_first ∧
  janice_third = highest_throw ∧
  janice_second < christine_third ∧
  janice_second > janice_first :=
by sorry

#check janice_throw_ratio

end janice_throw_ratio_l1676_167681


namespace power_equation_equality_l1676_167631

theorem power_equation_equality : 4^3 - 8 = 5^2 + 31 := by
  sorry

end power_equation_equality_l1676_167631


namespace functional_equation_solution_l1676_167650

/-- A function satisfying the given functional equation -/
def SatisfiesFunctionalEq (f : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ), f (f x + 9 * y) = f y + 9 * x + 24 * y

/-- The main theorem stating that any function satisfying the functional equation must be f(x) = 3x -/
theorem functional_equation_solution :
  ∀ (f : ℝ → ℝ), SatisfiesFunctionalEq f → (∀ x, f x = 3 * x) :=
by sorry

end functional_equation_solution_l1676_167650


namespace equation_represents_two_lines_l1676_167607

-- Define the equation
def equation (x y : ℝ) : Prop := x^2 - y^2 = 0

-- Define what it means to be a straight line
def is_straight_line (f : ℝ → ℝ) : Prop :=
  ∃ m b : ℝ, ∀ x : ℝ, f x = m * x + b

-- Theorem statement
theorem equation_represents_two_lines :
  ∃ f g : ℝ → ℝ, 
    (is_straight_line f ∧ is_straight_line g) ∧
    (∀ x y : ℝ, equation x y ↔ (y = f x ∨ y = g x)) :=
sorry

end equation_represents_two_lines_l1676_167607


namespace initial_odometer_reading_l1676_167601

/-- Calculates the initial odometer reading before a trip -/
theorem initial_odometer_reading
  (odometer_at_lunch : ℝ)
  (distance_traveled : ℝ)
  (h1 : odometer_at_lunch = 372)
  (h2 : distance_traveled = 159.7) :
  odometer_at_lunch - distance_traveled = 212.3 := by
sorry

end initial_odometer_reading_l1676_167601


namespace minimize_quadratic_l1676_167641

/-- The quadratic function f(x) = x^2 + 8x + 7 -/
def f (x : ℝ) : ℝ := x^2 + 8*x + 7

/-- Theorem stating that -4 minimizes the quadratic function f(x) = x^2 + 8x + 7 for all real x -/
theorem minimize_quadratic :
  ∀ x : ℝ, f (-4) ≤ f x :=
by sorry

end minimize_quadratic_l1676_167641


namespace range_of_t_l1676_167645

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_odd : ∀ x, f (-x) = -f x
axiom f_increasing : ∀ x y, x < y → x ∈ [-1, 1] → y ∈ [-1, 1] → f x < f y
axiom f_inequality : ∀ t, f (3*t) + f ((1/3) - t) > 0

-- Define the set of t that satisfies the conditions
def T : Set ℝ := {t | -1/6 < t ∧ t ≤ 1/3}

-- Theorem to prove
theorem range_of_t : ∀ t, (f (3*t) + f ((1/3) - t) > 0) ↔ t ∈ T := by sorry

end range_of_t_l1676_167645


namespace instantaneous_velocity_at_one_l1676_167612

-- Define the distance function
def S (t : ℝ) : ℝ := t^3 - 2

-- State the theorem
theorem instantaneous_velocity_at_one (t : ℝ) : 
  (deriv S) 1 = 3 := by sorry

end instantaneous_velocity_at_one_l1676_167612


namespace intersection_exists_l1676_167619

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.log (x - 2)}
def B : Set ℝ := {y | ∃ x, y = 2^x}

-- State the theorem
theorem intersection_exists : ∃ z, z ∈ A ∩ B := by
  sorry

end intersection_exists_l1676_167619


namespace negation_equivalence_l1676_167657

theorem negation_equivalence : 
  (¬ ∃ x₀ : ℝ, x₀ > 0 ∧ Real.log x₀ = x₀ - 1) ↔ 
  (∀ x : ℝ, x > 0 → Real.log x ≠ x - 1) :=
by sorry

end negation_equivalence_l1676_167657


namespace parallelogram_area_l1676_167687

/-- The area of a parallelogram with base 12 feet and height 5 feet is 60 square feet. -/
theorem parallelogram_area (base height : ℝ) (h1 : base = 12) (h2 : height = 5) :
  base * height = 60 := by
  sorry

end parallelogram_area_l1676_167687


namespace josh_remaining_money_l1676_167604

/-- Calculates the remaining money after Josh's shopping trip. -/
def remaining_money (initial_amount hat_cost pencil_cost cookie_cost cookie_count : ℚ) : ℚ :=
  initial_amount - (hat_cost + pencil_cost + cookie_cost * cookie_count)

/-- Theorem stating that Josh has $3 left after his shopping trip. -/
theorem josh_remaining_money :
  remaining_money 20 10 2 1.25 4 = 3 := by
  sorry

end josh_remaining_money_l1676_167604


namespace dianes_trip_length_l1676_167617

theorem dianes_trip_length :
  ∀ (total_length : ℝ),
  (1/4 : ℝ) * total_length + 24 + (1/3 : ℝ) * total_length = total_length →
  total_length = 57.6 := by
sorry

end dianes_trip_length_l1676_167617


namespace min_value_theorem_l1676_167649

theorem min_value_theorem (a b c d : ℝ) 
  (h1 : 0 ≤ a ∧ a < 2^(1/4))
  (h2 : 0 ≤ b ∧ b < 2^(1/4))
  (h3 : 0 ≤ c ∧ c < 2^(1/4))
  (h4 : 0 ≤ d ∧ d < 2^(1/4))
  (h5 : a^3 + b^3 + c^3 + d^3 = 2) :
  (a / Real.sqrt (2 - a^4)) + (b / Real.sqrt (2 - b^4)) + 
  (c / Real.sqrt (2 - c^4)) + (d / Real.sqrt (2 - d^4)) ≥ 2 := by
  sorry

end min_value_theorem_l1676_167649


namespace mother_father_age_ratio_l1676_167698

/-- Represents the ages and relationships in Darcie's family -/
structure Family where
  darcie_age : ℕ
  father_age : ℕ
  mother_age_ratio : ℚ
  darcie_mother_ratio : ℚ

/-- Theorem stating the ratio of mother's age to father's age -/
theorem mother_father_age_ratio (f : Family)
  (h1 : f.darcie_age = 4)
  (h2 : f.father_age = 30)
  (h3 : f.darcie_mother_ratio = 1 / 6)
  (h4 : f.mother_age_ratio * f.father_age = f.darcie_age / f.darcie_mother_ratio) :
  f.mother_age_ratio = 4 / 5 := by
  sorry


end mother_father_age_ratio_l1676_167698


namespace books_loaned_out_l1676_167672

theorem books_loaned_out (initial_books : ℕ) (return_rate : ℚ) (final_books : ℕ) :
  initial_books = 75 →
  return_rate = 65 / 100 →
  final_books = 61 →
  (initial_books - final_books : ℚ) / (1 - return_rate) = 40 := by
sorry

end books_loaned_out_l1676_167672


namespace cone_radius_l1676_167611

/-- Given a cone with surface area 6 and lateral surface that unfolds into a semicircle,
    prove that the radius of its base is √(2/π) -/
theorem cone_radius (r : ℝ) (l : ℝ) : 
  r > 0 →  -- radius is positive
  l > 0 →  -- slant height is positive
  2 * π * r = π * l →  -- lateral surface unfolds into a semicircle
  π * r^2 + π * r * l = 6 →  -- surface area is 6
  r = Real.sqrt (2 / π) := by
sorry

end cone_radius_l1676_167611


namespace sum_of_bases_is_fifteen_l1676_167606

/-- Represents a fraction in a given base --/
structure FractionInBase where
  numerator : ℕ
  denominator : ℕ
  base : ℕ

/-- Converts a repeating decimal to a fraction --/
def repeatingDecimalToFraction (digits : ℕ) (base : ℕ) : FractionInBase :=
  { numerator := digits,
    denominator := base^2 - 1,
    base := base }

theorem sum_of_bases_is_fifteen :
  let R₁ : ℕ := 9
  let R₂ : ℕ := 6
  let F₁_in_R₁ := repeatingDecimalToFraction 48 R₁
  let F₂_in_R₁ := repeatingDecimalToFraction 84 R₁
  let F₁_in_R₂ := repeatingDecimalToFraction 35 R₂
  let F₂_in_R₂ := repeatingDecimalToFraction 53 R₂
  R₁ + R₂ = 15 := by
  sorry

end sum_of_bases_is_fifteen_l1676_167606


namespace power_division_expression_simplification_l1676_167690

-- Problem 1
theorem power_division (a : ℝ) : a^6 / a^2 = a^4 := by sorry

-- Problem 2
theorem expression_simplification (m : ℝ) : m^2 * m^4 - (2*m^3)^2 = -3*m^6 := by sorry

end power_division_expression_simplification_l1676_167690


namespace bean_garden_rows_l1676_167635

/-- Given a garden with bean plants arranged in rows and columns,
    prove that with 15 columns and 780 total plants, there are 52 rows. -/
theorem bean_garden_rows (total_plants : ℕ) (columns : ℕ) (rows : ℕ) : 
  total_plants = 780 → columns = 15 → total_plants = rows * columns → rows = 52 := by
  sorry

end bean_garden_rows_l1676_167635


namespace brothers_combined_age_theorem_l1676_167662

/-- Represents the ages of two brothers -/
structure BrothersAges where
  adam : ℕ
  tom : ℕ

/-- Calculates the number of years until the brothers' combined age reaches a target -/
def yearsUntilCombinedAge (ages : BrothersAges) (targetAge : ℕ) : ℕ :=
  (targetAge - (ages.adam + ages.tom)) / 2

/-- Theorem: The number of years until Adam and Tom's combined age is 44 is 12 -/
theorem brothers_combined_age_theorem (ages : BrothersAges) 
  (h1 : ages.adam = 8) 
  (h2 : ages.tom = 12) : 
  yearsUntilCombinedAge ages 44 = 12 := by
  sorry

end brothers_combined_age_theorem_l1676_167662


namespace parabola_intersection_distance_l1676_167675

theorem parabola_intersection_distance : 
  let f (x : ℝ) := x^2 - 2*x - 3
  let roots := {x : ℝ | f x = 0}
  ∃ (a b : ℝ), a ∈ roots ∧ b ∈ roots ∧ a ≠ b ∧ |a - b| = 4 :=
by
  sorry

end parabola_intersection_distance_l1676_167675


namespace parking_space_unpainted_side_l1676_167689

/-- Represents a rectangular parking space with three painted sides. -/
structure ParkingSpace where
  width : ℝ
  length : ℝ
  painted_sum : ℝ
  area : ℝ

/-- The length of the unpainted side of a parking space. -/
def unpainted_side_length (p : ParkingSpace) : ℝ := p.length

theorem parking_space_unpainted_side
  (p : ParkingSpace)
  (h1 : p.painted_sum = 37)
  (h2 : p.area = 126)
  (h3 : p.painted_sum = 2 * p.width + p.length)
  (h4 : p.area = p.width * p.length) :
  unpainted_side_length p = 9 := by
  sorry

end parking_space_unpainted_side_l1676_167689


namespace not_power_of_two_l1676_167692

def lower_bound : Nat := 11111
def upper_bound : Nat := 99999

theorem not_power_of_two : ∃ (n : Nat), 
  (n = (upper_bound - lower_bound + 1) * upper_bound) ∧ 
  (n % 9 = 0) ∧
  (∀ (m : Nat), (2^m ≠ n)) := by
  sorry

end not_power_of_two_l1676_167692


namespace kanul_total_amount_l1676_167637

theorem kanul_total_amount (T : ℝ) : 
  T = 500 + 400 + 0.1 * T → T = 1000 := by
  sorry

end kanul_total_amount_l1676_167637


namespace sqrt_two_irrational_l1676_167636

theorem sqrt_two_irrational : Irrational (Real.sqrt 2) := by
  sorry

end sqrt_two_irrational_l1676_167636


namespace workshop_salary_calculation_l1676_167676

/-- Given a workshop with workers and technicians, calculate the average salary of non-technician workers. -/
theorem workshop_salary_calculation
  (total_workers : ℕ)
  (avg_salary_all : ℝ)
  (num_technicians : ℕ)
  (avg_salary_technicians : ℝ)
  (h_total_workers : total_workers = 21)
  (h_avg_salary_all : avg_salary_all = 8000)
  (h_num_technicians : num_technicians = 7)
  (h_avg_salary_technicians : avg_salary_technicians = 12000) :
  let num_rest := total_workers - num_technicians
  let total_salary := avg_salary_all * total_workers
  let total_salary_technicians := avg_salary_technicians * num_technicians
  let total_salary_rest := total_salary - total_salary_technicians
  total_salary_rest / num_rest = 6000 := by
  sorry

end workshop_salary_calculation_l1676_167676


namespace probability_not_adjacent_l1676_167671

def total_chairs : ℕ := 10
def broken_chair : ℕ := 5
def available_chairs : ℕ := total_chairs - 1

theorem probability_not_adjacent : 
  let total_ways := available_chairs.choose 2
  let adjacent_pairs := 6
  (1 - (adjacent_pairs : ℚ) / total_ways) = 5/6 := by sorry

end probability_not_adjacent_l1676_167671


namespace theater_bills_count_l1676_167688

-- Define the problem parameters
def total_tickets : ℕ := 300
def ticket_price : ℕ := 40
def total_revenue : ℕ := total_tickets * ticket_price

-- Define the variables for the number of each type of bill
def num_20_bills : ℕ := 238
def num_10_bills : ℕ := 2 * num_20_bills
def num_5_bills : ℕ := num_10_bills + 20

-- Define the theorem
theorem theater_bills_count :
  -- Conditions
  (20 * num_20_bills + 10 * num_10_bills + 5 * num_5_bills = total_revenue) →
  (num_10_bills = 2 * num_20_bills) →
  (num_5_bills = num_10_bills + 20) →
  -- Conclusion
  (num_20_bills + num_10_bills + num_5_bills = 1210) :=
by sorry

end theater_bills_count_l1676_167688


namespace sock_drawing_probability_l1676_167629

/-- The number of colors of socks --/
def num_colors : ℕ := 5

/-- The number of socks per color --/
def socks_per_color : ℕ := 2

/-- The total number of socks --/
def total_socks : ℕ := num_colors * socks_per_color

/-- The number of socks drawn --/
def socks_drawn : ℕ := 5

/-- The probability of drawing exactly one pair of socks with the same color
    and the rest all different colors --/
theorem sock_drawing_probability : 
  (num_colors * (Nat.choose (num_colors - 1) (socks_drawn - 2)) * 
   (socks_per_color ^ 2) * (socks_per_color ^ (socks_drawn - 2))) /
  (Nat.choose total_socks socks_drawn) = 40 / 63 :=
by sorry

end sock_drawing_probability_l1676_167629


namespace product_of_primes_summing_to_17_l1676_167621

theorem product_of_primes_summing_to_17 (p₁ p₂ p₃ p₄ : ℕ) : 
  p₁.Prime ∧ p₂.Prime ∧ p₃.Prime ∧ p₄.Prime ∧ 
  p₁ + p₂ + p₃ + p₄ = 17 → 
  p₁ * p₂ * p₃ * p₄ = 210 := by
sorry

end product_of_primes_summing_to_17_l1676_167621


namespace three_heads_in_four_tosses_l1676_167686

def coin_toss_probability (n : ℕ) (k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) * (1 / 2) ^ k * (1 / 2) ^ (n - k)

theorem three_heads_in_four_tosses :
  coin_toss_probability 4 3 = 1 / 4 := by
  sorry

end three_heads_in_four_tosses_l1676_167686


namespace work_completion_time_l1676_167630

theorem work_completion_time (x : ℝ) : 
  (x > 0) →  -- A's completion time is positive
  (2 * (1/x + 1/10) + 10 * (1/x) = 1) →  -- Work completion equation
  (x = 15) :=  -- A's solo completion time is 15 days
by sorry

end work_completion_time_l1676_167630


namespace clover_walking_distance_l1676_167600

/-- Clover's walking problem -/
theorem clover_walking_distance 
  (total_distance : ℝ) 
  (num_days : ℕ) 
  (walks_per_day : ℕ) :
  total_distance = 90 →
  num_days = 30 →
  walks_per_day = 2 →
  (total_distance / num_days) / walks_per_day = 1.5 :=
by sorry

end clover_walking_distance_l1676_167600


namespace students_without_A_l1676_167691

theorem students_without_A (total : ℕ) (history_A : ℕ) (math_A : ℕ) (both_A : ℕ) :
  total = 40 →
  history_A = 10 →
  math_A = 18 →
  both_A = 6 →
  total - ((history_A + math_A) - both_A) = 18 :=
by sorry

end students_without_A_l1676_167691


namespace complex_equation_solution_l1676_167626

theorem complex_equation_solution (z : ℂ) : z * (1 - I) = 3 - I → z = 2 + I := by
  sorry

end complex_equation_solution_l1676_167626


namespace lock_combinations_l1676_167651

def digits : ℕ := 10
def dials : ℕ := 4
def even_digits : ℕ := 5

theorem lock_combinations : 
  (even_digits) * (digits - 1) * (digits - 2) * (digits - 3) = 2520 :=
by sorry

end lock_combinations_l1676_167651


namespace tangerine_boxes_count_l1676_167615

/-- Given information about apples and tangerines, prove the number of tangerine boxes --/
theorem tangerine_boxes_count
  (apple_boxes : ℕ)
  (apples_per_box : ℕ)
  (tangerines_per_box : ℕ)
  (total_fruits : ℕ)
  (h1 : apple_boxes = 19)
  (h2 : apples_per_box = 46)
  (h3 : tangerines_per_box = 170)
  (h4 : total_fruits = 1894)
  : ∃ (tangerine_boxes : ℕ), tangerine_boxes = 6 ∧ 
    apple_boxes * apples_per_box + tangerine_boxes * tangerines_per_box = total_fruits :=
by
  sorry


end tangerine_boxes_count_l1676_167615


namespace distance_between_points_l1676_167666

/-- The distance between points (1, 3) and (-5, 7) is 2√13. -/
theorem distance_between_points : 
  let p1 : ℝ × ℝ := (1, 3)
  let p2 : ℝ × ℝ := (-5, 7)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = 2 * Real.sqrt 13 := by
  sorry

end distance_between_points_l1676_167666


namespace school_boys_count_l1676_167660

theorem school_boys_count :
  ∀ (boys girls : ℕ),
  (boys : ℚ) / (girls : ℚ) = 5 / 13 →
  girls = boys + 64 →
  boys = 40 := by
sorry

end school_boys_count_l1676_167660


namespace fraction_simplification_l1676_167639

theorem fraction_simplification (a b c : ℝ) 
  (h1 : a + 2*b + 3*c ≠ 0)
  (h2 : a^2 + 9*c^2 - 4*b^2 + 6*a*c ≠ 0) :
  (a^2 + 4*b^2 - 9*c^2 + 4*a*b) / (a^2 + 9*c^2 - 4*b^2 + 6*a*c) = (a + 2*b - 3*c) / (a - 2*b + 3*c) :=
by sorry

end fraction_simplification_l1676_167639


namespace cube_face_sum_l1676_167655

theorem cube_face_sum (a b c d e f g h : ℕ+) : 
  (a * b * c + a * e * c + a * b * f + a * e * f + 
   d * b * c + d * e * c + d * b * f + d * e * f) = 2107 →
  a + b + c + d + e + f + g + h = 57 := by
sorry

end cube_face_sum_l1676_167655


namespace white_area_is_69_l1676_167670

/-- Represents the dimensions of a rectangular sign -/
structure SignDimensions where
  width : ℕ
  height : ℕ

/-- Represents the area covered by a letter -/
structure LetterArea where
  area : ℕ

/-- Calculates the total area of the sign -/
def totalSignArea (dim : SignDimensions) : ℕ :=
  dim.width * dim.height

/-- Calculates the area covered by the letter M -/
def mArea : LetterArea :=
  { area := 2 * (6 * 1) + 2 * 2 }

/-- Calculates the area covered by the letter A -/
def aArea : LetterArea :=
  { area := 2 * 4 + 1 * 2 }

/-- Calculates the area covered by the letter T -/
def tArea : LetterArea :=
  { area := 1 * 4 + 6 * 1 }

/-- Calculates the area covered by the letter H -/
def hArea : LetterArea :=
  { area := 2 * (6 * 1) + 1 * 3 }

/-- Calculates the total area covered by all letters -/
def totalLettersArea : ℕ :=
  mArea.area + aArea.area + tArea.area + hArea.area

/-- The main theorem: proves that the white area of the sign is 69 square units -/
theorem white_area_is_69 (sign : SignDimensions) 
    (h1 : sign.width = 20) 
    (h2 : sign.height = 6) : 
    totalSignArea sign - totalLettersArea = 69 := by
  sorry


end white_area_is_69_l1676_167670


namespace distance_QR_l1676_167661

-- Define the triangle
def Triangle (D E F : ℝ × ℝ) : Prop :=
  let de := Real.sqrt ((E.1 - D.1)^2 + (E.2 - D.2)^2)
  let ef := Real.sqrt ((F.1 - E.1)^2 + (F.2 - E.2)^2)
  let df := Real.sqrt ((F.1 - D.1)^2 + (F.2 - D.2)^2)
  de = 5 ∧ ef = 12 ∧ df = 13

-- Define the circles
def Circle (Q : ℝ × ℝ) (D E F : ℝ × ℝ) : Prop :=
  let qe := Real.sqrt ((E.1 - Q.1)^2 + (E.2 - Q.2)^2)
  let qd := Real.sqrt ((D.1 - Q.1)^2 + (D.2 - Q.2)^2)
  qe = qd ∧ (E.1 - Q.1) * (F.1 - E.1) + (E.2 - Q.2) * (F.2 - E.2) = 0

def Circle' (R : ℝ × ℝ) (D E F : ℝ × ℝ) : Prop :=
  let rd := Real.sqrt ((D.1 - R.1)^2 + (D.2 - R.2)^2)
  let re := Real.sqrt ((E.1 - R.1)^2 + (E.2 - R.2)^2)
  rd = re ∧ (D.1 - R.1) * (F.1 - D.1) + (D.2 - R.2) * (F.2 - D.2) = 0

-- Theorem statement
theorem distance_QR (D E F Q R : ℝ × ℝ) :
  Triangle D E F →
  Circle Q D E F →
  Circle' R D E F →
  Real.sqrt ((R.1 - Q.1)^2 + (R.2 - Q.2)^2) = 25/12 := by
  sorry

end distance_QR_l1676_167661


namespace max_value_expression_l1676_167664

theorem max_value_expression (a b c d : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) 
  (h_sum : a + b + c + d ≤ 4) :
  (a * (b + 2 * c)) ^ (1/4) + 
  (b * (c + 2 * d)) ^ (1/4) + 
  (c * (d + 2 * a)) ^ (1/4) + 
  (d * (a + 2 * b)) ^ (1/4) ≤ 4 * 3 ^ (1/4) := by
sorry

end max_value_expression_l1676_167664


namespace p_sufficient_not_necessary_q_l1676_167610

theorem p_sufficient_not_necessary_q :
  (∀ x : ℝ, 0 < x ∧ x < 2 → -1 < x ∧ x < 3) ∧
  (∃ x : ℝ, -1 < x ∧ x < 3 ∧ ¬(0 < x ∧ x < 2)) := by
  sorry

end p_sufficient_not_necessary_q_l1676_167610


namespace line_ellipse_intersection_slopes_l1676_167658

-- Define the y-intercept
def y_intercept : ℝ := 8

-- Define the ellipse equation
def ellipse_eq (x y : ℝ) : Prop := 4 * x^2 + 25 * y^2 = 100

-- Define the line equation
def line_eq (m x : ℝ) : ℝ := m * x + y_intercept

-- Theorem statement
theorem line_ellipse_intersection_slopes :
  ∀ m : ℝ, (∃ x : ℝ, ellipse_eq x (line_eq m x)) ↔ m^2 ≥ 2.4 :=
by sorry

end line_ellipse_intersection_slopes_l1676_167658


namespace intersection_of_A_and_B_l1676_167673

-- Define sets A and B
def A : Set ℝ := {x | 2 + x ≥ 4}
def B : Set ℝ := {x | -1 ≤ x ∧ x ≤ 5}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 2 ≤ x ∧ x ≤ 5} := by
  sorry

end intersection_of_A_and_B_l1676_167673


namespace multiply_inverse_square_equals_cube_l1676_167646

theorem multiply_inverse_square_equals_cube (x : ℝ) : 
  x * (1/7)^2 = 7^3 → x = 16807 := by
sorry

end multiply_inverse_square_equals_cube_l1676_167646


namespace shadow_height_ratio_michaels_height_l1676_167603

/-- Given a flagpole and a person casting shadows at the same time, 
    calculate the person's height using the ratio of heights to shadows. -/
theorem shadow_height_ratio 
  (h₁ : ℝ) (s₁ : ℝ) (s₂ : ℝ) 
  (h₁_pos : h₁ > 0) (s₁_pos : s₁ > 0) (s₂_pos : s₂ > 0) :
  ∃ h₂ : ℝ, h₂ = (h₁ * s₂) / s₁ := by
  sorry

/-- Michael's height calculation based on the shadow ratio -/
theorem michaels_height 
  (flagpole_height : ℝ) (flagpole_shadow : ℝ) (michael_shadow : ℝ)
  (flagpole_height_eq : flagpole_height = 50)
  (flagpole_shadow_eq : flagpole_shadow = 25)
  (michael_shadow_eq : michael_shadow = 5) :
  ∃ michael_height : ℝ, michael_height = 10 := by
  sorry

end shadow_height_ratio_michaels_height_l1676_167603


namespace relationship_a_b_l1676_167644

theorem relationship_a_b (a b : ℝ) (ha : a^(1/5) > 1) (hb : 1 > b^(1/5)) : a > 1 ∧ 1 > b := by
  sorry

end relationship_a_b_l1676_167644


namespace angle_D_is_100_l1676_167640

-- Define a cyclic quadrilateral
structure CyclicQuadrilateral where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ
  sum_360 : A + B + C + D = 360
  ratio_abc : ∃ (x : ℝ), A = 3*x ∧ B = 4*x ∧ C = 6*x

-- Theorem statement
theorem angle_D_is_100 (q : CyclicQuadrilateral) : q.D = 100 := by
  sorry

end angle_D_is_100_l1676_167640


namespace statement_c_not_always_true_l1676_167623

theorem statement_c_not_always_true :
  ¬ ∀ (a b c : ℝ), a > b → a * c^2 > b * c^2 := by
  sorry

end statement_c_not_always_true_l1676_167623


namespace complex_root_problem_l1676_167679

theorem complex_root_problem (a b c : ℂ) (h_real : b.im = 0) 
  (h_sum : a + b + c = 4)
  (h_prod_sum : a * b + b * c + c * a = 5)
  (h_prod : a * b * c = 6) :
  b = 1 := by sorry

end complex_root_problem_l1676_167679


namespace floor_pi_minus_e_l1676_167627

theorem floor_pi_minus_e : ⌊π - Real.exp 1⌋ = 0 := by sorry

end floor_pi_minus_e_l1676_167627


namespace subcommittee_count_l1676_167633

theorem subcommittee_count (n : ℕ) (k : ℕ) (h1 : n = 30) (h2 : k = 3) :
  n * (Nat.choose (n - 1) (k - 1)) = 12180 :=
by sorry

end subcommittee_count_l1676_167633


namespace equilateral_triangle_perimeter_equilateral_triangle_perimeter_proof_l1676_167618

/-- The perimeter of an equilateral triangle, given an isosceles triangle with specific properties -/
theorem equilateral_triangle_perimeter : ℝ → ℝ → ℝ → Prop :=
  fun perimeter_isosceles base_isosceles perimeter_equilateral =>
    perimeter_isosceles = 40 ∧
    base_isosceles = 10 ∧
    ∃ (side : ℝ), 
      2 * side + base_isosceles = perimeter_isosceles ∧
      3 * side = perimeter_equilateral ∧
      perimeter_equilateral = 45

/-- Proof of the theorem -/
theorem equilateral_triangle_perimeter_proof :
  ∃ (perimeter_equilateral : ℝ),
    equilateral_triangle_perimeter 40 10 perimeter_equilateral :=
by
  sorry

#check equilateral_triangle_perimeter
#check equilateral_triangle_perimeter_proof

end equilateral_triangle_perimeter_equilateral_triangle_perimeter_proof_l1676_167618


namespace inverse_of_BP_squared_l1676_167622

/-- Given a 2x2 matrix B and a diagonal matrix P, prove that the inverse of (BP)² has a specific form. -/
theorem inverse_of_BP_squared (B P : Matrix (Fin 2) (Fin 2) ℚ) : 
  B⁻¹ = ![![3, 7], ![-2, -4]] →
  P = ![![1, 0], ![0, 2]] →
  ((B * P)^2)⁻¹ = ![![8, 28], ![-4, -12]] := by sorry

end inverse_of_BP_squared_l1676_167622


namespace gcd_power_two_minus_one_l1676_167653

theorem gcd_power_two_minus_one :
  Nat.gcd (2^1998 - 1) (2^1989 - 1) = 2^9 - 1 := by
  sorry

end gcd_power_two_minus_one_l1676_167653


namespace vanya_age_l1676_167699

/-- Represents the ages of Vanya, his dad, and Seryozha -/
structure Ages where
  vanya : ℕ
  dad : ℕ
  seryozha : ℕ

/-- The conditions given in the problem -/
def age_relationships (ages : Ages) : Prop :=
  ages.vanya * 3 = ages.dad ∧
  ages.vanya = ages.seryozha * 3 ∧
  ages.dad = ages.seryozha + 40

/-- The theorem stating Vanya's age -/
theorem vanya_age (ages : Ages) : age_relationships ages → ages.vanya = 15 := by
  sorry

end vanya_age_l1676_167699


namespace solution_product_l1676_167694

theorem solution_product (p q : ℝ) : 
  (p - 6) * (3 * p + 10) = p^2 - 19 * p + 50 →
  (q - 6) * (3 * q + 10) = q^2 - 19 * q + 50 →
  p ≠ q →
  (p + 2) * (q + 2) = 108 := by
sorry

end solution_product_l1676_167694


namespace tree_arrangement_probability_l1676_167652

def maple_trees : ℕ := 4
def oak_trees : ℕ := 5
def birch_trees : ℕ := 6
def total_trees : ℕ := maple_trees + oak_trees + birch_trees

def valid_arrangements : ℕ := (Nat.choose 7 maple_trees) * 1

def total_arrangements : ℕ := (Nat.factorial total_trees) / 
  (Nat.factorial maple_trees * Nat.factorial oak_trees * Nat.factorial birch_trees)

theorem tree_arrangement_probability : 
  (valid_arrangements : ℚ) / total_arrangements = 7 / 166320 := by sorry

end tree_arrangement_probability_l1676_167652
