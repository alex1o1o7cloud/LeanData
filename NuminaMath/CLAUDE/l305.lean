import Mathlib

namespace employee_pay_l305_30553

theorem employee_pay (total_pay m_pay n_pay : ℝ) : 
  total_pay = 550 →
  m_pay = 1.2 * n_pay →
  m_pay + n_pay = total_pay →
  n_pay = 250 := by
  sorry

end employee_pay_l305_30553


namespace sum_of_squares_l305_30531

theorem sum_of_squares : 1000 ^ 2 + 1001 ^ 2 + 1002 ^ 2 + 1003 ^ 2 + 1004 ^ 2 = 5020030 := by
  sorry

end sum_of_squares_l305_30531


namespace license_plate_count_l305_30577

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 14

/-- The length of the license plate -/
def plate_length : ℕ := 6

/-- The number of possible first letters (B or C) -/
def first_letter_choices : ℕ := 2

/-- The number of possible last letters (N) -/
def last_letter_choices : ℕ := 1

/-- The number of letters that cannot be used in the middle (B, C, M, N) -/
def excluded_middle_letters : ℕ := 4

theorem license_plate_count :
  (first_letter_choices * (alphabet_size - excluded_middle_letters) *
   (alphabet_size - excluded_middle_letters - 1) *
   (alphabet_size - excluded_middle_letters - 2) *
   (alphabet_size - excluded_middle_letters - 3) *
   last_letter_choices) = 15840 :=
by sorry

end license_plate_count_l305_30577


namespace number_of_girls_l305_30545

theorem number_of_girls (total_students : ℕ) (prob_girl : ℚ) (num_girls : ℕ) : 
  total_students = 20 →
  prob_girl = 2/5 →
  num_girls = (total_students : ℚ) * prob_girl →
  num_girls = 8 := by
sorry

end number_of_girls_l305_30545


namespace angle_between_planes_exists_l305_30537

-- Define the basic structures
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

structure Plane where
  normal : Point3D
  d : ℝ

-- Define the projection axis
def ProjectionAxis : Point3D → Point3D → Prop :=
  sorry

-- Define a point on the projection axis
def PointOnProjectionAxis (p : Point3D) : Prop :=
  ∃ (q : Point3D), ProjectionAxis p q

-- Define a plane passing through a point
def PlaneThroughPoint (plane : Plane) (point : Point3D) : Prop :=
  plane.normal.x * point.x + plane.normal.y * point.y + plane.normal.z * point.z + plane.d = 0

-- Define the angle between two planes
def AngleBetweenPlanes (plane1 plane2 : Plane) : ℝ :=
  sorry

-- Theorem statement
theorem angle_between_planes_exists :
  ∀ (p : Point3D) (plane1 plane2 : Plane),
    PointOnProjectionAxis p →
    PlaneThroughPoint plane1 p →
    ∃ (angle : ℝ), AngleBetweenPlanes plane1 plane2 = angle :=
  sorry

end angle_between_planes_exists_l305_30537


namespace quadratic_function_property_quadratic_function_property_independent_of_b_l305_30594

theorem quadratic_function_property (c d h b : ℝ) : 
  let f (x : ℝ) := c * x^2
  let x₁ := b - d - h
  let x₂ := b - d
  let x₃ := b + d
  let x₄ := b + d + h
  let y₁ := f x₁
  let y₂ := f x₂
  let y₃ := f x₃
  let y₄ := f x₄
  (y₁ + y₄) - (y₂ + y₃) = 2 * c * h * (2 * d + h) :=
by sorry

theorem quadratic_function_property_independent_of_b (c d h : ℝ) :
  ∀ b₁ b₂ : ℝ, 
  let f (x : ℝ) := c * x^2
  let x₁ (b : ℝ) := b - d - h
  let x₂ (b : ℝ) := b - d
  let x₃ (b : ℝ) := b + d
  let x₄ (b : ℝ) := b + d + h
  let y₁ (b : ℝ) := f (x₁ b)
  let y₂ (b : ℝ) := f (x₂ b)
  let y₃ (b : ℝ) := f (x₃ b)
  let y₄ (b : ℝ) := f (x₄ b)
  (y₁ b₁ + y₄ b₁) - (y₂ b₁ + y₃ b₁) = (y₁ b₂ + y₄ b₂) - (y₂ b₂ + y₃ b₂) :=
by sorry

end quadratic_function_property_quadratic_function_property_independent_of_b_l305_30594


namespace smallest_k_for_cos_squared_one_l305_30555

theorem smallest_k_for_cos_squared_one :
  ∃ k : ℕ+, 
    (∀ m : ℕ+, m < k → (Real.cos ((m.val ^ 2 + 7 ^ 2 : ℝ) * Real.pi / 180)) ^ 2 ≠ 1) ∧
    (Real.cos ((k.val ^ 2 + 7 ^ 2 : ℝ) * Real.pi / 180)) ^ 2 = 1 ∧
    k = 49 := by
  sorry

end smallest_k_for_cos_squared_one_l305_30555


namespace sugar_amount_theorem_l305_30576

def sugar_amount (sugar flour baking_soda chocolate_chips : ℚ) : Prop :=
  -- Ratio of sugar to flour is 5:4
  sugar / flour = 5 / 4 ∧
  -- Ratio of flour to baking soda is 10:1
  flour / baking_soda = 10 / 1 ∧
  -- Ratio of baking soda to chocolate chips is 3:2
  baking_soda / chocolate_chips = 3 / 2 ∧
  -- New ratio after adding 120 pounds of baking soda and 50 pounds of chocolate chips
  flour / (baking_soda + 120) = 16 / 3 ∧
  flour / (chocolate_chips + 50) = 16 / 2 ∧
  -- The amount of sugar is 1714 pounds
  sugar = 1714

theorem sugar_amount_theorem :
  ∃ sugar flour baking_soda chocolate_chips : ℚ,
    sugar_amount sugar flour baking_soda chocolate_chips :=
by
  sorry

end sugar_amount_theorem_l305_30576


namespace proposition_implication_l305_30560

theorem proposition_implication (p q : Prop) 
  (h1 : ¬(p ∧ q)) 
  (h2 : ¬¬p) : 
  ¬q := by
sorry

end proposition_implication_l305_30560


namespace wall_width_l305_30544

theorem wall_width (w h l : ℝ) (volume : ℝ) : 
  h = 4 * w →
  l = 3 * h →
  volume = w * h * l →
  volume = 10368 →
  w = 6 := by
  sorry

end wall_width_l305_30544


namespace sum_interior_angles_pentagon_l305_30534

/-- The sum of interior angles of a polygon with n sides is (n - 2) * 180 degrees. -/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- A pentagon is a polygon with 5 sides. -/
def pentagon : ℕ := 5

theorem sum_interior_angles_pentagon :
  sum_interior_angles pentagon = 540 := by
  sorry

end sum_interior_angles_pentagon_l305_30534


namespace length_of_cd_l305_30564

/-- Represents a point that divides a line segment in a given ratio -/
structure DividingPoint where
  ratio_left : ℚ
  ratio_right : ℚ

/-- Represents a line segment divided by two points -/
structure DividedSegment where
  length : ℝ
  point1 : DividingPoint
  point2 : DividingPoint
  distance_between_points : ℝ

/-- Theorem stating the length of CD given the conditions -/
theorem length_of_cd (cd : DividedSegment) : 
  cd.point1.ratio_left = 3 ∧ 
  cd.point1.ratio_right = 5 ∧ 
  cd.point2.ratio_left = 4 ∧ 
  cd.point2.ratio_right = 7 ∧ 
  cd.distance_between_points = 3 → 
  cd.length = 264 := by
  sorry

end length_of_cd_l305_30564


namespace train_passing_time_l305_30507

/-- The time it takes for two trains to pass each other -/
theorem train_passing_time (v1 l1 v2 l2 : ℝ) : 
  v1 > 0 → l1 > 0 → v2 > 0 → l2 > 0 →
  (l1 / v1 = 5) →
  (v1 = 2 * v2) →
  (l1 = 3 * l2) →
  (l1 + l2) / (v1 + v2) = 40 / 9 := by
  sorry

end train_passing_time_l305_30507


namespace sum_base7_equals_650_l305_30573

/-- Converts a number from base 7 to base 10 --/
def base7ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 7 --/
def base10ToBase7 (n : ℕ) : ℕ := sorry

/-- The sum of three numbers in base 7 --/
def sumBase7 (a b c : ℕ) : ℕ :=
  base10ToBase7 (base7ToBase10 a + base7ToBase10 b + base7ToBase10 c)

theorem sum_base7_equals_650 :
  sumBase7 543 65 6 = 650 := by sorry

end sum_base7_equals_650_l305_30573


namespace birthday_cake_red_candles_l305_30539

/-- The number of red candles on a birthday cake -/
def red_candles (total_candles yellow_candles blue_candles : ℕ) : ℕ :=
  total_candles - (yellow_candles + blue_candles)

/-- Theorem stating the number of red candles used for the birthday cake -/
theorem birthday_cake_red_candles :
  red_candles 79 27 38 = 14 := by
  sorry

end birthday_cake_red_candles_l305_30539


namespace junior_trip_fraction_l305_30533

theorem junior_trip_fraction (S J : ℚ) 
  (h1 : J = 2/3 * S) 
  (h2 : 2/3 * S + x * J = 1/2 * (S + J)) 
  (h3 : S > 0) 
  (h4 : J > 0) : 
  x = 1/4 := by
  sorry

end junior_trip_fraction_l305_30533


namespace expression_value_l305_30579

theorem expression_value (x y : ℤ) (hx : x = 3) (hy : y = 4) : 3 * x - 2 * y + 5 = 6 := by
  sorry

end expression_value_l305_30579


namespace general_term_formula_correct_l305_30526

/-- Arithmetic sequence with first term 3 and common difference 2 -/
def arithmeticSequence (n : ℕ) : ℕ := 3 + 2 * (n - 1)

/-- General term formula -/
def generalTerm (n : ℕ) : ℕ := 2 * n + 1

/-- Theorem stating that the general term formula is correct for the given arithmetic sequence -/
theorem general_term_formula_correct :
  ∀ n : ℕ, n > 0 → arithmeticSequence n = generalTerm n := by
  sorry

end general_term_formula_correct_l305_30526


namespace net_sales_effect_l305_30597

/-- Calculates the net effect on sales after two consecutive price reductions and sales increases -/
theorem net_sales_effect (initial_price_reduction : ℝ) 
                         (initial_sales_increase : ℝ)
                         (second_price_reduction : ℝ)
                         (second_sales_increase : ℝ) :
  initial_price_reduction = 0.20 →
  initial_sales_increase = 0.80 →
  second_price_reduction = 0.15 →
  second_sales_increase = 0.60 →
  let first_quarter_sales := 1 + initial_sales_increase
  let second_quarter_sales := first_quarter_sales * (1 + second_sales_increase)
  let net_effect := (second_quarter_sales - 1) * 100
  net_effect = 188 := by
sorry

end net_sales_effect_l305_30597


namespace always_odd_l305_30529

theorem always_odd (n : ℤ) : ∃ k : ℤ, n^2 + n + 5 = 2*k + 1 := by
  sorry

end always_odd_l305_30529


namespace lines_perpendicular_iff_b_eq_neg_ten_l305_30572

-- Define the slopes of the two lines
def slope1 : ℚ := -1/2
def slope2 (b : ℚ) : ℚ := -b/5

-- Define the perpendicularity condition
def perpendicular (b : ℚ) : Prop := slope1 * slope2 b = -1

-- Theorem statement
theorem lines_perpendicular_iff_b_eq_neg_ten :
  ∀ b : ℚ, perpendicular b ↔ b = -10 := by sorry

end lines_perpendicular_iff_b_eq_neg_ten_l305_30572


namespace rational_equation_solution_l305_30530

theorem rational_equation_solution : ∃ x : ℚ, 
  (x^2 - 6*x + 8) / (x^2 - 9*x + 14) = (x^2 - 3*x - 18) / (x^2 - 2*x - 24) ∧ 
  x = -5/4 := by
  sorry

end rational_equation_solution_l305_30530


namespace greatest_of_three_consecutive_integers_l305_30540

theorem greatest_of_three_consecutive_integers (x : ℤ) 
  (h : x + (x + 1) + (x + 2) = 24) : 
  max x (max (x + 1) (x + 2)) = 9 := by
  sorry

end greatest_of_three_consecutive_integers_l305_30540


namespace min_value_abc_l305_30519

theorem min_value_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_abc : a * b * c = 27) :
  a^2 + 6*a*b + 9*b^2 + 3*c^2 ≥ 126 ∧ 
  ∃ (a₀ b₀ c₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ 0 < c₀ ∧ a₀ * b₀ * c₀ = 27 ∧ 
    a₀^2 + 6*a₀*b₀ + 9*b₀^2 + 3*c₀^2 = 126 :=
by sorry

end min_value_abc_l305_30519


namespace subtract_negative_one_three_l305_30559

theorem subtract_negative_one_three : -1 - 3 = -4 := by
  sorry

end subtract_negative_one_three_l305_30559


namespace mutually_exclusive_not_opposing_l305_30528

/-- A bag containing two red balls and two black balls -/
structure Bag :=
  (red_balls : ℕ := 2)
  (black_balls : ℕ := 2)

/-- The event of drawing exactly one black ball -/
def exactly_one_black (bag : Bag) : Set (Fin 2 → Bool) :=
  {draw | (draw 0 = true ∧ draw 1 = false) ∨ (draw 0 = false ∧ draw 1 = true)}

/-- The event of drawing exactly two black balls -/
def exactly_two_black (bag : Bag) : Set (Fin 2 → Bool) :=
  {draw | draw 0 = true ∧ draw 1 = true}

/-- Two events are mutually exclusive if their intersection is empty -/
def mutually_exclusive (E F : Set (Fin 2 → Bool)) : Prop :=
  E ∩ F = ∅

/-- Two events are opposing if their union is the entire sample space -/
def opposing (E F : Set (Fin 2 → Bool)) : Prop :=
  E ∪ F = Set.univ

theorem mutually_exclusive_not_opposing (bag : Bag) :
  mutually_exclusive (exactly_one_black bag) (exactly_two_black bag) ∧
  ¬opposing (exactly_one_black bag) (exactly_two_black bag) :=
sorry

end mutually_exclusive_not_opposing_l305_30528


namespace f_properties_l305_30584

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -1/a + 2/x

theorem f_properties (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ > x₂ ∧ x₂ > 0 → f a x₁ < f a x₂) ∧
  (a < 0 → ∀ x : ℝ, x > 0 → f a x > 0) ∧
  (a > 0 → ∀ x : ℝ, 0 < x ∧ x < 2*a ↔ f a x > 0) :=
sorry

end f_properties_l305_30584


namespace intersection_implies_a_value_l305_30521

theorem intersection_implies_a_value (A B : Set ℝ) (a : ℝ) :
  A = {-1, 1, 3} →
  B = {a + 2, a^2 + 4} →
  A ∩ B = {3} →
  a = 1 := by
sorry

end intersection_implies_a_value_l305_30521


namespace sum_inequality_l305_30588

theorem sum_inequality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (h_prod : a * b * c * d = 1) :
  1 / Real.sqrt (1/2 + a + a*b + a*b*c) +
  1 / Real.sqrt (1/2 + b + b*c + b*c*d) +
  1 / Real.sqrt (1/2 + c + c*d + c*d*a) +
  1 / Real.sqrt (1/2 + d + d*a + d*a*b) ≥ Real.sqrt 2 := by
sorry

end sum_inequality_l305_30588


namespace min_value_sum_reciprocals_l305_30541

theorem min_value_sum_reciprocals (a b c : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (sum_eq_3 : a + b + c = 3) : 
  1 / (a + b) + 1 / (b + c) + 1 / (c + a) ≥ 3 / 2 := by
  sorry

end min_value_sum_reciprocals_l305_30541


namespace inscribed_square_and_circle_dimensions_l305_30562

-- Define the right triangle DEF
def triangle_DEF (DE EF DF : ℝ) : Prop :=
  DE = 5 ∧ EF = 12 ∧ DF = 13 ∧ DE ^ 2 + EF ^ 2 = DF ^ 2

-- Define the inscribed square PQRS
def inscribed_square (s : ℝ) (DE EF DF : ℝ) : Prop :=
  triangle_DEF DE EF DF ∧
  ∃ (P Q R S : ℝ × ℝ),
    -- P and Q on DF, R on DE, S on EF
    (P.1 + Q.1 = DF) ∧ (R.2 = DE) ∧ (S.1 = EF) ∧
    -- PQRS is a square with side length s
    (Q.1 - P.1 = s) ∧ (R.2 - Q.2 = s) ∧ (S.1 - R.1 = s) ∧ (P.2 - S.2 = s)

-- Define the inscribed circle
def inscribed_circle (r : ℝ) (s : ℝ) : Prop :=
  r = s / 2

-- Theorem statement
theorem inscribed_square_and_circle_dimensions :
  ∀ (DE EF DF s r : ℝ),
    inscribed_square s DE EF DF →
    inscribed_circle r s →
    s = 780 / 169 ∧ r = 390 / 338 := by
  sorry

end inscribed_square_and_circle_dimensions_l305_30562


namespace train_car_speed_ratio_l305_30513

/-- Given a bus, a train, and a car with the following properties:
  * The speed of the bus is 3/4 of the speed of the train
  * The bus travels 480 km in 8 hours
  * The car travels 450 km in 6 hours
  Prove that the ratio of the speed of the train to the speed of the car is 16:15 -/
theorem train_car_speed_ratio : 
  ∀ (bus_speed train_speed car_speed : ℝ),
  bus_speed = (3/4) * train_speed →
  bus_speed = 480 / 8 →
  car_speed = 450 / 6 →
  train_speed / car_speed = 16 / 15 := by
sorry

end train_car_speed_ratio_l305_30513


namespace smallest_winning_number_for_bernardo_l305_30524

theorem smallest_winning_number_for_bernardo :
  ∃ (N : ℕ), N = 22 ∧
  (∀ k : ℕ, k < N →
    (3*k ≤ 999 ∧
     3*k + 30 ≤ 999 ∧
     9*k + 90 ≤ 999 ∧
     9*k + 120 ≤ 999 ∧
     27*k + 360 ≤ 999)) ∧
  (3*N ≤ 999 ∧
   3*N + 30 ≤ 999 ∧
   9*N + 90 ≤ 999 ∧
   9*N + 120 ≤ 999 ∧
   27*N + 360 ≤ 999 ∧
   27*N + 390 > 999) :=
by sorry

end smallest_winning_number_for_bernardo_l305_30524


namespace intersection_of_A_and_B_l305_30504

def A : Set (ℝ × ℝ) := {p | p.2 = -p.1}
def B : Set (ℝ × ℝ) := {p | p.2 = p.1^2 - 2}

theorem intersection_of_A_and_B :
  A ∩ B = {(-2, 2), (1, -1)} := by sorry

end intersection_of_A_and_B_l305_30504


namespace sum_of_five_unit_fractions_l305_30549

theorem sum_of_five_unit_fractions :
  ∃ (a b c d e : ℕ+), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
                       b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
                       c ≠ d ∧ c ≠ e ∧ 
                       d ≠ e ∧
                       (1 : ℚ) = 1 / a + 1 / b + 1 / c + 1 / d + 1 / e :=
sorry

end sum_of_five_unit_fractions_l305_30549


namespace partial_fraction_decomposition_l305_30590

theorem partial_fraction_decomposition :
  ∃! (P Q R : ℝ), ∀ (x : ℝ), x ≠ 4 ∧ x ≠ 2 →
    3 * x^2 + 2 * x = (x - 4) * (x - 2)^2 * (P / (x - 4) + Q / (x - 2) + R / (x - 2)^2) ∧
    P = 14 ∧ Q = -11 ∧ R = -8 := by
  sorry

end partial_fraction_decomposition_l305_30590


namespace addition_problem_l305_30502

theorem addition_problem : (-5 : ℤ) + 8 + (-4) = -1 := by
  sorry

end addition_problem_l305_30502


namespace units_digit_of_product_of_first_four_composites_l305_30514

def first_four_composite_numbers : List Nat := [4, 6, 8, 9]

def product_of_list (l : List Nat) : Nat :=
  l.foldl (·*·) 1

def units_digit (n : Nat) : Nat :=
  n % 10

theorem units_digit_of_product_of_first_four_composites :
  units_digit (product_of_list first_four_composite_numbers) = 8 := by
  sorry

end units_digit_of_product_of_first_four_composites_l305_30514


namespace complex_equation_solution_l305_30546

theorem complex_equation_solution (Z : ℂ) : Z = Complex.I * (2 + Z) → Z = -1 + Complex.I := by
  sorry

end complex_equation_solution_l305_30546


namespace bucket_fill_time_l305_30520

/-- Given that it takes 2 minutes to fill two-thirds of a bucket,
    prove that it takes 3 minutes to fill the entire bucket. -/
theorem bucket_fill_time :
  let partial_time : ℚ := 2
  let partial_fill : ℚ := 2/3
  let full_time : ℚ := 3
  (partial_fill * full_time = partial_time) → full_time = 3 :=
by
  sorry

end bucket_fill_time_l305_30520


namespace rotated_line_equation_l305_30543

/-- Given a line with equation x - y + 1 = 0 and a point P(3, 4) on this line,
    rotating the line 90° counterclockwise around P results in a line with equation x + y - 7 = 0 -/
theorem rotated_line_equation (x y : ℝ) : 
  (x - y + 1 = 0 ∧ 3 - 4 + 1 = 0) → 
  (∃ (m : ℝ), m * (x - 3) + (y - 4) = 0 ∧ m = 1) →
  x + y - 7 = 0 := by
sorry

end rotated_line_equation_l305_30543


namespace xiao_ming_score_l305_30598

/-- Calculates the comprehensive score based on individual scores and weights -/
def comprehensive_score (written : ℝ) (practical : ℝ) (publicity : ℝ) 
  (written_weight : ℝ) (practical_weight : ℝ) (publicity_weight : ℝ) : ℝ :=
  written * written_weight + practical * practical_weight + publicity * publicity_weight

/-- Theorem stating that Xiao Ming's comprehensive score is 97 -/
theorem xiao_ming_score : 
  comprehensive_score 96 98 96 0.3 0.5 0.2 = 97 := by
  sorry

#eval comprehensive_score 96 98 96 0.3 0.5 0.2

end xiao_ming_score_l305_30598


namespace scientific_notation_of_1_6_million_l305_30580

theorem scientific_notation_of_1_6_million :
  ∃ (a : ℝ) (n : ℤ), 1600000 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 1.6 ∧ n = 6 := by
  sorry

end scientific_notation_of_1_6_million_l305_30580


namespace function_property_l305_30558

theorem function_property (f : ℤ → ℝ) 
  (h1 : ∀ x y : ℤ, f x * f y = f (x + y) + f (x - y))
  (h2 : f 1 = 5/2) :
  ∀ x : ℤ, f x = (5/2) ^ x := by sorry

end function_property_l305_30558


namespace area_inside_Q_outside_P_R_l305_30585

-- Define the circles
def circle_P : Real := 1
def circle_Q : Real := 2
def circle_R : Real := 1

-- Define the centers of the circles
def center_P : ℝ × ℝ := (0, 0)
def center_R : ℝ × ℝ := (2, 0)
def center_Q : ℝ × ℝ := (0, 0)

-- Define the tangency conditions
def Q_R_tangent : Prop := 
  (center_Q.1 - center_R.1)^2 + (center_Q.2 - center_R.2)^2 = (circle_Q + circle_R)^2

def R_P_tangent : Prop :=
  (center_R.1 - center_P.1)^2 + (center_R.2 - center_P.2)^2 = (circle_R + circle_P)^2

-- Theorem statement
theorem area_inside_Q_outside_P_R : 
  Q_R_tangent → R_P_tangent → 
  (π * circle_Q^2) - (π * circle_P^2) - (π * circle_R^2) = 2 * π := by
  sorry

end area_inside_Q_outside_P_R_l305_30585


namespace real_part_of_complex_number_l305_30518

theorem real_part_of_complex_number : 
  (1 + 2 / (Complex.I + 1)).re = 2 := by sorry

end real_part_of_complex_number_l305_30518


namespace only_negative_number_l305_30503

theorem only_negative_number (a b c d : ℤ) (h1 : a = 5) (h2 : b = 1) (h3 : c = -2) (h4 : d = 0) :
  (a < 0 ∨ b < 0 ∨ c < 0 ∨ d < 0) ∧ (c < 0) ∧ (a ≥ 0) ∧ (b ≥ 0) ∧ (d ≥ 0) :=
by sorry

end only_negative_number_l305_30503


namespace smallest_base_sum_l305_30550

theorem smallest_base_sum : ∃ (a b : ℕ), 
  a ≠ b ∧ 
  a > 1 ∧ 
  b > 1 ∧
  5 * a + 2 = 2 * b + 5 ∧ 
  (∀ (a' b' : ℕ), a' ≠ b' → a' > 1 → b' > 1 → 5 * a' + 2 = 2 * b' + 5 → a + b ≤ a' + b') ∧
  a + b = 9 :=
by sorry

end smallest_base_sum_l305_30550


namespace chocolate_bar_cost_l305_30592

/-- The cost of one chocolate bar given the conditions of the problem -/
theorem chocolate_bar_cost : 
  ∀ (scouts : ℕ) (smores_per_scout : ℕ) (smores_per_bar : ℕ) (total_cost : ℚ),
  scouts = 15 →
  smores_per_scout = 2 →
  smores_per_bar = 3 →
  total_cost = 15 →
  (total_cost / (scouts * smores_per_scout / smores_per_bar : ℚ)) = 3/2 := by
sorry

end chocolate_bar_cost_l305_30592


namespace boat_downstream_distance_l305_30557

/-- Calculates the distance traveled downstream by a boat given its speed in still water,
    the stream speed, and the time taken to travel downstream. -/
def distance_downstream (boat_speed : ℝ) (stream_speed : ℝ) (time : ℝ) : ℝ :=
  (boat_speed + stream_speed) * time

/-- Proves that a boat with a speed of 16 km/hr in still water, traveling in a stream
    with a speed of 4 km/hr for 3 hours, will travel 60 km downstream. -/
theorem boat_downstream_distance :
  distance_downstream 16 4 3 = 60 := by
  sorry

end boat_downstream_distance_l305_30557


namespace initial_pencils_count_l305_30501

/-- The number of pencils Sally took out of the drawer -/
def pencils_taken : ℕ := 4

/-- The number of pencils left in the drawer after Sally took some out -/
def pencils_left : ℕ := 5

/-- The initial number of pencils in the drawer -/
def initial_pencils : ℕ := pencils_taken + pencils_left

theorem initial_pencils_count : initial_pencils = 9 := by
  sorry

end initial_pencils_count_l305_30501


namespace french_toast_slices_per_loaf_l305_30596

/-- The number of slices in each loaf of bread for Suzanne's french toast -/
def slices_per_loaf (days_per_week : ℕ) (slices_per_day : ℕ) (weeks : ℕ) (total_loaves : ℕ) : ℕ :=
  (days_per_week * slices_per_day * weeks) / total_loaves

/-- Proof that the number of slices in each loaf is 6 -/
theorem french_toast_slices_per_loaf :
  slices_per_loaf 2 3 52 26 = 6 := by
  sorry

end french_toast_slices_per_loaf_l305_30596


namespace chipped_marbles_bag_l305_30548

/-- Represents the number of marbles in each bag -/
def bags : List Nat := [15, 18, 22, 24, 30]

/-- Represents the total number of marbles -/
def total : Nat := bags.sum

/-- Predicate to check if a list of two numbers from the bags list sums to a given value -/
def hasTwoSum (s : Nat) : Prop := ∃ (a b : Nat), a ∈ bags ∧ b ∈ bags ∧ a ≠ b ∧ a + b = s

/-- The main theorem stating that the bag with chipped marbles contains 24 marbles -/
theorem chipped_marbles_bag : 
  ∃ (jane george : Nat), 
    jane ∈ bags ∧ 
    george ∈ bags ∧ 
    jane ≠ george ∧
    hasTwoSum jane ∧ 
    hasTwoSum george ∧ 
    jane = 3 * george ∧ 
    total - jane - george = 24 := by
  sorry

end chipped_marbles_bag_l305_30548


namespace solution_system1_solution_system2_l305_30586

-- Define the first system of equations
def system1 (x y : ℝ) : Prop :=
  x + y + 3 = 10 ∧ 4 * (x + y) - y = 25

-- Define the second system of equations
def system2 (x y : ℝ) : Prop :=
  (2 * y - 4 * x) / 3 + 2 * x = 4 ∧ y - 2 * x + 3 = 6

-- Theorem for the first system
theorem solution_system1 : ∃ x y : ℝ, system1 x y ∧ x = 4 ∧ y = 3 := by
  sorry

-- Theorem for the second system
theorem solution_system2 : ∃ x y : ℝ, system2 x y ∧ x = 1 ∧ y = 5 := by
  sorry

end solution_system1_solution_system2_l305_30586


namespace angle_with_double_supplement_l305_30535

theorem angle_with_double_supplement (α : ℝ) :
  (180 - α = 2 * α) → α = 60 := by
  sorry

end angle_with_double_supplement_l305_30535


namespace lemon_juice_for_dozen_cupcakes_l305_30538

/-- The number of tablespoons of lemon juice provided by one lemon -/
def tablespoons_per_lemon : ℕ := 4

/-- The number of lemons needed for 3 dozen cupcakes -/
def lemons_for_three_dozen : ℕ := 9

/-- The number of tablespoons of lemon juice needed for a dozen cupcakes -/
def tablespoons_for_dozen : ℕ := 12

/-- Proves that the number of tablespoons of lemon juice needed for a dozen cupcakes is 12 -/
theorem lemon_juice_for_dozen_cupcakes : 
  tablespoons_for_dozen = (lemons_for_three_dozen * tablespoons_per_lemon) / 3 :=
by sorry

end lemon_juice_for_dozen_cupcakes_l305_30538


namespace may_savings_l305_30561

def savings (month : Nat) : Nat :=
  match month with
  | 0 => 10  -- January (0-indexed)
  | n + 1 => 2 * savings n

theorem may_savings : savings 4 = 160 := by
  sorry

end may_savings_l305_30561


namespace exactlyOnePen_exactlyTwoPencils_mutually_exclusive_not_opposite_l305_30542

/-- Represents the contents of a pencil case -/
structure PencilCase where
  pencils : ℕ
  pens : ℕ

/-- Represents the outcome of selecting two items from a pencil case -/
inductive Selection
  | TwoPencils
  | OnePencilOnePen
  | TwoPens

/-- Defines the pencil case with 2 pencils and 2 pens -/
def myPencilCase : PencilCase := { pencils := 2, pens := 2 }

/-- Event: Exactly 1 pen is selected -/
def exactlyOnePen (s : Selection) : Prop :=
  s = Selection.OnePencilOnePen

/-- Event: Exactly 2 pencils are selected -/
def exactlyTwoPencils (s : Selection) : Prop :=
  s = Selection.TwoPencils

/-- Two events are mutually exclusive -/
def mutuallyExclusive (e1 e2 : Selection → Prop) : Prop :=
  ∀ s, ¬(e1 s ∧ e2 s)

/-- Two events are opposite -/
def opposite (e1 e2 : Selection → Prop) : Prop :=
  ∀ s, e1 s ↔ ¬(e2 s)

theorem exactlyOnePen_exactlyTwoPencils_mutually_exclusive_not_opposite :
  mutuallyExclusive exactlyOnePen exactlyTwoPencils ∧
  ¬(opposite exactlyOnePen exactlyTwoPencils) :=
by sorry

end exactlyOnePen_exactlyTwoPencils_mutually_exclusive_not_opposite_l305_30542


namespace sliding_triangle_forms_ellipse_l305_30567

/-- Triangle ABC with A and B on perpendicular lines -/
structure SlidingTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  perpendicular : A.2 = 0 ∧ B.1 = 0
  non_right_angle_at_C : ∀ (t : ℝ), (C.1 - A.1) * (C.2 - B.2) ≠ (C.2 - A.2) * (C.1 - B.1)

/-- The locus of point C forms an ellipse -/
def is_ellipse (locus : Set (ℝ × ℝ)) : Prop :=
  ∃ (a b h k : ℝ), a > 0 ∧ b > 0 ∧
    ∀ (x y : ℝ), (x, y) ∈ locus ↔ (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1

/-- The theorem statement -/
theorem sliding_triangle_forms_ellipse (triangle : SlidingTriangle) :
  ∃ (locus : Set (ℝ × ℝ)), is_ellipse locus ∧ ∀ (t : ℝ), triangle.C ∈ locus :=
sorry

end sliding_triangle_forms_ellipse_l305_30567


namespace range_of_m_l305_30512

theorem range_of_m (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : 1/x + 4/y = 1) (h2 : ∃ m : ℝ, x + y < m^2 - 8*m) : 
  ∃ m : ℝ, (m < -1 ∨ m > 9) ∧ x + y < m^2 - 8*m := by
  sorry

end range_of_m_l305_30512


namespace smallest_irreducible_n_l305_30510

def is_irreducible (n k : ℕ) : Prop :=
  Nat.gcd k (n + k + 2) = 1

def all_irreducible (n : ℕ) : Prop :=
  ∀ k : ℕ, 68 ≤ k → k ≤ 133 → is_irreducible n k

theorem smallest_irreducible_n :
  (all_irreducible 65 ∧
   all_irreducible 135 ∧
   (∀ n : ℕ, n < 65 → ¬all_irreducible n) ∧
   (∀ n : ℕ, 65 < n → n < 135 → ¬all_irreducible n)) :=
sorry

end smallest_irreducible_n_l305_30510


namespace domain_range_equal_iff_l305_30570

/-- The function f(x) = √(ax² + bx) where b > 0 -/
noncomputable def f (a b x : ℝ) : ℝ := Real.sqrt (a * x^2 + b * x)

/-- The domain of f -/
def domain (a b : ℝ) : Set ℝ :=
  if a > 0 then {x | x ≤ -b/a ∨ x ≥ 0}
  else if a < 0 then {x | 0 ≤ x ∧ x ≤ -b/a}
  else {x | x ≥ 0}

/-- The range of f -/
def range (a b : ℝ) : Set ℝ :=
  if a ≥ 0 then {y | y ≥ 0}
  else {y | 0 ≤ y ∧ y ≤ b / (2 * Real.sqrt (-a))}

theorem domain_range_equal_iff (b : ℝ) (hb : b > 0) :
  ∀ a : ℝ, domain a b = range a b ↔ a = -4 ∨ a = 0 := by sorry

end domain_range_equal_iff_l305_30570


namespace fraction_sum_proof_l305_30565

theorem fraction_sum_proof (fractions : Finset ℚ) 
  (h1 : fractions.card = 9)
  (h2 : ∀ f ∈ fractions, ∃ n : ℕ+, f = 1 / n)
  (h3 : (fractions.sum id) = 1)
  (h4 : (1 / 3) ∈ fractions ∧ (1 / 7) ∈ fractions ∧ (1 / 9) ∈ fractions ∧ 
        (1 / 11) ∈ fractions ∧ (1 / 33) ∈ fractions)
  (h5 : ∃ f1 f2 f3 f4 : ℚ, f1 ∈ fractions ∧ f2 ∈ fractions ∧ f3 ∈ fractions ∧ f4 ∈ fractions ∧
        ∃ n1 n2 n3 n4 : ℕ, f1 = 1 / n1 ∧ f2 = 1 / n2 ∧ f3 = 1 / n3 ∧ f4 = 1 / n4 ∧
        n1 % 10 = 5 ∧ n2 % 10 = 5 ∧ n3 % 10 = 5 ∧ n4 % 10 = 5) :
  ∃ f1 f2 f3 f4 : ℚ, f1 ∈ fractions ∧ f2 ∈ fractions ∧ f3 ∈ fractions ∧ f4 ∈ fractions ∧
  f1 = 1 / 5 ∧ f2 = 1 / 15 ∧ f3 = 1 / 45 ∧ f4 = 1 / 385 :=
by sorry

end fraction_sum_proof_l305_30565


namespace triangle_max_perimeter_l305_30599

theorem triangle_max_perimeter (a b c : ℝ) (A B C : ℝ) :
  a = 1 →
  2 * Real.cos C + c = 2 * b →
  a + b + c ≤ 3 :=
by sorry

end triangle_max_perimeter_l305_30599


namespace basketball_lineup_combinations_l305_30511

def team_size : ℕ := 12
def quadruplets_size : ℕ := 4
def starters_size : ℕ := 5
def max_quadruplets_in_lineup : ℕ := 2

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem basketball_lineup_combinations :
  (choose (team_size - quadruplets_size) starters_size) +
  (choose quadruplets_size 1 * choose (team_size - quadruplets_size) (starters_size - 1)) +
  (choose quadruplets_size 2 * choose (team_size - quadruplets_size) (starters_size - 2)) = 672 := by
  sorry

end basketball_lineup_combinations_l305_30511


namespace fraction_simplification_l305_30506

theorem fraction_simplification (b x : ℝ) (h : b^2 + x^4 ≠ 0) :
  (Real.sqrt (b^2 + x^4) - (x^4 - b^2) / (2 * Real.sqrt (b^2 + x^4))) / (b^2 + x^4) =
  (3 * b^2 + x^4) / (2 * (b^2 + x^4)^(3/2)) := by
  sorry

end fraction_simplification_l305_30506


namespace food_drive_cans_l305_30525

theorem food_drive_cans (rachel jaydon mark : ℕ) : 
  jaydon = 2 * rachel + 5 →
  mark = 4 * jaydon →
  rachel + jaydon + mark = 135 →
  mark = 100 := by
sorry

end food_drive_cans_l305_30525


namespace x_axis_conditions_l305_30516

/-- A line in 2D space represented by the equation Ax + By + C = 0 -/
structure Line where
  A : ℝ
  B : ℝ
  C : ℝ

/-- Predicate to check if a line is the x-axis -/
def is_x_axis (l : Line) : Prop :=
  ∀ x y : ℝ, l.A * x + l.B * y + l.C = 0 ↔ y = 0

/-- Theorem stating the conditions for a line to be the x-axis -/
theorem x_axis_conditions (l : Line) : 
  is_x_axis l ↔ l.B ≠ 0 ∧ l.A = 0 ∧ l.C = 0 := by sorry

end x_axis_conditions_l305_30516


namespace riddle_guessing_probabilities_l305_30593

-- Define the probabilities of A and B guessing correctly
def prob_A_correct : ℚ := 5/6
def prob_B_correct : ℚ := 3/5

-- Define the probability of A winning in one activity
def prob_A_wins_one : ℚ := prob_A_correct * (1 - prob_B_correct)

-- Define the probability of A winning at least 2 out of 3 activities
def prob_A_wins_two_out_of_three : ℚ :=
  3 * (prob_A_wins_one^2 * (1 - prob_A_wins_one)) + prob_A_wins_one^3

-- State the theorem
theorem riddle_guessing_probabilities :
  prob_A_wins_one = 1/3 ∧ prob_A_wins_two_out_of_three = 7/27 := by
  sorry

end riddle_guessing_probabilities_l305_30593


namespace function_satisfying_condition_l305_30509

theorem function_satisfying_condition (f : ℝ → ℝ) :
  (∀ x y : ℝ, f x * f y = 1 + x * y + f (x + y)) →
  ((∀ x : ℝ, f x = 2 * x - 1) ∨ (∀ x : ℝ, f x = x^2 - 1)) :=
by sorry

end function_satisfying_condition_l305_30509


namespace square_area_ratio_l305_30563

theorem square_area_ratio (y : ℝ) (h : y > 0) :
  (y ^ 2) / ((3 * y) ^ 2) = 1 / 9 := by
sorry

end square_area_ratio_l305_30563


namespace unique_root_sum_l305_30527

-- Define the function f(x) = x^3 - x + 1
def f (x : ℝ) : ℝ := x^3 - x + 1

-- Theorem statement
theorem unique_root_sum (a b : ℤ) : 
  (∃! x : ℝ, a < x ∧ x < b ∧ f x = 0) →  -- Exactly one root in (a, b)
  (b - a = 1) →                          -- b - a = 1
  (a + b = -3) :=                        -- Conclusion: a + b = -3
by
  sorry  -- Proof omitted

end unique_root_sum_l305_30527


namespace specific_factory_production_l305_30522

/-- A factory produces toys with the following parameters:
  * weekly_production: The total number of toys produced in a week
  * work_days: The number of days worked in a week
  * constant_daily_production: Whether the daily production is constant throughout the week
-/
structure ToyFactory where
  weekly_production : ℕ
  work_days : ℕ
  constant_daily_production : Prop

/-- Calculate the daily toy production for a given factory -/
def daily_production (factory : ToyFactory) : ℕ :=
  factory.weekly_production / factory.work_days

/-- Theorem stating that for a factory producing 6500 toys per week,
    working 5 days a week, with constant daily production,
    the daily production is 1300 toys -/
theorem specific_factory_production :
  ∀ (factory : ToyFactory),
    factory.weekly_production = 6500 ∧
    factory.work_days = 5 ∧
    factory.constant_daily_production →
    daily_production factory = 1300 := by
  sorry

end specific_factory_production_l305_30522


namespace year_square_minus_product_l305_30556

theorem year_square_minus_product (n : ℕ) : n^2 - (n - 1) * n = n :=
by sorry

end year_square_minus_product_l305_30556


namespace divisible_by_120_l305_30508

theorem divisible_by_120 (n : ℕ) : ∃ k : ℤ, n * (n^2 - 1) * (n^2 - 5*n + 26) = 120 * k := by
  sorry

end divisible_by_120_l305_30508


namespace smallest_w_value_l305_30532

theorem smallest_w_value : ∃ (w : ℕ+),
  (∀ (x : ℕ+), 
    (2^6 ∣ 2547 * x) ∧ 
    (3^5 ∣ 2547 * x) ∧ 
    (5^4 ∣ 2547 * x) ∧ 
    (7^3 ∣ 2547 * x) ∧ 
    (13^4 ∣ 2547 * x) → 
    w ≤ x) ∧
  (2^6 ∣ 2547 * w) ∧
  (3^5 ∣ 2547 * w) ∧
  (5^4 ∣ 2547 * w) ∧
  (7^3 ∣ 2547 * w) ∧
  (13^4 ∣ 2547 * w) ∧
  w = 1592010000 :=
by sorry

end smallest_w_value_l305_30532


namespace stream_speed_l305_30595

theorem stream_speed (boat_speed : ℝ) (downstream_distance : ℝ) (upstream_distance : ℝ) :
  boat_speed = 30 →
  downstream_distance = 80 →
  upstream_distance = 40 →
  (downstream_distance / (boat_speed + x) = upstream_distance / (boat_speed - x)) →
  x = 10 :=
by
  sorry

end stream_speed_l305_30595


namespace parallel_vectors_iff_y_eq_3_l305_30569

/-- Two vectors in ℝ² are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = (k * b.1, k * b.2)

/-- The problem statement -/
theorem parallel_vectors_iff_y_eq_3 :
  let a : ℝ × ℝ := (4, 2)
  let b : ℝ × ℝ := (6, y)
  parallel a b ↔ y = 3 :=
by sorry

end parallel_vectors_iff_y_eq_3_l305_30569


namespace intersection_of_A_and_B_l305_30505

def A : Set ℤ := {0, 1, 2}
def B : Set ℤ := {x : ℤ | x^2 < 3}

theorem intersection_of_A_and_B : A ∩ B = {0, 1} := by sorry

end intersection_of_A_and_B_l305_30505


namespace t_integer_characterization_t_irreducible_characterization_l305_30574

def t (n : ℤ) : ℚ := (5 * n + 9) / (n - 3)

def is_integer_t (n : ℤ) : Prop := ∃ (k : ℤ), t n = k

def is_irreducible_t (n : ℤ) : Prop :=
  ∃ (a b : ℤ), t n = a / b ∧ Int.gcd a b = 1

theorem t_integer_characterization (n : ℤ) (h : n > 3) :
  is_integer_t n ↔ n ∈ ({4, 5, 6, 7, 9, 11, 15, 27} : Set ℤ) :=
sorry

theorem t_irreducible_characterization (n : ℤ) (h : n > 3) :
  is_irreducible_t n ↔ (∃ (k : ℤ), k > 0 ∧ (n = 6 * k + 1 ∨ n = 6 * k + 5)) :=
sorry

end t_integer_characterization_t_irreducible_characterization_l305_30574


namespace equation_solutions_l305_30583

theorem equation_solutions :
  (∀ x : ℝ, 2 * x^2 + 1 = 3 * x ↔ x = 1 ∨ x = 1/2) ∧
  (∀ x : ℝ, (2*x - 1)^2 = (3 - x)^2 ↔ x = -2 ∨ x = 4/3) := by
  sorry

end equation_solutions_l305_30583


namespace solve_for_p_l305_30587

theorem solve_for_p (n m p : ℚ) 
  (h1 : 5/6 = n/90)
  (h2 : 5/6 = (m + n)/105)
  (h3 : 5/6 = (p - m)/150) : 
  p = 137.5 := by sorry

end solve_for_p_l305_30587


namespace teal_color_survey_l305_30515

theorem teal_color_survey (total : ℕ) (green : ℕ) (both : ℕ) (neither : ℕ) 
  (h_total : total = 150)
  (h_green : green = 90)
  (h_both : both = 45)
  (h_neither : neither = 24) :
  ∃ blue : ℕ, blue = 81 ∧ blue = total - (green - both) - both - neither :=
by sorry

end teal_color_survey_l305_30515


namespace sum_of_xyz_l305_30523

theorem sum_of_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x * y + y * z = 30) (h2 : y * z + z * x = 36) (h3 : z * x + x * y = 42) :
  x + y + z = 13 := by sorry

end sum_of_xyz_l305_30523


namespace library_visitors_theorem_l305_30581

/-- Calculates the total number of visitors to a library in a week given specific conditions --/
theorem library_visitors_theorem (monday_visitors : ℕ) 
  (h1 : monday_visitors = 50)
  (h2 : ∃ tuesday_visitors : ℕ, tuesday_visitors = 2 * monday_visitors)
  (h3 : ∃ wednesday_visitors : ℕ, wednesday_visitors = 2 * monday_visitors)
  (h4 : ∃ thursday_visitors : ℕ, thursday_visitors = 3 * (2 * monday_visitors))
  (h5 : ∃ weekend_visitors : ℕ, weekend_visitors = 3 * 20) :
  monday_visitors + 
  (2 * monday_visitors) + 
  (2 * monday_visitors) + 
  (3 * (2 * monday_visitors)) + 
  (3 * 20) = 610 := by
    sorry

end library_visitors_theorem_l305_30581


namespace max_value_theorem_l305_30575

theorem max_value_theorem (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a^2 + b^2 + c^2 = 1) :
  2*a*b + 2*a*c*Real.sqrt 3 ≤ 1 := by
sorry

end max_value_theorem_l305_30575


namespace election_result_l305_30591

theorem election_result (total_voters : ℝ) (rep_percent : ℝ) (dem_percent : ℝ) 
  (dem_x_vote_percent : ℝ) (x_win_margin : ℝ) :
  rep_percent / dem_percent = 3 / 2 →
  rep_percent + dem_percent = 100 →
  dem_x_vote_percent = 25 →
  x_win_margin = 16.000000000000014 →
  ∃ (rep_x_vote_percent : ℝ),
    rep_x_vote_percent * rep_percent + dem_x_vote_percent * dem_percent = 
    (100 + x_win_margin) / 2 ∧
    rep_x_vote_percent = 80 :=
by sorry

end election_result_l305_30591


namespace ratio_odd_even_divisors_l305_30571

def M : ℕ := 36 * 36 * 85 * 128

def sum_odd_divisors (n : ℕ) : ℕ := sorry

def sum_even_divisors (n : ℕ) : ℕ := sorry

theorem ratio_odd_even_divisors :
  (sum_odd_divisors M) * 4094 = sum_even_divisors M :=
sorry

end ratio_odd_even_divisors_l305_30571


namespace income_relationship_l305_30582

theorem income_relationship (juan tim mart : ℝ) 
  (h1 : mart = 1.4 * tim) 
  (h2 : tim = 0.6 * juan) : 
  mart = 0.84 * juan := by
sorry

end income_relationship_l305_30582


namespace factor_expression_l305_30578

theorem factor_expression (x : ℝ) : 9*x^2 + 3*x = 3*x*(3*x + 1) := by
  sorry

end factor_expression_l305_30578


namespace right_triangle_leg_sum_l305_30536

/-- Given a right triangle with consecutive whole number leg lengths and hypotenuse 29,
    prove that the sum of the leg lengths is 41. -/
theorem right_triangle_leg_sum : 
  ∃ (a b : ℕ), 
    a + 1 = b ∧                   -- legs are consecutive whole numbers
    a^2 + b^2 = 29^2 ∧            -- Pythagorean theorem for hypotenuse 29
    a + b = 41 :=                 -- sum of leg lengths is 41
by sorry

end right_triangle_leg_sum_l305_30536


namespace seventh_power_sum_l305_30547

theorem seventh_power_sum (α β γ : ℂ)
  (h1 : α + β + γ = 2)
  (h2 : α^2 + β^2 + γ^2 = 5)
  (h3 : α^3 + β^3 + γ^3 = 10) :
  α^7 + β^7 + γ^7 = 65.38 := by
  sorry

end seventh_power_sum_l305_30547


namespace two_lines_intersecting_at_distance_5_l305_30589

/-- Given a line and a point, find two lines passing through the point and intersecting the given line at a distance of 5 from the given point. -/
theorem two_lines_intersecting_at_distance_5 :
  ∃ (l₂₁ l₂₂ : ℝ → ℝ → Prop),
    (∀ x y, l₂₁ x y ↔ x = 1) ∧
    (∀ x y, l₂₂ x y ↔ 3 * x + 4 * y + 1 = 0) ∧
    (∀ x y, l₂₁ x y → l₂₁ 1 (-1)) ∧
    (∀ x y, l₂₂ x y → l₂₂ 1 (-1)) ∧
    (∃ x₁ y₁, l₂₁ x₁ y₁ ∧ 2 * x₁ + y₁ - 6 = 0 ∧ (x₁ - 1)^2 + (y₁ + 1)^2 = 5^2) ∧
    (∃ x₂ y₂, l₂₂ x₂ y₂ ∧ 2 * x₂ + y₂ - 6 = 0 ∧ (x₂ - 1)^2 + (y₂ + 1)^2 = 5^2) :=
by sorry


end two_lines_intersecting_at_distance_5_l305_30589


namespace union_of_A_and_B_complement_of_A_l305_30517

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5, 6, 7}

-- Define set A
def A : Set Nat := {2, 4, 5}

-- Define set B
def B : Set Nat := {2, 7}

-- Theorem for A ∪ B
theorem union_of_A_and_B : A ∪ B = {2, 4, 5, 7} := by sorry

-- Theorem for complement of A with respect to U
theorem complement_of_A : (U \ A) = {1, 3, 6, 7} := by sorry

end union_of_A_and_B_complement_of_A_l305_30517


namespace absolute_value_equation_product_l305_30500

theorem absolute_value_equation_product (x : ℝ) : 
  (|18 / x + 4| = 3) → (∃ y : ℝ, (|18 / y + 4| = 3) ∧ x * y = 324 / 7) :=
sorry

end absolute_value_equation_product_l305_30500


namespace union_subset_iff_m_range_no_m_for_equality_l305_30552

-- Define the sets P and S
def P : Set ℝ := {x : ℝ | x^2 - 8*x - 20 ≤ 0}
def S (m : ℝ) : Set ℝ := {x : ℝ | |x - 1| ≤ m}

-- Theorem 1: (P ∪ S) ⊆ P if and only if m ∈ (-∞, 3]
theorem union_subset_iff_m_range (m : ℝ) : 
  (P ∪ S m) ⊆ P ↔ m ≤ 3 :=
sorry

-- Theorem 2: There does not exist an m such that P = S
theorem no_m_for_equality : 
  ¬∃ m : ℝ, P = S m :=
sorry

end union_subset_iff_m_range_no_m_for_equality_l305_30552


namespace quadratic_inequality_solution_set_l305_30566

theorem quadratic_inequality_solution_set (m : ℝ) :
  m > 2 → ∀ x : ℝ, x^2 - 2*x + m > 0 :=
by
  sorry

end quadratic_inequality_solution_set_l305_30566


namespace job_completion_time_l305_30551

/-- Given workers A, B, and C who can complete a job individually in 18, 30, and 45 days respectively,
    prove that they can complete the job together in 9 days. -/
theorem job_completion_time (a b c : ℝ) (ha : a = 18) (hb : b = 30) (hc : c = 45) :
  (1 / a + 1 / b + 1 / c)⁻¹ = 9 := by
  sorry

end job_completion_time_l305_30551


namespace line_intercept_sum_l305_30554

/-- A line passing through (5, 3) with slope 3 has x-intercept + y-intercept = -8 -/
theorem line_intercept_sum : ∀ (f : ℝ → ℝ),
  (f 5 = 3) →                        -- The line passes through (5, 3)
  (∀ x y, f y - f x = 3 * (y - x)) → -- The slope is 3
  (∃ a, f a = 0) →                   -- x-intercept exists
  (∃ b, f 0 = b) →                   -- y-intercept exists
  (∃ a b, f a = 0 ∧ f 0 = b ∧ a + b = -8) :=
by sorry

end line_intercept_sum_l305_30554


namespace banana_orange_equivalence_l305_30568

theorem banana_orange_equivalence :
  ∀ (banana_value orange_value : ℝ),
  (3/4 * 12 * banana_value = 6 * orange_value) →
  (1/4 * 12 * banana_value = 2 * orange_value) :=
by
  sorry

end banana_orange_equivalence_l305_30568
