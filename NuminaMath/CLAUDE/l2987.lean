import Mathlib

namespace NUMINAMATH_CALUDE_third_term_geometric_series_l2987_298760

/-- Theorem: Third term of a specific geometric series -/
theorem third_term_geometric_series
  (q : ℝ) 
  (h₁ : |q| < 1)
  (h₂ : 2 / (1 - q) = 8 / 5)
  (h₃ : 2 * q = -1 / 2) :
  2 * q^2 = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_third_term_geometric_series_l2987_298760


namespace NUMINAMATH_CALUDE_only_two_valid_plans_l2987_298704

/-- Represents a deployment plan for trucks -/
structure DeploymentPlan where
  typeA : ℕ
  typeB : ℕ

/-- Checks if a deployment plan is valid according to the given conditions -/
def isValidPlan (p : DeploymentPlan) : Prop :=
  p.typeA + p.typeB = 70 ∧
  p.typeB ≤ 3 * p.typeA ∧
  25 * p.typeA + 15 * p.typeB ≤ 1245

/-- The set of all valid deployment plans -/
def validPlans : Set DeploymentPlan :=
  {p | isValidPlan p}

/-- The theorem stating that there are only two valid deployment plans -/
theorem only_two_valid_plans :
  validPlans = {DeploymentPlan.mk 18 52, DeploymentPlan.mk 19 51} :=
by
  sorry

end NUMINAMATH_CALUDE_only_two_valid_plans_l2987_298704


namespace NUMINAMATH_CALUDE_smallest_n_equality_l2987_298736

def C (n : ℕ) : ℚ := 989 * (1 - (1/3)^n) / (1 - 1/3)

def D (n : ℕ) : ℚ := 2744 * (1 - (-1/3)^n) / (1 + 1/3)

theorem smallest_n_equality : ∃ (n : ℕ), n > 0 ∧ C n = D n ∧ ∀ (m : ℕ), m > 0 ∧ m < n → C m ≠ D m :=
  sorry

end NUMINAMATH_CALUDE_smallest_n_equality_l2987_298736


namespace NUMINAMATH_CALUDE_two_number_difference_l2987_298770

theorem two_number_difference (x y : ℝ) : 
  x + y = 40 → 3 * y - 4 * x = 20 → |y - x| = 11.42 := by
  sorry

end NUMINAMATH_CALUDE_two_number_difference_l2987_298770


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l2987_298752

/-- Given an ellipse with semi-major axis a and semi-minor axis b, 
    a chord of the ellipse, and the midpoint of the chord,
    prove that the eccentricity of the ellipse is √5/5 -/
theorem ellipse_eccentricity 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hab : a > b) 
  (hchord : ∀ x y : ℝ, x - y + 5 = 0 → ∃ t : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1) 
  (hmidpoint : ∃ x1 y1 x2 y2 : ℝ, 
    (x1^2 / a^2) + (y1^2 / b^2) = 1 ∧ 
    (x2^2 / a^2) + (y2^2 / b^2) = 1 ∧ 
    x1 - y1 + 5 = 0 ∧ 
    x2 - y2 + 5 = 0 ∧ 
    (x1 + x2) / 2 = -4 ∧ 
    (y1 + y2) / 2 = 1) : 
  (Real.sqrt (1 - b^2 / a^2)) = Real.sqrt 5 / 5 := by
sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l2987_298752


namespace NUMINAMATH_CALUDE_existence_of_alpha_beta_l2987_298707

-- Define the Intermediate Value Property
def has_intermediate_value_property (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ y, (f a < y ∧ y < f b) ∨ (f b < y ∧ y < f a) → ∃ c, a < c ∧ c < b ∧ f c = y

-- State the theorem
theorem existence_of_alpha_beta
  (f : ℝ → ℝ) (a b : ℝ) (h_ivp : has_intermediate_value_property f a b)
  (h_sign : f a * f b < 0) :
  ∃ α β, a < α ∧ α < β ∧ β < b ∧ f α + f β = f α * f β :=
sorry

end NUMINAMATH_CALUDE_existence_of_alpha_beta_l2987_298707


namespace NUMINAMATH_CALUDE_reflection_of_M_across_y_axis_l2987_298790

/-- Reflects a point across the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

/-- The original point -/
def M : ℝ × ℝ := (3, 2)

theorem reflection_of_M_across_y_axis :
  reflect_y M = (-3, 2) := by sorry

end NUMINAMATH_CALUDE_reflection_of_M_across_y_axis_l2987_298790


namespace NUMINAMATH_CALUDE_correct_num_shirts_l2987_298778

/-- The number of different colored neckties -/
def num_neckties : ℕ := 6

/-- The probability that all boxes contain a necktie and a shirt of the same color -/
def match_probability : ℚ := 1 / 120

/-- The number of different colored shirts -/
def num_shirts : ℕ := 2

/-- Theorem stating that given the number of neckties and the match probability,
    the number of shirts is correct -/
theorem correct_num_shirts :
  (1 : ℚ) / num_shirts ^ num_neckties = match_probability := by sorry

end NUMINAMATH_CALUDE_correct_num_shirts_l2987_298778


namespace NUMINAMATH_CALUDE_compound_interest_calculation_l2987_298728

/-- Compound interest calculation -/
theorem compound_interest_calculation 
  (initial_deposit : ℝ) 
  (interest_rate : ℝ) 
  (time_period : ℕ) 
  (h1 : initial_deposit = 20000)
  (h2 : interest_rate = 0.03)
  (h3 : time_period = 5) :
  initial_deposit * (1 + interest_rate) ^ time_period = 
    20000 * (1 + 0.03) ^ 5 :=
by sorry

end NUMINAMATH_CALUDE_compound_interest_calculation_l2987_298728


namespace NUMINAMATH_CALUDE_cos_18_degrees_l2987_298708

theorem cos_18_degrees : Real.cos (18 * π / 180) = (1 + Real.sqrt 5) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_18_degrees_l2987_298708


namespace NUMINAMATH_CALUDE_real_part_of_z_l2987_298710

theorem real_part_of_z (z : ℂ) (h : Complex.I * (z + 1) = -3 + 2 * Complex.I) : 
  z.re = 1 := by sorry

end NUMINAMATH_CALUDE_real_part_of_z_l2987_298710


namespace NUMINAMATH_CALUDE_line_through_point_with_equal_intercepts_l2987_298725

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line in 2D space
structure Line2D where
  slope : ℝ
  intercept : ℝ

-- Function to check if a line passes through a point
def passesThrough (l : Line2D) (p : Point2D) : Prop :=
  p.y = l.slope * p.x + l.intercept

-- Function to check if a line has equal intercepts on both axes
def hasEqualIntercepts (l : Line2D) : Prop :=
  l.intercept = -l.slope * l.intercept ∨ l.slope = -1

-- The main theorem
theorem line_through_point_with_equal_intercepts :
  ∃ (l1 l2 : Line2D),
    (passesThrough l1 ⟨-2, 3⟩ ∧ hasEqualIntercepts l1) ∧
    (passesThrough l2 ⟨-2, 3⟩ ∧ hasEqualIntercepts l2) ∧
    ((l1.slope = -3/2 ∧ l1.intercept = 0) ∨ (l2.slope = -1 ∧ l2.intercept = 1)) :=
by sorry

end NUMINAMATH_CALUDE_line_through_point_with_equal_intercepts_l2987_298725


namespace NUMINAMATH_CALUDE_tangent_line_equation_l2987_298781

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 3*x

-- Define the point of tangency
def P : ℝ × ℝ := (1, 4)

-- State the theorem
theorem tangent_line_equation :
  let m := (2 * P.1 + 3) -- Slope of the tangent line
  (5 : ℝ) * x - y - 1 = 0 ↔ y - P.2 = m * (x - P.1) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l2987_298781


namespace NUMINAMATH_CALUDE_apple_orange_ratio_l2987_298735

/-- Given a basket of fruit with apples and oranges, prove the ratio of apples to oranges --/
theorem apple_orange_ratio (total_fruit : ℕ) (oranges : ℕ) : 
  total_fruit = 40 → oranges = 10 → (total_fruit - oranges) / oranges = 3 := by
  sorry

#check apple_orange_ratio

end NUMINAMATH_CALUDE_apple_orange_ratio_l2987_298735


namespace NUMINAMATH_CALUDE_max_area_isosceles_trapezoid_in_circle_l2987_298723

/-- An isosceles trapezoid inscribed in a circle -/
structure IsoscelesTrapezoidInCircle where
  r : ℝ  -- radius of the circle
  x : ℝ  -- length of the legs of the trapezoid
  a : ℝ  -- length of one parallel side
  d : ℝ  -- length of the other parallel side
  h : x ≥ 2 * r  -- condition that legs are at least as long as the diameter
  tangent : a + d = 2 * x  -- condition for tangent quadrilateral

/-- The area of an isosceles trapezoid inscribed in a circle -/
def area (t : IsoscelesTrapezoidInCircle) : ℝ := 2 * t.x * t.r

/-- Theorem: The maximum area of an isosceles trapezoid inscribed in a circle with radius r is 4r^2 -/
theorem max_area_isosceles_trapezoid_in_circle (t : IsoscelesTrapezoidInCircle) :
  area t ≤ 4 * t.r^2 :=
sorry

end NUMINAMATH_CALUDE_max_area_isosceles_trapezoid_in_circle_l2987_298723


namespace NUMINAMATH_CALUDE_simplify_t_l2987_298777

theorem simplify_t (t : ℝ) : t = 1 / (3 - Real.rpow 3 (1/3)) → t = (3 + Real.rpow 3 (1/3)) / 6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_t_l2987_298777


namespace NUMINAMATH_CALUDE_find_other_number_l2987_298749

theorem find_other_number (a b : ℕ+) (h1 : Nat.lcm a b = 5040) (h2 : Nat.gcd a b = 12) (h3 : a = 240) : b = 252 := by
  sorry

end NUMINAMATH_CALUDE_find_other_number_l2987_298749


namespace NUMINAMATH_CALUDE_achieve_any_distribution_l2987_298767

-- Define the Student type
def Student : Type := ℕ

-- Define the friendship relation
def IsFriend (s1 s2 : Student) : Prop := sorry

-- Define the candy distribution
def CandyCount : Student → Fin 7 := sorry

-- Define the property of friendship for the set of students
def FriendshipProperty (students : Set Student) : Prop :=
  ∀ s1 s2 : Student, s1 ∈ students → s2 ∈ students → s1 ≠ s2 →
    ∃ s3 ∈ students, (IsFriend s3 s1 ∧ ¬IsFriend s3 s2) ∨ (IsFriend s3 s2 ∧ ¬IsFriend s3 s1)

-- Define a step in the candy distribution process
def DistributionStep (students : Set Student) (initial : Student → Fin 7) : Student → Fin 7 := sorry

-- Theorem: Any desired candy distribution can be achieved in finitely many steps
theorem achieve_any_distribution 
  (students : Set Student) 
  (h_finite : Finite students) 
  (h_friendship : FriendshipProperty students) 
  (initial : Student → Fin 7) 
  (target : Student → Fin 7) :
  ∃ n : ℕ, ∃ steps : Fin n → (Set Student), 
    (DistributionStep students)^[n] initial = target := by
  sorry

end NUMINAMATH_CALUDE_achieve_any_distribution_l2987_298767


namespace NUMINAMATH_CALUDE_unique_solution_implies_relation_l2987_298738

-- Define the system of equations
def system (a b x y : ℝ) : Prop :=
  y = x^2 + a*x + b ∧ x = y^2 + a*y + b

-- Theorem statement
theorem unique_solution_implies_relation (a b : ℝ) :
  (∃! p : ℝ × ℝ, system a b p.1 p.2) →
  a^2 = 2*(a + 2*b) - 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_implies_relation_l2987_298738


namespace NUMINAMATH_CALUDE_pen_count_is_31_l2987_298793

/-- The number of pens after a series of events --/
def final_pen_count (initial : ℕ) (mike_gives : ℕ) (cindy_multiplier : ℕ) (sharon_takes : ℕ) : ℕ :=
  ((initial + mike_gives) * cindy_multiplier) - sharon_takes

/-- Theorem stating that given the initial conditions, the final number of pens is 31 --/
theorem pen_count_is_31 : final_pen_count 5 20 2 19 = 31 := by
  sorry

end NUMINAMATH_CALUDE_pen_count_is_31_l2987_298793


namespace NUMINAMATH_CALUDE_constant_prime_sequence_l2987_298761

theorem constant_prime_sequence (p : ℕ → ℕ) (k : ℤ) :
  (∀ n, n ≥ 1 → Nat.Prime (p n)) →
  (∀ n, n ≥ 1 → p (n + 2) = p (n + 1) + p n + k) →
  ∃ c, ∀ n, n ≥ 1 → p n = c :=
sorry

end NUMINAMATH_CALUDE_constant_prime_sequence_l2987_298761


namespace NUMINAMATH_CALUDE_lcm_factor_14_l2987_298765

theorem lcm_factor_14 (A B : ℕ+) (h1 : Nat.gcd A B = 16) (h2 : A = 224) :
  ∃ (X Y : ℕ+), Nat.lcm A B = 16 * X * Y ∧ (X = 14 ∨ Y = 14) := by
sorry

end NUMINAMATH_CALUDE_lcm_factor_14_l2987_298765


namespace NUMINAMATH_CALUDE_cubic_function_coefficient_l2987_298787

/-- Given a cubic function f(x) = ax^3 + bx^2 + cx + d, 
    if f(-1) = 0, f(1) = 0, and f(0) = 2, then b = -2 -/
theorem cubic_function_coefficient (a b c d : ℝ) : 
  let f := λ x : ℝ => a * x^3 + b * x^2 + c * x + d
  (f (-1) = 0) → (f 1 = 0) → (f 0 = 2) → b = -2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_coefficient_l2987_298787


namespace NUMINAMATH_CALUDE_dan_has_16_balloons_l2987_298724

/-- The number of red balloons that Fred has -/
def fred_balloons : ℕ := 10

/-- The number of red balloons that Sam has -/
def sam_balloons : ℕ := 46

/-- The total number of red balloons -/
def total_balloons : ℕ := 72

/-- The number of red balloons that Dan has -/
def dan_balloons : ℕ := total_balloons - (fred_balloons + sam_balloons)

theorem dan_has_16_balloons : dan_balloons = 16 := by
  sorry

end NUMINAMATH_CALUDE_dan_has_16_balloons_l2987_298724


namespace NUMINAMATH_CALUDE_A_D_independent_l2987_298714

-- Define the sample space
def Ω : Type := Fin 6 × Fin 6

-- Define the probability measure
noncomputable def P : Set Ω → ℝ := sorry

-- Define events A and D
def A : Set Ω := {ω : Ω | ω.1 = 0}
def D : Set Ω := {ω : Ω | ω.1.val + ω.2.val = 6}

-- State the theorem
theorem A_D_independent : P (A ∩ D) = P A * P D := by sorry

end NUMINAMATH_CALUDE_A_D_independent_l2987_298714


namespace NUMINAMATH_CALUDE_third_term_value_l2987_298729

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem third_term_value (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 1 = 23 →
  a 6 = 53 →
  a 3 = 38 :=
by sorry

end NUMINAMATH_CALUDE_third_term_value_l2987_298729


namespace NUMINAMATH_CALUDE_poster_area_is_28_l2987_298709

/-- The area of a rectangular poster -/
def poster_area (width height : ℝ) : ℝ := width * height

/-- Theorem: The area of a rectangular poster with width 4 inches and height 7 inches is 28 square inches -/
theorem poster_area_is_28 : poster_area 4 7 = 28 := by
  sorry

end NUMINAMATH_CALUDE_poster_area_is_28_l2987_298709


namespace NUMINAMATH_CALUDE_coins_given_to_laura_l2987_298718

def coins_to_laura (piggy_bank : ℕ) (brother : ℕ) (father : ℕ) (final_count : ℕ) : ℕ :=
  piggy_bank + brother + father - final_count

theorem coins_given_to_laura :
  coins_to_laura 15 13 8 15 = 21 := by
  sorry

end NUMINAMATH_CALUDE_coins_given_to_laura_l2987_298718


namespace NUMINAMATH_CALUDE_special_numbers_l2987_298719

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def satisfies_condition (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d ∣ n → is_prime (d^2 - d + 1) ∧ is_prime (d^2 + d + 1)

theorem special_numbers :
  ∀ n : ℕ, satisfies_condition n ↔ n = 2 ∨ n = 3 ∨ n = 6 :=
sorry

end NUMINAMATH_CALUDE_special_numbers_l2987_298719


namespace NUMINAMATH_CALUDE_cost_of_thousand_gum_l2987_298720

/-- The cost of a single piece of gum in cents -/
def cost_of_one_gum : ℕ := 1

/-- The number of pieces of gum -/
def num_gum : ℕ := 1000

/-- The number of cents in a dollar -/
def cents_per_dollar : ℕ := 100

/-- The cost of multiple pieces of gum in dollars -/
def cost_in_dollars (n : ℕ) : ℚ :=
  (n * cost_of_one_gum : ℚ) / cents_per_dollar

theorem cost_of_thousand_gum :
  cost_in_dollars num_gum = 10 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_thousand_gum_l2987_298720


namespace NUMINAMATH_CALUDE_concert_ticket_discount_l2987_298742

theorem concert_ticket_discount (ticket_price : ℕ) (total_tickets : ℕ) (total_paid : ℕ) 
  (h1 : ticket_price = 40)
  (h2 : total_tickets = 12)
  (h3 : total_paid = 476)
  (h4 : total_tickets > 10) : 
  (ticket_price * total_tickets - total_paid) / (ticket_price * (total_tickets - 10)) * 100 = 5 := by
  sorry

end NUMINAMATH_CALUDE_concert_ticket_discount_l2987_298742


namespace NUMINAMATH_CALUDE_polynomial_with_negative_roots_l2987_298732

/-- A polynomial of degree 4 with integer coefficients -/
structure Polynomial4 where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ

/-- The sum of coefficients of the polynomial -/
def Polynomial4.sum (p : Polynomial4) : ℤ :=
  p.a + p.b + p.c + p.d

/-- Predicate to check if all roots of the polynomial are negative integers -/
def has_all_negative_integer_roots (p : Polynomial4) : Prop :=
  ∃ (s₁ s₂ s₃ s₄ : ℕ), 
    s₁ > 0 ∧ s₂ > 0 ∧ s₃ > 0 ∧ s₄ > 0 ∧
    p.a = s₁ + s₂ + s₃ + s₄ ∧
    p.b = s₁*s₂ + s₁*s₃ + s₁*s₄ + s₂*s₃ + s₂*s₄ + s₃*s₄ ∧
    p.c = s₁*s₂*s₃ + s₁*s₂*s₄ + s₁*s₃*s₄ + s₂*s₃*s₄ ∧
    p.d = s₁*s₂*s₃*s₄

theorem polynomial_with_negative_roots (p : Polynomial4) 
  (h1 : has_all_negative_integer_roots p) 
  (h2 : p.sum = 2003) : 
  p.d = 1992 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_with_negative_roots_l2987_298732


namespace NUMINAMATH_CALUDE_variance_linear_transform_l2987_298780

-- Define the variance of a dataset
def variance (data : List ℝ) : ℝ := sorry

-- Define a linear transformation of a dataset
def linearTransform (a b : ℝ) (data : List ℝ) : List ℝ := 
  data.map (fun x => a * x + b)

theorem variance_linear_transform (data : List ℝ) :
  variance data = 2 → variance (linearTransform 3 (-2) data) = 18 := by
  sorry

end NUMINAMATH_CALUDE_variance_linear_transform_l2987_298780


namespace NUMINAMATH_CALUDE_complement_of_M_l2987_298763

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {4, 5}

theorem complement_of_M :
  (U \ M) = {1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_l2987_298763


namespace NUMINAMATH_CALUDE_polynomial_divisibility_sum_l2987_298739

/-- A complex number ω such that ω^2 + ω + 1 = 0 -/
def ω : ℂ := sorry

/-- The polynomial x^103 + Cx^2 + Dx + E -/
def f (C D E : ℝ) (x : ℂ) : ℂ := x^103 + C*x^2 + D*x + E

/-- The polynomial x^2 + x + 1 -/
def g (x : ℂ) : ℂ := x^2 + x + 1

theorem polynomial_divisibility_sum (C D E : ℝ) :
  (∀ x, g x = 0 → f C D E x = 0) → C + D + E = 2 := by sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_sum_l2987_298739


namespace NUMINAMATH_CALUDE_hotel_rooms_l2987_298751

theorem hotel_rooms (total_lamps : ℕ) (lamps_per_room : ℕ) (h1 : total_lamps = 147) (h2 : lamps_per_room = 7) :
  total_lamps / lamps_per_room = 21 := by
  sorry

end NUMINAMATH_CALUDE_hotel_rooms_l2987_298751


namespace NUMINAMATH_CALUDE_electricity_constant_is_correct_l2987_298746

/-- Represents the relationship between electricity bill and consumption -/
def electricity_equation (x y : ℝ) : Prop := y = 0.54 * x

/-- The constant in the electricity equation -/
def electricity_constant : ℝ := 0.54

/-- Theorem stating that the constant in the electricity equation is 0.54 -/
theorem electricity_constant_is_correct :
  ∀ x y : ℝ, electricity_equation x y → 
  ∃ c : ℝ, (∀ x' y' : ℝ, electricity_equation x' y' → y' = c * x') ∧ c = electricity_constant :=
sorry

end NUMINAMATH_CALUDE_electricity_constant_is_correct_l2987_298746


namespace NUMINAMATH_CALUDE_floor_y_length_l2987_298731

/-- Represents a rectangular floor with length and width -/
structure RectangularFloor where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular floor -/
def area (floor : RectangularFloor) : ℝ :=
  floor.length * floor.width

theorem floor_y_length 
  (floor_x floor_y : RectangularFloor)
  (equal_area : area floor_x = area floor_y)
  (x_dimensions : floor_x.length = 10 ∧ floor_x.width = 18)
  (y_width : floor_y.width = 9) :
  floor_y.length = 20 := by
sorry

end NUMINAMATH_CALUDE_floor_y_length_l2987_298731


namespace NUMINAMATH_CALUDE_sum_of_distinct_remainders_div_13_l2987_298762

def remainders : List Nat :=
  (List.range 10).map (fun n => (n + 1)^2 % 13)

def distinct_remainders : List Nat :=
  remainders.eraseDups

theorem sum_of_distinct_remainders_div_13 :
  (distinct_remainders.sum) / 13 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_distinct_remainders_div_13_l2987_298762


namespace NUMINAMATH_CALUDE_card_combinations_l2987_298741

theorem card_combinations : Nat.choose 40 7 = 1860480 := by sorry

end NUMINAMATH_CALUDE_card_combinations_l2987_298741


namespace NUMINAMATH_CALUDE_work_completion_rate_l2987_298703

/-- Given that A can finish a work in 12 days and B can do the same work in half the time taken by A,
    prove that working together, they can finish 1/4 of the work in a day. -/
theorem work_completion_rate (days_A : ℕ) (days_B : ℕ) : 
  days_A = 12 →
  days_B = days_A / 2 →
  (1 : ℚ) / days_A + (1 : ℚ) / days_B = (1 : ℚ) / 4 :=
by sorry

end NUMINAMATH_CALUDE_work_completion_rate_l2987_298703


namespace NUMINAMATH_CALUDE_years_B_is_two_l2987_298730

/-- Represents the loan details and interest calculation --/
structure LoanDetails where
  principal_B : ℕ := 5000
  principal_C : ℕ := 3000
  years_C : ℕ := 4
  rate : ℚ := 1/10
  total_interest : ℕ := 2200

/-- Calculates the number of years A lent to B --/
def years_B (loan : LoanDetails) : ℚ :=
  (loan.total_interest - (loan.principal_C * loan.rate * loan.years_C)) / (loan.principal_B * loan.rate)

/-- Theorem stating that the number of years A lent to B is 2 --/
theorem years_B_is_two (loan : LoanDetails) : years_B loan = 2 := by
  sorry

end NUMINAMATH_CALUDE_years_B_is_two_l2987_298730


namespace NUMINAMATH_CALUDE_bailey_shot_percentage_l2987_298794

theorem bailey_shot_percentage (total_shots : ℕ) (scored_shots : ℕ) 
  (h1 : total_shots = 8) (h2 : scored_shots = 6) : 
  (1 - scored_shots / total_shots) * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_bailey_shot_percentage_l2987_298794


namespace NUMINAMATH_CALUDE_circle_fixed_point_l2987_298764

theorem circle_fixed_point (a : ℝ) (ha : a ≠ 1) :
  ∀ x y : ℝ, x^2 + y^2 - 2*a*x + 2*(a-2)*y + 2 = 0 → x = 1 ∧ y = 1 :=
by sorry

end NUMINAMATH_CALUDE_circle_fixed_point_l2987_298764


namespace NUMINAMATH_CALUDE_smallest_c_value_l2987_298795

theorem smallest_c_value (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (∀ x, a * Real.cos (b * x + c) ≤ a * Real.cos c) → c = 0 := by
  sorry

end NUMINAMATH_CALUDE_smallest_c_value_l2987_298795


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l2987_298705

theorem arithmetic_calculation : 2^2 + 3 * 4 - 5 + (6 - 1) = 16 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l2987_298705


namespace NUMINAMATH_CALUDE_sum_of_factors_72_l2987_298759

/-- Sum of positive factors of a natural number -/
def sumOfFactors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id

/-- Theorem: The sum of all positive factors of 72 is 195 -/
theorem sum_of_factors_72 : sumOfFactors 72 = 195 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_factors_72_l2987_298759


namespace NUMINAMATH_CALUDE_geometric_sequence_b_value_l2987_298747

theorem geometric_sequence_b_value (b : ℝ) (h1 : b > 0) 
  (h2 : ∃ r : ℝ, r > 0 ∧ b = 30 * r ∧ 9/4 = b * r) : 
  b = 3 * Real.sqrt 30 / 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_b_value_l2987_298747


namespace NUMINAMATH_CALUDE_log_difference_equality_l2987_298798

theorem log_difference_equality : 
  Real.sqrt (Real.log 12 / Real.log 4 - Real.log 12 / Real.log 5) = 
  Real.sqrt ((Real.log 12 * Real.log 1.25) / (Real.log 4 * Real.log 5)) := by
  sorry

end NUMINAMATH_CALUDE_log_difference_equality_l2987_298798


namespace NUMINAMATH_CALUDE_constant_sequence_conditions_l2987_298758

/-- A sequence of real numbers -/
def Sequence := ℕ → ℝ

/-- A sequence is constant if all its terms are equal -/
def is_constant (a : Sequence) : Prop :=
  ∀ n m : ℕ, a n = a m

/-- A sequence is geometric if the ratio of consecutive terms is constant -/
def is_geometric (a : Sequence) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

/-- A sequence is arithmetic if the difference of consecutive terms is constant -/
def is_arithmetic (a : Sequence) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem constant_sequence_conditions (k b : ℝ) (hk : k ≠ 0) (hb : b ≠ 0) (a : Sequence) :
  (is_geometric a ∧ is_geometric (fun n ↦ k * a n + b)) ∨
  (is_arithmetic a ∧ is_geometric (fun n ↦ k * a n + b)) ∨
  (is_geometric a ∧ is_arithmetic (fun n ↦ k * a n + b))
  → is_constant a := by
  sorry

end NUMINAMATH_CALUDE_constant_sequence_conditions_l2987_298758


namespace NUMINAMATH_CALUDE_square_area_l2987_298727

-- Define the square
structure Square where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

-- Define the properties of the square
def is_valid_square (s : Square) : Prop :=
  let (x₀, _) := s.A
  let (_, y₁) := s.B
  let (x₂, y₂) := s.C
  let (_, y₃) := s.D
  x₀ = x₂ ∧                   -- A and C on same vertical line
  y₁ = 2 ∧ y₂ = 8 ∧ y₃ = 6 ∧  -- y-coordinates are 0, 2, 6, 8
  s.A.2 = 0 ∧ s.C.2 = 8       -- A has y-coordinate 0, C has y-coordinate 8

-- Theorem statement
theorem square_area (s : Square) (h : is_valid_square s) : 
  (s.C.2 - s.A.2) * (s.C.2 - s.A.2) = 64 := by
  sorry

end NUMINAMATH_CALUDE_square_area_l2987_298727


namespace NUMINAMATH_CALUDE_intersection_M_N_l2987_298702

def M : Set ℕ := {0, 1, 2}

def N : Set ℕ := {x : ℕ | ∃ a ∈ M, x = 2 * a}

theorem intersection_M_N : M ∩ N = {0, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2987_298702


namespace NUMINAMATH_CALUDE_line_slope_l2987_298713

/-- A straight line in the xy-plane with y-intercept 4 and passing through (199, 800) has slope 4 -/
theorem line_slope (m : ℝ) : 
  (∃ (f : ℝ → ℝ), (∀ x, f x = m * x + 4) ∧ f 199 = 800) → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_l2987_298713


namespace NUMINAMATH_CALUDE_product_remainder_mod_five_l2987_298737

theorem product_remainder_mod_five :
  (2024 * 1980 * 1848 * 1720) % 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_mod_five_l2987_298737


namespace NUMINAMATH_CALUDE_modular_inverse_13_mod_101_l2987_298772

theorem modular_inverse_13_mod_101 : ∃ x : ℤ, (13 * x) % 101 = 1 ∧ 0 ≤ x ∧ x < 101 :=
by
  use 70
  sorry

end NUMINAMATH_CALUDE_modular_inverse_13_mod_101_l2987_298772


namespace NUMINAMATH_CALUDE_ravens_age_l2987_298740

theorem ravens_age (phoebe_age : ℕ) (raven_age : ℕ) : 
  phoebe_age = 10 →
  raven_age + 5 = 4 * (phoebe_age + 5) →
  raven_age = 55 := by
sorry

end NUMINAMATH_CALUDE_ravens_age_l2987_298740


namespace NUMINAMATH_CALUDE_max_value_fraction_l2987_298754

theorem max_value_fraction (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  ∃ (M : ℝ), M = 1/4 ∧ 
  (x * y * z * (x + y + z)) / ((x + z)^2 * (z + y)^2) ≤ M ∧
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    (a * b * c * (a + b + c)) / ((a + c)^2 * (c + b)^2) = M :=
by sorry

end NUMINAMATH_CALUDE_max_value_fraction_l2987_298754


namespace NUMINAMATH_CALUDE_quadratic_integer_roots_l2987_298768

/-- The polynomial x^2 + ax + 2a has two distinct integer roots if and only if a = -1 or a = 9 -/
theorem quadratic_integer_roots (a : ℤ) : 
  (∃ x y : ℤ, x ≠ y ∧ x^2 + a*x + 2*a = 0 ∧ y^2 + a*y + 2*a = 0) ↔ (a = -1 ∨ a = 9) :=
sorry

end NUMINAMATH_CALUDE_quadratic_integer_roots_l2987_298768


namespace NUMINAMATH_CALUDE_rhombus_longest_diagonal_l2987_298743

/-- Given a rhombus with area 144 and diagonal ratio 4:3, prove its longest diagonal is 8√6 -/
theorem rhombus_longest_diagonal (area : ℝ) (ratio : ℝ) (long_diag : ℝ) : 
  area = 144 → ratio = 4/3 → long_diag = 8 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_longest_diagonal_l2987_298743


namespace NUMINAMATH_CALUDE_intersection_segment_length_l2987_298726

-- Define the circles
def circle_O₁ (x y : ℝ) : Prop := x^2 + y^2 = 5
def circle_O₂ (x y m : ℝ) : Prop := (x - m)^2 + y^2 = 20

-- Define the intersection points
def intersection_points (A B : ℝ × ℝ) (m : ℝ) : Prop :=
  circle_O₁ A.1 A.2 ∧ circle_O₂ A.1 A.2 m ∧
  circle_O₁ B.1 B.2 ∧ circle_O₂ B.1 B.2 m

-- Define perpendicular tangents at point A
def perpendicular_tangents (A : ℝ × ℝ) (m : ℝ) : Prop :=
  ∃ (t₁ t₂ : ℝ × ℝ → ℝ × ℝ), 
    (t₁ A = (0, 0)) ∧ (t₂ A = (m, 0)) ∧ 
    (t₁ A • t₂ A = 0)  -- Dot product of tangent vectors is zero

-- Theorem statement
theorem intersection_segment_length 
  (A B : ℝ × ℝ) (m : ℝ) : 
  intersection_points A B m → 
  perpendicular_tangents A m → 
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 4 :=
sorry

end NUMINAMATH_CALUDE_intersection_segment_length_l2987_298726


namespace NUMINAMATH_CALUDE_bike_cost_l2987_298734

/-- The cost of Jenn's bike given her savings in quarters and leftover money -/
theorem bike_cost (num_jars : ℕ) (quarters_per_jar : ℕ) (quarter_value : ℚ) (leftover : ℕ) : 
  num_jars = 5 →
  quarters_per_jar = 160 →
  quarter_value = 1/4 →
  leftover = 20 →
  (num_jars * quarters_per_jar : ℕ) * quarter_value - leftover = 200 := by
  sorry

end NUMINAMATH_CALUDE_bike_cost_l2987_298734


namespace NUMINAMATH_CALUDE_income_expenditure_ratio_l2987_298715

/-- Represents the financial state of a person --/
structure FinancialState where
  income : ℕ
  savings : ℕ

/-- Calculates the expenditure given income and savings --/
def calculateExpenditure (fs : FinancialState) : ℕ :=
  fs.income - fs.savings

/-- Calculates the ratio of two numbers --/
def calculateRatio (a b : ℕ) : ℕ × ℕ :=
  let gcd := Nat.gcd a b
  (a / gcd, b / gcd)

/-- Theorem stating the ratio of income to expenditure --/
theorem income_expenditure_ratio (fs : FinancialState) 
  (h1 : fs.income = 18000) 
  (h2 : fs.savings = 2000) : 
  calculateRatio fs.income (calculateExpenditure fs) = (9, 8) := by
  sorry

end NUMINAMATH_CALUDE_income_expenditure_ratio_l2987_298715


namespace NUMINAMATH_CALUDE_max_abs_diff_f_g_l2987_298722

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2
def g (x : ℝ) : ℝ := x^3

-- Define the absolute difference function
def absDiff (x : ℝ) : ℝ := |f x - g x|

-- State the theorem
theorem max_abs_diff_f_g :
  ∃ (c : ℝ), c ∈ Set.Icc 0 1 ∧ 
  (∀ x, x ∈ Set.Icc 0 1 → absDiff x ≤ absDiff c) ∧
  absDiff c = 4/27 :=
sorry

end NUMINAMATH_CALUDE_max_abs_diff_f_g_l2987_298722


namespace NUMINAMATH_CALUDE_range_of_m_given_p_q_l2987_298789

/-- The range of m given the conditions of p and q -/
theorem range_of_m_given_p_q :
  ∀ (m : ℝ),
  (∀ x : ℝ, x^2 - 8*x - 20 > 0 → (x - (1 - m)) * (x - (1 + m)) > 0) ∧
  (∃ x : ℝ, (x - (1 - m)) * (x - (1 + m)) > 0 ∧ x^2 - 8*x - 20 ≤ 0) ∧
  m > 0 →
  0 < m ∧ m ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_given_p_q_l2987_298789


namespace NUMINAMATH_CALUDE_first_group_weavers_l2987_298701

/-- The number of weavers in the first group -/
def num_weavers : ℕ := 4

/-- The number of mats woven by the first group -/
def mats_first_group : ℕ := 4

/-- The number of days taken by the first group -/
def days_first_group : ℕ := 4

/-- The number of weavers in the second group -/
def weavers_second_group : ℕ := 10

/-- The number of mats woven by the second group -/
def mats_second_group : ℕ := 25

/-- The number of days taken by the second group -/
def days_second_group : ℕ := 10

/-- The rate of weaving is constant across both groups -/
axiom constant_rate : (mats_first_group : ℚ) / (num_weavers * days_first_group) = 
                      (mats_second_group : ℚ) / (weavers_second_group * days_second_group)

theorem first_group_weavers : num_weavers = 4 := by
  sorry

end NUMINAMATH_CALUDE_first_group_weavers_l2987_298701


namespace NUMINAMATH_CALUDE_smallest_satisfying_number_l2987_298756

def reverse_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.foldl (λ acc d => acc * 10 + d) 0

def all_odd_digits (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, d % 2 = 1

def satisfies_condition (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ all_odd_digits (n + reverse_digits n)

theorem smallest_satisfying_number :
  satisfies_condition 209 ∧
  ∀ m : ℕ, satisfies_condition m → m ≥ 209 :=
sorry

end NUMINAMATH_CALUDE_smallest_satisfying_number_l2987_298756


namespace NUMINAMATH_CALUDE_stacy_berries_l2987_298774

/-- The number of berries each person has -/
structure BerryDistribution where
  sophie : ℕ
  sylar : ℕ
  steve : ℕ
  stacy : ℕ

/-- The conditions of the berry distribution problem -/
def valid_distribution (b : BerryDistribution) : Prop :=
  b.sylar = 5 * b.sophie ∧
  b.steve = 2 * b.sylar ∧
  b.stacy = 4 * b.steve ∧
  b.sophie + b.sylar + b.steve + b.stacy = 2200

/-- Theorem stating that Stacy has 1560 berries -/
theorem stacy_berries (b : BerryDistribution) (h : valid_distribution b) : b.stacy = 1560 := by
  sorry

end NUMINAMATH_CALUDE_stacy_berries_l2987_298774


namespace NUMINAMATH_CALUDE_arctan_sum_of_cubic_roots_l2987_298785

theorem arctan_sum_of_cubic_roots (x₁ x₂ x₃ : ℝ) : 
  x₁^3 - 10*x₁ + 11 = 0 →
  x₂^3 - 10*x₂ + 11 = 0 →
  x₃^3 - 10*x₃ + 11 = 0 →
  -5 < x₁ ∧ x₁ < 5 →
  -5 < x₂ ∧ x₂ < 5 →
  -5 < x₃ ∧ x₃ < 5 →
  Real.arctan x₁ + Real.arctan x₂ + Real.arctan x₃ = π/4 := by
sorry

end NUMINAMATH_CALUDE_arctan_sum_of_cubic_roots_l2987_298785


namespace NUMINAMATH_CALUDE_lifting_ratio_after_training_l2987_298797

/-- Calculates the ratio of lifting total to bodyweight after training -/
theorem lifting_ratio_after_training 
  (initial_total : ℝ)
  (initial_weight : ℝ)
  (total_increase_percent : ℝ)
  (weight_increase : ℝ)
  (h1 : initial_total = 2200)
  (h2 : initial_weight = 245)
  (h3 : total_increase_percent = 0.15)
  (h4 : weight_increase = 8) :
  (initial_total * (1 + total_increase_percent)) / (initial_weight + weight_increase) = 10 :=
by
  sorry

#check lifting_ratio_after_training

end NUMINAMATH_CALUDE_lifting_ratio_after_training_l2987_298797


namespace NUMINAMATH_CALUDE_complete_square_quadratic_l2987_298792

theorem complete_square_quadratic :
  ∀ x : ℝ, x^2 - 4*x - 6 = 0 ↔ (x - 2)^2 = 10 :=
by sorry

end NUMINAMATH_CALUDE_complete_square_quadratic_l2987_298792


namespace NUMINAMATH_CALUDE_smallest_multiples_of_17_l2987_298776

theorem smallest_multiples_of_17 :
  (∃ n : ℕ, n * 17 = 34 ∧ ∀ m : ℕ, m * 17 ≥ 10 ∧ m * 17 < 100 → m * 17 ≥ 34) ∧
  (∃ n : ℕ, n * 17 = 1003 ∧ ∀ m : ℕ, m * 17 ≥ 1000 ∧ m * 17 < 10000 → m * 17 ≥ 1003) :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiples_of_17_l2987_298776


namespace NUMINAMATH_CALUDE_thirteenth_divisible_by_three_l2987_298712

theorem thirteenth_divisible_by_three (start : ℕ) (count : ℕ) : 
  start > 10 → 
  start % 3 = 0 → 
  ∀ n < start, n > 10 → n % 3 ≠ 0 →
  count = 13 →
  (start + 3 * (count - 1) = 48) :=
sorry

end NUMINAMATH_CALUDE_thirteenth_divisible_by_three_l2987_298712


namespace NUMINAMATH_CALUDE_parallelogram_area_example_l2987_298750

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 12 m and height 6 m is 72 sq m -/
theorem parallelogram_area_example : parallelogram_area 12 6 = 72 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_example_l2987_298750


namespace NUMINAMATH_CALUDE_andrews_age_is_five_l2987_298744

/-- Andrew's age in years -/
def andrews_age : ℕ := 5

/-- Andrew's grandfather's age in years -/
def grandfathers_age : ℕ := andrews_age * 10

/-- Age difference between Andrew's grandfather and Andrew -/
def age_difference : ℕ := 45

theorem andrews_age_is_five :
  andrews_age = 5 ∧
  grandfathers_age = andrews_age * 10 ∧
  grandfathers_age - andrews_age = age_difference :=
by sorry

end NUMINAMATH_CALUDE_andrews_age_is_five_l2987_298744


namespace NUMINAMATH_CALUDE_ratio_a_to_c_l2987_298788

theorem ratio_a_to_c (a b c d : ℚ) 
  (hab : a / b = 5 / 4)
  (hcd : c / d = 4 / 1)
  (hdb : d / b = 1 / 5) :
  a / c = 25 / 16 := by
  sorry

end NUMINAMATH_CALUDE_ratio_a_to_c_l2987_298788


namespace NUMINAMATH_CALUDE_max_area_rectangular_pen_l2987_298717

/-- Given a rectangular pen with a perimeter of 60 feet, 
    the maximum possible area is 225 square feet. -/
theorem max_area_rectangular_pen : 
  ∀ x y : ℝ, 
    x > 0 → y > 0 → 
    2 * x + 2 * y = 60 → 
    x * y ≤ 225 := by
  sorry

end NUMINAMATH_CALUDE_max_area_rectangular_pen_l2987_298717


namespace NUMINAMATH_CALUDE_negation_of_proposition_l2987_298706

theorem negation_of_proposition (p : Prop) :
  (¬p ↔ ∃ x > 0, Real.exp x < x + 1) ↔ (p ↔ ∀ x > 0, Real.exp x ≥ x + 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l2987_298706


namespace NUMINAMATH_CALUDE_ab_greater_than_a_plus_b_l2987_298721

theorem ab_greater_than_a_plus_b (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a - b = a / b) : a * b > a + b := by
  sorry

end NUMINAMATH_CALUDE_ab_greater_than_a_plus_b_l2987_298721


namespace NUMINAMATH_CALUDE_protest_duration_increase_l2987_298779

/-- Given two protests with a total duration of 9 days, where the first protest lasts 4 days,
    the percentage increase in duration from the first to the second protest is 25%. -/
theorem protest_duration_increase (d₁ d₂ : ℝ) : 
  d₁ = 4 → d₁ + d₂ = 9 → (d₂ - d₁) / d₁ * 100 = 25 := by
  sorry

#check protest_duration_increase

end NUMINAMATH_CALUDE_protest_duration_increase_l2987_298779


namespace NUMINAMATH_CALUDE_pencils_per_row_l2987_298769

/-- Given a total of 720 pencils arranged in 30 rows, prove that there are 24 pencils in each row. -/
theorem pencils_per_row (total_pencils : ℕ) (total_rows : ℕ) (pencils_per_row : ℕ) 
  (h1 : total_pencils = 720) 
  (h2 : total_rows = 30) 
  (h3 : total_pencils = total_rows * pencils_per_row) : 
  pencils_per_row = 24 := by
  sorry

end NUMINAMATH_CALUDE_pencils_per_row_l2987_298769


namespace NUMINAMATH_CALUDE_exactly_one_positive_integer_solution_l2987_298796

theorem exactly_one_positive_integer_solution : 
  ∃! (n : ℕ), n > 0 ∧ 25 - 5 * n > 15 :=
sorry

end NUMINAMATH_CALUDE_exactly_one_positive_integer_solution_l2987_298796


namespace NUMINAMATH_CALUDE_school_travel_time_l2987_298755

/-- 
If a boy reaches school 4 minutes earlier when walking at 9/8 of his usual rate,
then his usual time to reach the school is 36 minutes.
-/
theorem school_travel_time (usual_rate : ℝ) (usual_time : ℝ) 
  (h : usual_rate * usual_time = (9/8 * usual_rate) * (usual_time - 4)) :
  usual_time = 36 := by
  sorry

end NUMINAMATH_CALUDE_school_travel_time_l2987_298755


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l2987_298771

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l2987_298771


namespace NUMINAMATH_CALUDE_pool_capacity_l2987_298753

/-- The capacity of a pool given three valves with specific flow rates. -/
theorem pool_capacity (v1 : ℝ) (r : ℝ) : 
  (v1 * 120 = r) →  -- First valve fills the pool in 2 hours
  ((v1 + (v1 + 50) + (v1 - 25)) * 36 = r) →  -- All valves fill the pool in 36 minutes
  (r = 9000) :=
by sorry

end NUMINAMATH_CALUDE_pool_capacity_l2987_298753


namespace NUMINAMATH_CALUDE_smallest_sum_is_28_l2987_298775

/-- Converts a number from base 6 to base 10 --/
def base6To10 (x y z : Nat) : Nat :=
  36 * x + 6 * y + z

/-- Converts a number from base b to base 10 --/
def baseBTo10 (b : Nat) : Nat :=
  3 * b + 3

/-- Represents the conditions of the problem --/
def validConfiguration (x y z b : Nat) : Prop :=
  x ≤ 5 ∧ y ≤ 5 ∧ z ≤ 5 ∧ b > 6 ∧ base6To10 x y z = baseBTo10 b

theorem smallest_sum_is_28 :
  ∃ x y z b, validConfiguration x y z b ∧
  ∀ x' y' z' b', validConfiguration x' y' z' b' →
    x + y + z + b ≤ x' + y' + z' + b' ∧
    x + y + z + b = 28 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_is_28_l2987_298775


namespace NUMINAMATH_CALUDE_rohans_age_l2987_298733

theorem rohans_age :
  ∀ R : ℕ, (R + 15 = 4 * (R - 15)) → R = 25 := by
  sorry

end NUMINAMATH_CALUDE_rohans_age_l2987_298733


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2987_298766

theorem sufficient_not_necessary_condition (x : ℝ) :
  (∀ x, x > 1 → x^2 + 2*x > 0) ∧
  (∃ x, x^2 + 2*x > 0 ∧ ¬(x > 1)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2987_298766


namespace NUMINAMATH_CALUDE_remainder_problem_l2987_298711

theorem remainder_problem (N : ℤ) (D : ℤ) (h1 : D = 398) (h2 : ∃ Q', 2*N = D*Q' + 112) :
  ∃ Q, N = D*Q + 56 :=
sorry

end NUMINAMATH_CALUDE_remainder_problem_l2987_298711


namespace NUMINAMATH_CALUDE_abs_greater_than_two_solution_set_l2987_298716

theorem abs_greater_than_two_solution_set :
  {x : ℝ | |x| > 2} = {x : ℝ | x > 2 ∨ x < -2} := by sorry

end NUMINAMATH_CALUDE_abs_greater_than_two_solution_set_l2987_298716


namespace NUMINAMATH_CALUDE_joe_cars_count_l2987_298783

theorem joe_cars_count (initial_cars additional_cars : ℕ) : 
  initial_cars = 50 → additional_cars = 12 → initial_cars + additional_cars = 62 := by
  sorry

end NUMINAMATH_CALUDE_joe_cars_count_l2987_298783


namespace NUMINAMATH_CALUDE_smallest_possible_median_l2987_298745

def number_set (y : ℤ) : Finset ℤ := {y, 3*y, 4, 1, 7}

def is_median (m : ℤ) (s : Finset ℤ) : Prop :=
  2 * (s.filter (· ≤ m)).card ≥ s.card ∧
  2 * (s.filter (· ≥ m)).card ≥ s.card

theorem smallest_possible_median :
  ∃ y : ℤ, is_median 1 (number_set y) ∧
  ∀ m : ℤ, ∀ z : ℤ, is_median m (number_set z) → m ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_smallest_possible_median_l2987_298745


namespace NUMINAMATH_CALUDE_limit_one_minus_cos_over_exp_squared_l2987_298786

theorem limit_one_minus_cos_over_exp_squared :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 
    0 < |x| ∧ |x| < δ → 
    |((1 - Real.cos x) / (Real.exp (3 * x) - 1)^2) - (1/18)| < ε := by
  sorry

end NUMINAMATH_CALUDE_limit_one_minus_cos_over_exp_squared_l2987_298786


namespace NUMINAMATH_CALUDE_polygon_sides_l2987_298757

theorem polygon_sides (n : ℕ) : 
  (n ≥ 3) →                           -- n is at least 3 (for a valid polygon)
  ((n - 2) * 180 = 3 * 360) →         -- sum of interior angles = 3 * sum of exterior angles
  n = 8                               -- the polygon has 8 sides
:= by sorry

end NUMINAMATH_CALUDE_polygon_sides_l2987_298757


namespace NUMINAMATH_CALUDE_wade_drink_cost_l2987_298799

/-- The cost of each drink given Wade's purchases -/
theorem wade_drink_cost (total_spent : ℝ) (sandwich_cost : ℝ) (num_sandwiches : ℕ) (num_drinks : ℕ) 
  (h1 : total_spent = 26)
  (h2 : sandwich_cost = 6)
  (h3 : num_sandwiches = 3)
  (h4 : num_drinks = 2) :
  (total_spent - num_sandwiches * sandwich_cost) / num_drinks = 4 := by
  sorry

end NUMINAMATH_CALUDE_wade_drink_cost_l2987_298799


namespace NUMINAMATH_CALUDE_quarters_undetermined_l2987_298748

/-- Represents Mike's coin collection --/
structure CoinCollection where
  quarters : ℕ
  nickels : ℕ

/-- Represents the borrowing transaction --/
def borrow (c : CoinCollection) (borrowed : ℕ) : CoinCollection :=
  { quarters := c.quarters, nickels := c.nickels - borrowed }

theorem quarters_undetermined (initial_nickels borrowed remaining_nickels : ℕ) 
  (h1 : initial_nickels = 87)
  (h2 : borrowed = 75)
  (h3 : remaining_nickels = 12)
  (h4 : initial_nickels = borrowed + remaining_nickels) :
  ∀ q1 q2 : ℕ, ∃ c1 c2 : CoinCollection,
    c1.nickels = initial_nickels ∧
    c2.nickels = initial_nickels ∧
    c1.quarters = q1 ∧
    c2.quarters = q2 ∧
    (borrow c1 borrowed).nickels = remaining_nickels ∧
    (borrow c2 borrowed).nickels = remaining_nickels :=
sorry

end NUMINAMATH_CALUDE_quarters_undetermined_l2987_298748


namespace NUMINAMATH_CALUDE_discretionary_income_ratio_l2987_298784

/-- Jill's financial situation --/
def jill_finances (net_salary : ℚ) (discretionary_income : ℚ) : Prop :=
  net_salary = 3600 ∧
  0.30 * discretionary_income + 0.20 * discretionary_income + 0.35 * discretionary_income + 108 = discretionary_income ∧
  discretionary_income > 0

/-- The ratio of discretionary income to net salary is 1:5 --/
theorem discretionary_income_ratio
  (net_salary discretionary_income : ℚ)
  (h : jill_finances net_salary discretionary_income) :
  discretionary_income / net_salary = 1 / 5 :=
by sorry

end NUMINAMATH_CALUDE_discretionary_income_ratio_l2987_298784


namespace NUMINAMATH_CALUDE_cookies_eaten_l2987_298773

theorem cookies_eaten (initial_cookies bought_cookies final_cookies : ℕ) :
  initial_cookies = 40 →
  bought_cookies = 37 →
  final_cookies = 75 →
  initial_cookies + bought_cookies - final_cookies = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_cookies_eaten_l2987_298773


namespace NUMINAMATH_CALUDE_female_managers_count_l2987_298700

theorem female_managers_count (total_employees : ℕ) (female_employees : ℕ) (managers : ℕ) (male_managers : ℕ) : 
  female_employees = 700 →
  managers = (2 : ℕ) * total_employees / (5 : ℕ) →
  male_managers = (2 : ℕ) * (total_employees - female_employees) / (5 : ℕ) →
  (2 : ℕ) * female_employees / (5 : ℕ) = 280 :=
by
  sorry


end NUMINAMATH_CALUDE_female_managers_count_l2987_298700


namespace NUMINAMATH_CALUDE_probability_red_or_white_l2987_298791

/-- Probability of selecting a red or white marble from a bag -/
theorem probability_red_or_white (total : ℕ) (blue : ℕ) (red : ℕ) :
  total = 20 →
  blue = 5 →
  red = 9 →
  (red + (total - blue - red)) / total = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_probability_red_or_white_l2987_298791


namespace NUMINAMATH_CALUDE_number_plus_expression_l2987_298782

theorem number_plus_expression (x : ℝ) : x + 2 * (8 - 3) = 15 → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_number_plus_expression_l2987_298782
