import Mathlib

namespace NUMINAMATH_CALUDE_plate_on_square_table_l3577_357752

/-- Given a square table with a round plate, if the distances from the edge of the plate
    to three edges of the table are 10 cm, 63 cm, and 20 cm respectively, then the distance
    from the edge of the plate to the fourth edge of the table is 53 cm. -/
theorem plate_on_square_table (d1 d2 d3 d4 : ℝ) (h1 : d1 = 10) (h2 : d2 = 63) (h3 : d3 = 20)
    (h_square : d1 + d2 = d3 + d4) : d4 = 53 := by
  sorry

end NUMINAMATH_CALUDE_plate_on_square_table_l3577_357752


namespace NUMINAMATH_CALUDE_max_M_value_l3577_357755

/-- J is a function that takes a natural number m and returns 10^5 + m -/
def J (m : ℕ) : ℕ := 10^5 + m

/-- M is a function that takes a natural number a and returns the number of factors of 2
    in the prime factorization of J(2^a) -/
def M (a : ℕ) : ℕ := (J (2^a)).factors.count 2

/-- The maximum value of M(a) for a ≥ 0 is 5 -/
theorem max_M_value : ∃ (k : ℕ), k = 5 ∧ ∀ (a : ℕ), M a ≤ k :=
sorry

end NUMINAMATH_CALUDE_max_M_value_l3577_357755


namespace NUMINAMATH_CALUDE_revenue_comparison_l3577_357738

/-- Given a projected revenue increase of 40% and an actual revenue decrease of 30% from the previous year,
    the actual revenue is 50% of the projected revenue. -/
theorem revenue_comparison (previous_revenue : ℝ) (projected_increase : ℝ) (actual_decrease : ℝ)
    (h1 : projected_increase = 0.4)
    (h2 : actual_decrease = 0.3) :
    (previous_revenue * (1 - actual_decrease)) / (previous_revenue * (1 + projected_increase)) = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_revenue_comparison_l3577_357738


namespace NUMINAMATH_CALUDE_marias_profit_is_75_l3577_357787

/-- Calculates Maria's profit from bread sales given the specified conditions. -/
def marias_profit (total_loaves : ℕ) (cost_per_loaf : ℚ) 
  (morning_price : ℚ) (afternoon_price_ratio : ℚ) (evening_price : ℚ) : ℚ :=
  let morning_sales := total_loaves / 3 * morning_price
  let afternoon_loaves := total_loaves - total_loaves / 3
  let afternoon_sales := afternoon_loaves / 2 * (afternoon_price_ratio * morning_price)
  let evening_loaves := afternoon_loaves - afternoon_loaves / 2
  let evening_sales := evening_loaves * evening_price
  let total_revenue := morning_sales + afternoon_sales + evening_sales
  let total_cost := total_loaves * cost_per_loaf
  total_revenue - total_cost

/-- Theorem stating that Maria's profit is $75 given the specified conditions. -/
theorem marias_profit_is_75 : 
  marias_profit 60 1 3 (3/4) (3/2) = 75 := by
  sorry

end NUMINAMATH_CALUDE_marias_profit_is_75_l3577_357787


namespace NUMINAMATH_CALUDE_intersection_S_T_equals_T_l3577_357743

-- Define set S
def S : Set Int := {s | ∃ n : Int, s = 2 * n + 1}

-- Define set T
def T : Set Int := {t | ∃ n : Int, t = 4 * n + 1}

-- Theorem statement
theorem intersection_S_T_equals_T : S ∩ T = T := by sorry

end NUMINAMATH_CALUDE_intersection_S_T_equals_T_l3577_357743


namespace NUMINAMATH_CALUDE_james_future_age_l3577_357735

def justin_age : ℕ := 26
def jessica_age_at_justin_birth : ℕ := 6
def james_age_diff_jessica : ℕ := 7
def years_in_future : ℕ := 5

theorem james_future_age :
  justin_age + jessica_age_at_justin_birth + james_age_diff_jessica + years_in_future = 44 :=
by sorry

end NUMINAMATH_CALUDE_james_future_age_l3577_357735


namespace NUMINAMATH_CALUDE_hyperbola_m_value_l3577_357763

/-- A hyperbola with equation mx^2 + y^2 = 1 -/
structure Hyperbola (m : ℝ) where
  equation : ∀ x y : ℝ, m * x^2 + y^2 = 1

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola m) : ℝ := sorry

/-- The slope of an asymptote of a hyperbola -/
def asymptote_slope (h : Hyperbola m) : ℝ := sorry

theorem hyperbola_m_value (m : ℝ) (h : Hyperbola m) :
  (∃ k : ℝ, k > 0 ∧ eccentricity h = 2 * k ∧ asymptote_slope h = k) →
  m = -3 := by sorry

end NUMINAMATH_CALUDE_hyperbola_m_value_l3577_357763


namespace NUMINAMATH_CALUDE_artwork_area_l3577_357793

/-- Given a rectangular artwork with a frame, calculate its area -/
theorem artwork_area (outer_width outer_height frame_width_top frame_width_side : ℕ) 
  (h1 : outer_width = 100)
  (h2 : outer_height = 140)
  (h3 : frame_width_top = 15)
  (h4 : frame_width_side = 20) :
  (outer_width - 2 * frame_width_side) * (outer_height - 2 * frame_width_top) = 6600 := by
  sorry

#check artwork_area

end NUMINAMATH_CALUDE_artwork_area_l3577_357793


namespace NUMINAMATH_CALUDE_intersection_circle_line_l3577_357712

theorem intersection_circle_line :
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = 1}
  let line := {(x, y) : ℝ × ℝ | y = x + 1}
  let intersection := circle ∩ line
  intersection = {(-1, 0), (0, 1)} := by
  sorry

end NUMINAMATH_CALUDE_intersection_circle_line_l3577_357712


namespace NUMINAMATH_CALUDE_zero_after_double_one_l3577_357766

/-- Represents a binary sequence -/
def BinarySequence := List Bool

/-- Counts the occurrences of a given subsequence in a binary sequence -/
def count_subsequence (seq : BinarySequence) (subseq : BinarySequence) : Nat :=
  sorry

/-- The main theorem -/
theorem zero_after_double_one (seq : BinarySequence) : 
  (count_subsequence seq [false, true] = 16) →
  (count_subsequence seq [true, false] = 15) →
  (count_subsequence seq [false, true, false] = 8) →
  (count_subsequence seq [true, true, false] = 7) :=
sorry

end NUMINAMATH_CALUDE_zero_after_double_one_l3577_357766


namespace NUMINAMATH_CALUDE_dad_borrowed_75_nickels_l3577_357791

/-- The number of nickels borrowed by Mike's dad -/
def nickels_borrowed (initial_nickels current_nickels : ℕ) : ℕ :=
  initial_nickels - current_nickels

/-- Proof that Mike's dad borrowed 75 nickels -/
theorem dad_borrowed_75_nickels (initial_nickels current_nickels : ℕ) 
  (h1 : initial_nickels = 87)
  (h2 : current_nickels = 12) :
  nickels_borrowed initial_nickels current_nickels = 75 := by
  sorry

#eval nickels_borrowed 87 12

end NUMINAMATH_CALUDE_dad_borrowed_75_nickels_l3577_357791


namespace NUMINAMATH_CALUDE_age_sum_theorem_l3577_357797

theorem age_sum_theorem (f d : ℕ) (h1 : f * d = 238) (h2 : (f + 4) * (d + 4) = 378) :
  f + d = 31 := by
  sorry

end NUMINAMATH_CALUDE_age_sum_theorem_l3577_357797


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l3577_357706

theorem quadratic_roots_property (p q : ℝ) : 
  (3 * p^2 - 5 * p - 7 = 0) → 
  (3 * q^2 - 5 * q - 7 = 0) → 
  p ≠ q →
  (3 * p^2 - 3 * q^2) * (p - q)⁻¹ = 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l3577_357706


namespace NUMINAMATH_CALUDE_gear_speed_ratio_l3577_357759

structure Gear where
  teeth : ℕ
  speed : ℚ

def meshed (g1 g2 : Gear) : Prop :=
  g1.teeth * g1.speed = g2.teeth * g2.speed

theorem gear_speed_ratio 
  (A B C D : Gear)
  (h_mesh_AB : meshed A B)
  (h_mesh_BC : meshed B C)
  (h_mesh_CD : meshed C D)
  (h_prime_p : Nat.Prime A.teeth)
  (h_prime_q : Nat.Prime B.teeth)
  (h_prime_r : Nat.Prime C.teeth)
  (h_prime_s : Nat.Prime D.teeth)
  (h_distinct : A.teeth ≠ B.teeth ∧ A.teeth ≠ C.teeth ∧ A.teeth ≠ D.teeth ∧
                B.teeth ≠ C.teeth ∧ B.teeth ≠ D.teeth ∧ C.teeth ≠ D.teeth)
  (h_speed_ratio : A.speed / A.teeth = B.speed / B.teeth ∧
                   B.speed / B.teeth = C.speed / C.teeth ∧
                   C.speed / C.teeth = D.speed / D.teeth) :
  ∃ (k : ℚ), A.speed = k * D.teeth * C.teeth ∧
             B.speed = k * D.teeth * A.teeth ∧
             C.speed = k * D.teeth * C.teeth ∧
             D.speed = k * C.teeth * A.teeth := by
  sorry

end NUMINAMATH_CALUDE_gear_speed_ratio_l3577_357759


namespace NUMINAMATH_CALUDE_smallest_valid_number_l3577_357770

def is_valid (n : ℕ) : Prop :=
  11 ∣ n ∧ ∀ k : ℕ, 2 ≤ k → k ≤ 8 → n % k = 3

theorem smallest_valid_number : 
  is_valid 5043 ∧ ∀ m : ℕ, m < 5043 → ¬(is_valid m) :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_number_l3577_357770


namespace NUMINAMATH_CALUDE_elizabeth_salon_cost_l3577_357782

/-- Represents a salon visit with hair cut length and treatment cost -/
structure SalonVisit where
  cutLength : Float
  treatmentCost : Float
  discountPercentage : Float

/-- Calculate the discounted cost for a salon visit -/
def discountedCost (visit : SalonVisit) : Float :=
  visit.treatmentCost * (1 - visit.discountPercentage)

/-- Calculate the total cost of salon visits after discounts -/
def totalCost (visits : List SalonVisit) : Float :=
  visits.map discountedCost |>.sum

/-- Theorem: The total cost of Elizabeth's salon visits is $88.25 -/
theorem elizabeth_salon_cost :
  let visits : List SalonVisit := [
    { cutLength := 0.375, treatmentCost := 25, discountPercentage := 0.1 },
    { cutLength := 0.5, treatmentCost := 35, discountPercentage := 0.15 },
    { cutLength := 0.75, treatmentCost := 45, discountPercentage := 0.2 }
  ]
  totalCost visits = 88.25 := by
  sorry

end NUMINAMATH_CALUDE_elizabeth_salon_cost_l3577_357782


namespace NUMINAMATH_CALUDE_function_is_constant_l3577_357702

/-- A function satisfying the given conditions is constant and non-zero -/
theorem function_is_constant (f : ℝ → ℝ) 
  (h1 : ∀ x y, f x + f y ≠ 0)
  (h2 : ∀ x y, (f x - f (x - y)) / (f x + f (x + y)) + (f x - f (x + y)) / (f x + f (x - y)) = 0) :
  ∃ c : ℝ, c ≠ 0 ∧ ∀ x, f x = c :=
sorry

end NUMINAMATH_CALUDE_function_is_constant_l3577_357702


namespace NUMINAMATH_CALUDE_principal_amount_proof_l3577_357715

/-- Proves that the principal amount is 800 given the specified conditions -/
theorem principal_amount_proof (rate : ℝ) (time : ℝ) (total_amount : ℝ) : 
  rate = 0.0375 → time = 5 → total_amount = 950 →
  (∃ (principal : ℝ), principal * (1 + rate * time) = total_amount ∧ principal = 800) := by
  sorry

#check principal_amount_proof

end NUMINAMATH_CALUDE_principal_amount_proof_l3577_357715


namespace NUMINAMATH_CALUDE_derivative_at_a_l3577_357761

theorem derivative_at_a (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, DifferentiableAt ℝ f x) →
  (∀ ε > 0, ∃ δ > 0, ∀ Δx ≠ 0, |Δx| < δ → 
    |((f (a + 2*Δx) - f a) / (3*Δx)) - 1| < ε) →
  deriv f a = 3/2 := by sorry

end NUMINAMATH_CALUDE_derivative_at_a_l3577_357761


namespace NUMINAMATH_CALUDE_town_population_growth_l3577_357720

/-- Given an initial population and a final population after a certain number of years,
    calculate the average percent increase of population per year. -/
def average_percent_increase (initial_population final_population : ℕ) (years : ℕ) : ℚ :=
  ((final_population - initial_population : ℚ) / initial_population / years) * 100

/-- Theorem: The average percent increase of population per year for a town
    that grew from 175000 to 297500 in 10 years is 7%. -/
theorem town_population_growth : average_percent_increase 175000 297500 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_town_population_growth_l3577_357720


namespace NUMINAMATH_CALUDE_negative_representation_is_spending_l3577_357741

-- Define a type for monetary transactions
inductive MonetaryTransaction
| Receive (amount : ℤ)
| Spend (amount : ℤ)

-- Define a function to represent transactions as integers
def represent (t : MonetaryTransaction) : ℤ :=
  match t with
  | MonetaryTransaction.Receive amount => amount
  | MonetaryTransaction.Spend amount => -amount

-- State the theorem
theorem negative_representation_is_spending :
  (represent (MonetaryTransaction.Receive 100) = 100) →
  (represent (MonetaryTransaction.Spend 80) = -80) :=
by sorry

end NUMINAMATH_CALUDE_negative_representation_is_spending_l3577_357741


namespace NUMINAMATH_CALUDE_cuboid_height_calculation_l3577_357795

/-- The surface area of a cuboid given its length, breadth, and height. -/
def surfaceArea (l b h : ℝ) : ℝ := 2 * (l * b + l * h + b * h)

/-- Theorem: For a cuboid with surface area 720, length 12, and breadth 6, the height is 16. -/
theorem cuboid_height_calculation (SA l b h : ℝ) 
  (h_SA : SA = 720) 
  (h_l : l = 12) 
  (h_b : b = 6) 
  (h_surface_area : surfaceArea l b h = SA) : h = 16 := by
  sorry

#check cuboid_height_calculation

end NUMINAMATH_CALUDE_cuboid_height_calculation_l3577_357795


namespace NUMINAMATH_CALUDE_circle_and_tangent_lines_l3577_357740

-- Define the points
def A : ℝ × ℝ := (3, 0)
def B : ℝ × ℝ := (1, -2)
def N : ℝ × ℝ := (5, 3)

-- Define the line l: 2x + y - 4 = 0
def l (x y : ℝ) : Prop := 2 * x + y - 4 = 0

-- Define the circle M
def circle_M (x y : ℝ) : Prop := (x - 3)^2 + (y + 2)^2 = 4

-- Define the tangent lines
def tangent_line_1 (x : ℝ) : Prop := x = 5
def tangent_line_2 (x y : ℝ) : Prop := 21 * x - 20 * y - 45 = 0

theorem circle_and_tangent_lines :
  ∃ (center_x center_y : ℝ),
    -- The center of M lies on line l
    l center_x center_y ∧
    -- M passes through A and B
    circle_M A.1 A.2 ∧ circle_M B.1 B.2 ∧
    -- The tangent lines pass through N and are tangent to M
    (tangent_line_1 N.1 ∨ tangent_line_2 N.1 N.2) ∧
    (∀ x y, tangent_line_1 x ∨ tangent_line_2 x y → 
      ((x - center_x)^2 + (y - center_y)^2 = 4 → x = N.1 ∧ y = N.2)) :=
sorry

end NUMINAMATH_CALUDE_circle_and_tangent_lines_l3577_357740


namespace NUMINAMATH_CALUDE_walkers_meet_at_start_l3577_357703

/-- Represents a point on the rectangular loop -/
structure Point :=
  (position : ℕ)

/-- Represents a person walking on the loop -/
structure Walker :=
  (speed : ℕ)
  (direction : Bool) -- True for clockwise, False for counterclockwise

/-- The total number of blocks in the rectangular loop -/
def total_blocks : ℕ := 24

/-- Calculates the meeting point of two walkers -/
def meeting_point (w1 w2 : Walker) (start : Point) : Point :=
  sorry

/-- Theorem stating that the walkers meet at their starting point -/
theorem walkers_meet_at_start (start : Point) :
  let hector := Walker.mk 1 true
  let jane := Walker.mk 3 false
  (meeting_point hector jane start).position = start.position :=
sorry

end NUMINAMATH_CALUDE_walkers_meet_at_start_l3577_357703


namespace NUMINAMATH_CALUDE_sample_survey_most_appropriate_l3577_357725

/-- Represents a survey method --/
inductive SurveyMethod
  | InterestGroup
  | FamiliarFriends
  | AllStudents
  | SampleSurvey

/-- Criteria for evaluating survey methods --/
structure SurveyCriteria where
  representativeness : Bool
  practicality : Bool
  efficiency : Bool

/-- Evaluates a survey method based on given criteria --/
def evaluateSurveyMethod (method : SurveyMethod) : SurveyCriteria :=
  match method with
  | SurveyMethod.InterestGroup => { representativeness := false, practicality := true, efficiency := true }
  | SurveyMethod.FamiliarFriends => { representativeness := false, practicality := true, efficiency := true }
  | SurveyMethod.AllStudents => { representativeness := true, practicality := false, efficiency := false }
  | SurveyMethod.SampleSurvey => { representativeness := true, practicality := true, efficiency := true }

/-- Determines if a survey method is appropriate based on all criteria being met --/
def isAppropriateMethod (criteria : SurveyCriteria) : Bool :=
  criteria.representativeness ∧ criteria.practicality ∧ criteria.efficiency

/-- Theorem stating that the sample survey method is the most appropriate --/
theorem sample_survey_most_appropriate :
  ∀ (method : SurveyMethod),
    method = SurveyMethod.SampleSurvey ↔ isAppropriateMethod (evaluateSurveyMethod method) :=
  sorry


end NUMINAMATH_CALUDE_sample_survey_most_appropriate_l3577_357725


namespace NUMINAMATH_CALUDE_at_least_one_made_by_bellini_l3577_357790

/-- Represents the maker of a casket -/
inductive Maker
  | Bellini
  | SonOfBellini
  | Other

/-- Represents a casket -/
structure Casket where
  material : String
  inscription : String
  maker : Maker

/-- The statement on the gold casket -/
def gold_inscription (silver : Casket) : Prop :=
  silver.maker = Maker.SonOfBellini

/-- The statement on the silver casket -/
def silver_inscription (gold : Casket) : Prop :=
  gold.maker ≠ Maker.SonOfBellini

/-- The main theorem -/
theorem at_least_one_made_by_bellini 
  (gold : Casket) 
  (silver : Casket) 
  (h_gold_material : gold.material = "gold")
  (h_silver_material : silver.material = "silver")
  (h_gold_inscription : gold.inscription = "The silver casket was made by the son of Bellini")
  (h_silver_inscription : silver.inscription = "The gold casket was not made by the son of Bellini") :
  gold.maker = Maker.Bellini ∨ silver.maker = Maker.Bellini :=
by
  sorry


end NUMINAMATH_CALUDE_at_least_one_made_by_bellini_l3577_357790


namespace NUMINAMATH_CALUDE_fraction_value_l3577_357777

theorem fraction_value (x : ℝ) (h : 1 - 4/x + 4/(x^2) = 0) : 2/x = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l3577_357777


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_2023_l3577_357769

theorem reciprocal_of_negative_2023 :
  ((-2023)⁻¹ : ℚ) = -1 / 2023 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_2023_l3577_357769


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3577_357799

theorem quadratic_equation_solution : 
  ∃ (x₁ x₂ : ℝ), x₁ = -8 ∧ x₂ = -4 ∧
  ∀ x : ℝ, x^2 + 6*x + 8 = -2*(x + 4)*(x + 5) ↔ x = x₁ ∨ x = x₂ := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3577_357799


namespace NUMINAMATH_CALUDE_range_of_x_range_of_p_l3577_357707

-- Define the inequality function
def inequality (x p : ℝ) : Prop := x^2 + p*x + 1 > 2*x + p

-- Theorem 1
theorem range_of_x (p : ℝ) (h : |p| ≤ 2) :
  ∀ x, inequality x p → x < -1 ∨ x > 3 :=
sorry

-- Theorem 2
theorem range_of_p (x : ℝ) (h : 2 ≤ x ∧ x ≤ 4) :
  ∀ p, inequality x p → p > -1 :=
sorry

end NUMINAMATH_CALUDE_range_of_x_range_of_p_l3577_357707


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l3577_357742

theorem polynomial_divisibility (p q : ℚ) : 
  (∀ x : ℚ, (x + 3) * (x - 2) ∣ (x^5 - x^4 + x^3 - p*x^2 + q*x - 8)) → 
  p = -173/15 ∧ q = -466/15 := by
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l3577_357742


namespace NUMINAMATH_CALUDE_pats_pool_ratio_l3577_357728

theorem pats_pool_ratio : 
  let total_pools : ℕ := 800
  let ark_pools : ℕ := 200
  let supply_pools : ℕ := total_pools - ark_pools
  supply_pools / ark_pools = 3 := by
  sorry

end NUMINAMATH_CALUDE_pats_pool_ratio_l3577_357728


namespace NUMINAMATH_CALUDE_train_length_l3577_357785

/-- Given a train that crosses a 150-meter platform in 27 seconds and a signal pole in 18 seconds,
    prove that the length of the train is 300 meters. -/
theorem train_length (platform_length : ℝ) (platform_time : ℝ) (pole_time : ℝ) 
    (h1 : platform_length = 150)
    (h2 : platform_time = 27)
    (h3 : pole_time = 18) : 
  ∃ (train_length : ℝ), train_length = 300 := by
  sorry


end NUMINAMATH_CALUDE_train_length_l3577_357785


namespace NUMINAMATH_CALUDE_range_of_a_l3577_357775

def p (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (a - 1) * x < (a - 1) * y

def q (a : ℝ) : Prop := ∀ x : ℝ, -x^2 + 2*x - 2 ≤ a

theorem range_of_a :
  ∀ a : ℝ, (p a ∨ q a) ∧ ¬(p a ∧ q a) → a ∈ Set.Icc (-1 : ℝ) 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3577_357775


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_parameter_product_l3577_357732

/-- Given an ellipse and a hyperbola with specific foci, prove the product of their parameters. -/
theorem ellipse_hyperbola_parameter_product :
  ∀ (p q : ℝ),
  (∀ (x y : ℝ), x^2 / p^2 + y^2 / q^2 = 1 → (x = 0 ∧ y = 5) ∨ (x = 0 ∧ y = -5)) →
  (∀ (x y : ℝ), x^2 / p^2 - y^2 / q^2 = 1 → (x = 8 ∧ y = 0) ∨ (x = -8 ∧ y = 0)) →
  |p * q| = Real.sqrt 12371 / 2 := by
sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_parameter_product_l3577_357732


namespace NUMINAMATH_CALUDE_circle_increase_l3577_357747

/-- Theorem: When the radius of a circle is increased by 50%, 
    the circumference increases by 50% and the area increases by 125%. -/
theorem circle_increase (r : ℝ) (h : r > 0) : 
  let new_r := 1.5 * r
  let circ := 2 * Real.pi * r
  let new_circ := 2 * Real.pi * new_r
  let area := Real.pi * r^2
  let new_area := Real.pi * new_r^2
  (new_circ - circ) / circ = 0.5 ∧ (new_area - area) / area = 1.25 := by
  sorry

end NUMINAMATH_CALUDE_circle_increase_l3577_357747


namespace NUMINAMATH_CALUDE_expression_equals_four_l3577_357786

theorem expression_equals_four :
  2 * Real.cos (30 * π / 180) + (-1/2)⁻¹ + |Real.sqrt 3 - 2| + (2 * Real.sqrt (9/4))^0 + Real.sqrt 9 = 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_four_l3577_357786


namespace NUMINAMATH_CALUDE_functional_equation_solution_l3577_357711

/-- A function from nonzero reals to nonzero reals -/
def NonzeroRealFunction := ℝ* → ℝ*

/-- The property that f(x+y)(f(x) + f(y)) = f(x)f(y) for all nonzero real x and y -/
def SatisfiesProperty (f : NonzeroRealFunction) : Prop :=
  ∀ x y : ℝ*, f (x + y) * (f x + f y) = f x * f y

/-- The property that a function is increasing -/
def IsIncreasing (f : NonzeroRealFunction) : Prop :=
  ∀ x y : ℝ*, x < y → f x < f y

theorem functional_equation_solution :
  ∀ f : NonzeroRealFunction, IsIncreasing f → SatisfiesProperty f →
  ∃ a : ℝ, a > 0 ∧ ∀ x : ℝ*, f x = 1 / (a * x) :=
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l3577_357711


namespace NUMINAMATH_CALUDE_square_field_area_l3577_357716

/-- The area of a square field given the time and speed of a horse running around it -/
theorem square_field_area (time : ℝ) (speed : ℝ) : 
  time = 8 → speed = 12 → (time * speed / 4) ^ 2 = 576 := by
  sorry

end NUMINAMATH_CALUDE_square_field_area_l3577_357716


namespace NUMINAMATH_CALUDE_train_speed_conversion_l3577_357737

/-- Conversion factor from meters per second to kilometers per hour -/
def mps_to_kmph : ℝ := 3.6

/-- The train's speed in meters per second -/
def train_speed_mps : ℝ := 60.0048

/-- Theorem: Given a train's speed of 60.0048 meters per second, 
    its speed in kilometers per hour is equal to 216.01728 -/
theorem train_speed_conversion :
  train_speed_mps * mps_to_kmph = 216.01728 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_conversion_l3577_357737


namespace NUMINAMATH_CALUDE_min_distance_between_curves_l3577_357731

/-- The minimum distance between two points on given curves with the same y-coordinate -/
theorem min_distance_between_curves : ∃ (d : ℝ), d = (5 + Real.log 2) / 4 ∧
  ∀ (x₁ x₂ y : ℝ), 
    y = Real.exp (2 * x₁ + 1) → 
    y = Real.sqrt (2 * x₂ - 1) → 
    d ≤ |x₂ - x₁| := by
  sorry

end NUMINAMATH_CALUDE_min_distance_between_curves_l3577_357731


namespace NUMINAMATH_CALUDE_lanas_tulips_l3577_357784

/-- The number of tulips Lana picked -/
def tulips : ℕ := sorry

/-- The total number of flowers Lana picked -/
def total_flowers : ℕ := sorry

/-- The number of flowers Lana used -/
def used_flowers : ℕ := 70

/-- The number of roses Lana picked -/
def roses : ℕ := 37

/-- The extra flowers Lana picked -/
def extra_flowers : ℕ := 3

theorem lanas_tulips :
  (total_flowers = tulips + roses) →
  (total_flowers = used_flowers + extra_flowers) →
  tulips = 36 := by sorry

end NUMINAMATH_CALUDE_lanas_tulips_l3577_357784


namespace NUMINAMATH_CALUDE_abs_negative_thirteen_l3577_357724

theorem abs_negative_thirteen : |(-13 : ℤ)| = 13 := by
  sorry

end NUMINAMATH_CALUDE_abs_negative_thirteen_l3577_357724


namespace NUMINAMATH_CALUDE_platform_length_l3577_357773

/-- Calculates the length of a platform given train specifications -/
theorem platform_length (train_length : ℝ) (platform_crossing_time : ℝ) (pole_crossing_time : ℝ) :
  train_length = 300 →
  platform_crossing_time = 42 →
  pole_crossing_time = 18 →
  ∃ platform_length : ℝ, platform_length = 400 := by
  sorry

end NUMINAMATH_CALUDE_platform_length_l3577_357773


namespace NUMINAMATH_CALUDE_simplify_fraction_l3577_357794

theorem simplify_fraction : 
  1 / (1 / (1/3)^1 + 1 / (1/3)^2 + 1 / (1/3)^3) = 1 / 39 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3577_357794


namespace NUMINAMATH_CALUDE_white_squares_95th_figure_l3577_357730

/-- The number of white squares in the nth figure of the sequence -/
def white_squares (n : ℕ) : ℕ := 8 + 5 * (n - 1)

/-- Theorem: The 95th figure in the sequence has 478 white squares -/
theorem white_squares_95th_figure : white_squares 95 = 478 := by
  sorry

end NUMINAMATH_CALUDE_white_squares_95th_figure_l3577_357730


namespace NUMINAMATH_CALUDE_jill_red_packs_l3577_357783

/-- The number of bouncy balls in each package -/
def balls_per_pack : ℕ := 18

/-- The number of packs of yellow bouncy balls Jill bought -/
def yellow_packs : ℕ := 4

/-- The additional number of red bouncy balls compared to yellow bouncy balls -/
def additional_red_balls : ℕ := 18

/-- The number of packs of red bouncy balls Jill bought -/
def red_packs : ℕ := 5

theorem jill_red_packs : 
  red_packs * balls_per_pack = yellow_packs * balls_per_pack + additional_red_balls :=
by sorry

end NUMINAMATH_CALUDE_jill_red_packs_l3577_357783


namespace NUMINAMATH_CALUDE_campers_total_l3577_357774

/-- The total number of campers participating in all activities -/
def total_campers (morning_rowing : ℕ) (morning_hiking : ℕ) (morning_climbing : ℕ)
                  (afternoon_rowing : ℕ) (afternoon_hiking : ℕ) (afternoon_biking : ℕ) : ℕ :=
  morning_rowing + morning_hiking + morning_climbing +
  afternoon_rowing + afternoon_hiking + afternoon_biking

/-- Theorem stating that the total number of campers is 180 -/
theorem campers_total :
  total_campers 13 59 25 21 47 15 = 180 := by
  sorry

#eval total_campers 13 59 25 21 47 15

end NUMINAMATH_CALUDE_campers_total_l3577_357774


namespace NUMINAMATH_CALUDE_x_range_for_positive_f_l3577_357751

def f (a x : ℝ) : ℝ := x^2 + (a - 4) * x + 4 - 2 * a

theorem x_range_for_positive_f :
  (∀ a ∈ Set.Icc (-1) 1, ∀ x, f a x > 0) →
  {x : ℝ | x < 1 ∨ x > 3} = {x : ℝ | ∃ a ∈ Set.Icc (-1) 1, f a x > 0} :=
by sorry

end NUMINAMATH_CALUDE_x_range_for_positive_f_l3577_357751


namespace NUMINAMATH_CALUDE_sons_age_l3577_357727

/-- Given a woman and her son, where the woman's age is three years more than twice her son's age,
    and the sum of their ages is 84, prove that the son's age is 27. -/
theorem sons_age (S W : ℕ) (h1 : W = 2 * S + 3) (h2 : W + S = 84) : S = 27 := by
  sorry

end NUMINAMATH_CALUDE_sons_age_l3577_357727


namespace NUMINAMATH_CALUDE_impossibleTiling_l3577_357764

/-- Represents the types of pieces that can be used for tiling -/
inductive PieceType
  | A  -- 2x2 piece with one corner square of a different color
  | B  -- L-shaped piece covering 3 unit squares
  | C  -- 2x2 piece covering one square of each of four different colors

/-- Represents a board that can be tiled -/
structure Board where
  rows : Nat
  cols : Nat

/-- Represents a tiling of a board with a specific piece type -/
structure Tiling where
  board : Board
  pieceType : PieceType
  pieceCount : Nat

/-- Checks if a tiling is valid for a given board and piece type -/
def isValidTiling (t : Tiling) : Prop :=
  t.board.rows = 10 ∧ t.board.cols = 10 ∧ t.pieceCount = 25

/-- The main theorem stating that it's impossible to tile a 10x10 board with 25 pieces of any type -/
theorem impossibleTiling (t : Tiling) : isValidTiling t → False := by
  sorry


end NUMINAMATH_CALUDE_impossibleTiling_l3577_357764


namespace NUMINAMATH_CALUDE_solution_is_two_intersecting_lines_l3577_357719

/-- The set of points (x, y) satisfying the equation (x+3y)³ = x³ + 9y³ -/
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 + 3 * p.2)^3 = p.1^3 + 9 * p.2^3}

/-- A line in ℝ² defined by a*x + b*y + c = 0 -/
def Line (a b c : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | a * p.1 + b * p.2 + c = 0}

theorem solution_is_two_intersecting_lines :
  ∃ (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ),
    (a₁ ≠ 0 ∨ b₁ ≠ 0) ∧ 
    (a₂ ≠ 0 ∨ b₂ ≠ 0) ∧
    (a₁ * b₂ ≠ a₂ * b₁) ∧
    S = Line a₁ b₁ c₁ ∪ Line a₂ b₂ c₂ :=
  sorry

end NUMINAMATH_CALUDE_solution_is_two_intersecting_lines_l3577_357719


namespace NUMINAMATH_CALUDE_jasmine_solution_percentage_l3577_357723

theorem jasmine_solution_percentage
  (initial_volume : ℝ)
  (added_jasmine : ℝ)
  (added_water : ℝ)
  (final_percentage : ℝ)
  (h1 : initial_volume = 80)
  (h2 : added_jasmine = 8)
  (h3 : added_water = 12)
  (h4 : final_percentage = 16)
  : (initial_volume * (final_percentage / 100) - added_jasmine) / initial_volume * 100 = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_jasmine_solution_percentage_l3577_357723


namespace NUMINAMATH_CALUDE_roots_of_equation_l3577_357749

theorem roots_of_equation : 
  {x : ℝ | (18 / (x^2 - 9) - 3 / (x - 3) = 2)} = {3, -6} := by sorry

end NUMINAMATH_CALUDE_roots_of_equation_l3577_357749


namespace NUMINAMATH_CALUDE_mountain_hut_distance_l3577_357778

/-- The distance from the mountain hut to the station -/
def distance : ℝ := 15

/-- The time (in hours) from when the coach spoke until the train departs -/
def train_departure_time : ℝ := 3

theorem mountain_hut_distance :
  (distance / 4 = train_departure_time + 3/4) ∧
  (distance / 6 = train_departure_time - 1/2) →
  distance = 15 := by sorry

end NUMINAMATH_CALUDE_mountain_hut_distance_l3577_357778


namespace NUMINAMATH_CALUDE_inequality_solution_l3577_357721

theorem inequality_solution (x : ℝ) : 
  (6 * x^2 + 24 * x - 63) / ((3 * x - 4) * (x + 5)) < 4 ↔ 
  x < -5 ∨ x > 4/3 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3577_357721


namespace NUMINAMATH_CALUDE_oil_quantity_function_correct_l3577_357796

/-- Represents the remaining oil quantity in a tank after a given time of outflow. -/
def Q (t : ℝ) : ℝ := 40 - 0.2 * t

/-- The initial quantity of oil in the tank. -/
def initial_quantity : ℝ := 40

/-- The rate at which oil flows out of the tank. -/
def flow_rate : ℝ := 0.2

theorem oil_quantity_function_correct (t : ℝ) (h : t ≥ 0) :
  Q t = initial_quantity - flow_rate * t :=
by sorry

end NUMINAMATH_CALUDE_oil_quantity_function_correct_l3577_357796


namespace NUMINAMATH_CALUDE_slot_machine_game_l3577_357758

-- Define the slot machine game
def SlotMachine :=
  -- A slot machine that outputs positive integers k with probability 2^(-k)
  Unit

-- Define the winning condition for Ann
def AnnWins (n m : ℕ) : Prop :=
  -- Ann wins if she receives at least n tokens before Drew receives m tokens
  True

-- Define the winning condition for Drew
def DrewWins (n m : ℕ) : Prop :=
  -- Drew wins if he receives m tokens before Ann receives n tokens
  True

-- Define the equal probability of winning
def EqualProbability (n m : ℕ) : Prop :=
  -- The probability of Ann winning equals the probability of Drew winning
  True

-- Theorem statement
theorem slot_machine_game (m : ℕ) (h : m = 2^2018) :
  ∃ n : ℕ, EqualProbability n m ∧ n % 2018 = 2 :=
sorry

end NUMINAMATH_CALUDE_slot_machine_game_l3577_357758


namespace NUMINAMATH_CALUDE_remainder_theorem_l3577_357746

def polynomial (x : ℝ) : ℝ := 5*x^6 - 3*x^5 + 6*x^4 - 7*x^3 + 3*x^2 + 5*x - 14

def divisor (x : ℝ) : ℝ := 3*x - 6

theorem remainder_theorem :
  ∃ (q : ℝ → ℝ), ∀ x, polynomial x = (divisor x) * q x + 272 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3577_357746


namespace NUMINAMATH_CALUDE_gcf_of_180_240_300_l3577_357734

theorem gcf_of_180_240_300 : Nat.gcd 180 (Nat.gcd 240 300) = 60 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_180_240_300_l3577_357734


namespace NUMINAMATH_CALUDE_sin_squared_minus_two_sin_range_l3577_357789

theorem sin_squared_minus_two_sin_range (x : ℝ) : -1 ≤ Real.sin x ^ 2 - 2 * Real.sin x ∧ Real.sin x ^ 2 - 2 * Real.sin x ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_sin_squared_minus_two_sin_range_l3577_357789


namespace NUMINAMATH_CALUDE_largest_integer_is_110_l3577_357757

theorem largest_integer_is_110 (p q r s : ℤ) 
  (sum_pqr : p + q + r = 210)
  (sum_pqs : p + q + s = 230)
  (sum_prs : p + r + s = 250)
  (sum_qrs : q + r + s = 270) :
  max p (max q (max r s)) = 110 := by
  sorry

end NUMINAMATH_CALUDE_largest_integer_is_110_l3577_357757


namespace NUMINAMATH_CALUDE_expression_simplification_l3577_357736

theorem expression_simplification (a : ℝ) (h : a = Real.sqrt 2 + 1) :
  (1 - 1/a) / ((a^2 - 2*a + 1)/a) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3577_357736


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l3577_357779

theorem quadratic_no_real_roots : 
  ∀ (x : ℝ), x^2 - 2*x + 3 ≠ 0 := by sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l3577_357779


namespace NUMINAMATH_CALUDE_solve_linear_equation_l3577_357726

theorem solve_linear_equation : ∃ x : ℝ, 2 * x - 5 = 15 ∧ x = 10 := by sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l3577_357726


namespace NUMINAMATH_CALUDE_possible_values_of_k_l3577_357750

def A : Set ℝ := {-1, 1}

def B (k : ℝ) : Set ℝ := {x : ℝ | k * x = 1}

theorem possible_values_of_k :
  ∀ k : ℝ, B k ⊆ A ↔ k = -1 ∨ k = 0 ∨ k = 1 :=
by sorry

end NUMINAMATH_CALUDE_possible_values_of_k_l3577_357750


namespace NUMINAMATH_CALUDE_gcd_problem_l3577_357754

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 570 * k) :
  Int.gcd (4 * b^3 + 2 * b^2 + 5 * b + 171) b = 171 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l3577_357754


namespace NUMINAMATH_CALUDE_df_ab_ratio_l3577_357771

/-- Definition of the ellipse C -/
def C (x y : ℝ) : Prop := x^2 / 25 + y^2 / 9 = 1

/-- Definition of the right focus F -/
def F : ℝ × ℝ := (4, 0)

/-- Definition of line l passing through F -/
def l (k : ℝ) (x y : ℝ) : Prop := y - F.2 = k * (x - F.1)

/-- Definition of points A and B on the ellipse -/
def A (k : ℝ) : ℝ × ℝ := sorry
def B (k : ℝ) : ℝ × ℝ := sorry

/-- Definition of line l' (perpendicular bisector of AB) -/
def l' (k : ℝ) (x y : ℝ) : Prop :=
  y - (A k).2 / 2 - (B k).2 / 2 = -1/k * (x - (A k).1 / 2 - (B k).1 / 2)

/-- Definition of point D (intersection of l' and x-axis) -/
def D (k : ℝ) : ℝ × ℝ := sorry

/-- Theorem: The ratio DF/AB is equal to 2/5 -/
theorem df_ab_ratio (k : ℝ) :
  let df := Real.sqrt ((D k).1 - F.1)^2 + (D k).2^2
  let ab := Real.sqrt ((A k).1 - (B k).1)^2 + ((A k).2 - (B k).2)^2
  df / ab = 2/5 := by sorry

end NUMINAMATH_CALUDE_df_ab_ratio_l3577_357771


namespace NUMINAMATH_CALUDE_trig_identity_l3577_357772

theorem trig_identity : 4 * Real.cos (50 * π / 180) - Real.tan (40 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l3577_357772


namespace NUMINAMATH_CALUDE_upstream_time_calculation_l3577_357753

/-- Represents the swimming scenario -/
structure SwimmingScenario where
  downstream_time : ℝ  -- Time to swim downstream and across still lake
  upstream_time : ℝ    -- Time to swim upstream and across still lake
  all_downstream_time : ℝ  -- Time if entire journey was downstream

/-- The theorem stating the upstream time given the conditions -/
theorem upstream_time_calculation (s : SwimmingScenario) 
  (h1 : s.downstream_time = 1)
  (h2 : s.upstream_time = 2)
  (h3 : s.all_downstream_time = 5/6) :
  2 / ((2 / s.upstream_time) + (1 / s.downstream_time - 1 / s.all_downstream_time)) = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_upstream_time_calculation_l3577_357753


namespace NUMINAMATH_CALUDE_cosine_matrix_det_zero_l3577_357748

theorem cosine_matrix_det_zero : 
  let M : Matrix (Fin 3) (Fin 3) ℝ := ![
    ![Real.cos 1, Real.cos 2, Real.cos 3],
    ![Real.cos 4, Real.cos 5, Real.cos 6],
    ![Real.cos 7, Real.cos 8, Real.cos 9]
  ]
  Matrix.det M = 0 := by
sorry

end NUMINAMATH_CALUDE_cosine_matrix_det_zero_l3577_357748


namespace NUMINAMATH_CALUDE_custom_op_equation_solution_l3577_357756

-- Define the custom operation *
def custom_op (a b : ℚ) : ℚ := 4 * a - 2 * b

-- State the theorem
theorem custom_op_equation_solution :
  ∃ x : ℚ, custom_op 3 (custom_op 6 x) = -2 ∧ x = 17/2 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_equation_solution_l3577_357756


namespace NUMINAMATH_CALUDE_exists_abs_le_neg_l3577_357798

theorem exists_abs_le_neg : ∃ a : ℝ, |a| ≤ -a := by sorry

end NUMINAMATH_CALUDE_exists_abs_le_neg_l3577_357798


namespace NUMINAMATH_CALUDE_tshirt_production_rate_l3577_357718

theorem tshirt_production_rate (rate1 : ℝ) (total : ℕ) (rate2 : ℝ) : 
  rate1 = 12 → total = 15 → rate2 = (120 : ℝ) / ((total : ℝ) - 60 / rate1) → rate2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_tshirt_production_rate_l3577_357718


namespace NUMINAMATH_CALUDE_sequence_increasing_b_range_l3577_357767

theorem sequence_increasing_b_range (b : ℝ) :
  (∀ n : ℕ, n^2 + b*n < (n+1)^2 + b*(n+1)) →
  b > -3 :=
by sorry

end NUMINAMATH_CALUDE_sequence_increasing_b_range_l3577_357767


namespace NUMINAMATH_CALUDE_just_passed_count_l3577_357701

def total_students : ℕ := 1000

def first_division_percent : ℚ := 25 / 100
def second_division_percent : ℚ := 35 / 100
def third_division_percent : ℚ := 20 / 100
def fourth_division_percent : ℚ := 10 / 100
def failed_percent : ℚ := 4 / 100

theorem just_passed_count : 
  ∃ (just_passed : ℕ), 
    just_passed = total_students - 
      (first_division_percent * total_students).num - 
      (second_division_percent * total_students).num - 
      (third_division_percent * total_students).num - 
      (fourth_division_percent * total_students).num - 
      (failed_percent * total_students).num ∧ 
    just_passed = 60 := by
  sorry

end NUMINAMATH_CALUDE_just_passed_count_l3577_357701


namespace NUMINAMATH_CALUDE_equilateral_cone_lateral_surface_angle_l3577_357765

/-- Represents a cone with an equilateral triangle as its front view -/
structure EquilateralCone where
  side_length : ℝ
  generatrix_length : ℝ
  base_radius : ℝ
  lateral_surface_angle : ℝ

/-- The properties of an EquilateralCone -/
def is_valid_equilateral_cone (c : EquilateralCone) : Prop :=
  c.generatrix_length = c.side_length ∧
  c.base_radius = c.side_length / 2 ∧
  2 * Real.pi * c.base_radius = (c.lateral_surface_angle * Real.pi * c.generatrix_length) / 180

/-- Theorem: The lateral surface angle of an EquilateralCone is 180° -/
theorem equilateral_cone_lateral_surface_angle (c : EquilateralCone) 
  (h : is_valid_equilateral_cone c) : c.lateral_surface_angle = 180 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_cone_lateral_surface_angle_l3577_357765


namespace NUMINAMATH_CALUDE_ghee_mixture_theorem_l3577_357745

/-- Represents the composition of a ghee mixture -/
structure GheeMixture where
  total : ℝ
  pure_ghee : ℝ
  vanaspati : ℝ

/-- The original ghee mixture before addition -/
def original_mixture : GheeMixture :=
  { total := 30
  , pure_ghee := 15
  , vanaspati := 15 }

/-- The amount of pure ghee added to the mixture -/
def added_pure_ghee : ℝ := 20

/-- The final mixture after addition of pure ghee -/
def final_mixture : GheeMixture :=
  { total := original_mixture.total + added_pure_ghee
  , pure_ghee := original_mixture.pure_ghee + added_pure_ghee
  , vanaspati := original_mixture.vanaspati }

theorem ghee_mixture_theorem :
  (original_mixture.pure_ghee = original_mixture.vanaspati) ∧
  (original_mixture.pure_ghee + original_mixture.vanaspati = original_mixture.total) ∧
  (final_mixture.vanaspati / final_mixture.total = 0.3) →
  original_mixture.total = 30 := by
  sorry

end NUMINAMATH_CALUDE_ghee_mixture_theorem_l3577_357745


namespace NUMINAMATH_CALUDE_farm_area_ratio_l3577_357705

theorem farm_area_ratio :
  ∀ (s : ℝ),
  s > 0 →
  3 * s + 4 * s = 12 →
  (6 - s^2) / 6 = 145 / 147 :=
by
  sorry

end NUMINAMATH_CALUDE_farm_area_ratio_l3577_357705


namespace NUMINAMATH_CALUDE_vector_subtraction_and_scalar_multiplication_l3577_357788

theorem vector_subtraction_and_scalar_multiplication :
  let v₁ : Fin 2 → ℝ := ![3, -8]
  let v₂ : Fin 2 → ℝ := ![4, 6]
  let scalar : ℝ := -5
  let result : Fin 2 → ℝ := ![23, 22]
  v₁ - scalar • v₂ = result := by sorry

end NUMINAMATH_CALUDE_vector_subtraction_and_scalar_multiplication_l3577_357788


namespace NUMINAMATH_CALUDE_discount_is_twenty_percent_l3577_357780

/-- Represents the purchase of cucumbers and pencils with a discount --/
structure Purchase where
  cucumber_count : ℕ
  pencil_count : ℕ
  cucumber_price : ℕ
  initial_pencil_price : ℕ
  total_spent : ℕ

/-- Calculates the percentage discount on pencils --/
def calculate_discount_percentage (p : Purchase) : ℚ :=
  let total_cucumber_cost := p.cucumber_count * p.cucumber_price
  let total_pencil_cost := p.total_spent - total_cucumber_cost
  let full_price_pencils := p.pencil_count * p.initial_pencil_price
  let discount_amount := full_price_pencils - total_pencil_cost
  (discount_amount : ℚ) / (full_price_pencils : ℚ) * 100

/-- Theorem stating that under given conditions, the discount percentage is 20% --/
theorem discount_is_twenty_percent (p : Purchase) 
    (h1 : p.cucumber_count = 100)
    (h2 : p.cucumber_count = 2 * p.pencil_count)
    (h3 : p.cucumber_price = 20)
    (h4 : p.initial_pencil_price = 20)
    (h5 : p.total_spent = 2800) :
  calculate_discount_percentage p = 20 := by
  sorry

#eval calculate_discount_percentage ⟨100, 50, 20, 20, 2800⟩

end NUMINAMATH_CALUDE_discount_is_twenty_percent_l3577_357780


namespace NUMINAMATH_CALUDE_girls_adjacent_probability_l3577_357739

/-- The number of ways to arrange two boys and two girls in a line -/
def totalArrangements : ℕ := 24

/-- The number of ways to arrange two boys and two girls in a line with the girls adjacent -/
def favorableArrangements : ℕ := 12

/-- The probability of two girls being adjacent when two boys and two girls are randomly arranged in a line -/
def probabilityGirlsAdjacent : ℚ := favorableArrangements / totalArrangements

theorem girls_adjacent_probability :
  probabilityGirlsAdjacent = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_girls_adjacent_probability_l3577_357739


namespace NUMINAMATH_CALUDE_coles_return_speed_coles_return_speed_is_105_l3577_357762

/-- Calculates the average speed of the return trip given the conditions of Cole's journey -/
theorem coles_return_speed (speed_to_work : ℝ) (total_time : ℝ) (time_to_work : ℝ) : ℝ :=
  let distance_to_work := speed_to_work * (time_to_work / 60)
  let time_back_home := total_time - (time_to_work / 60)
  distance_to_work / time_back_home

/-- Proves that Cole's average speed driving back home is 105 km/h -/
theorem coles_return_speed_is_105 :
  coles_return_speed 75 6 210 = 105 := by
  sorry

end NUMINAMATH_CALUDE_coles_return_speed_coles_return_speed_is_105_l3577_357762


namespace NUMINAMATH_CALUDE_max_value_linear_program_l3577_357710

theorem max_value_linear_program (x y : ℝ) 
  (h1 : x - y ≥ 0) 
  (h2 : x + 2*y ≤ 3) 
  (h3 : x - 2*y ≤ 1) : 
  ∀ z, z = x + 6*y → z ≤ 7 :=
by sorry

end NUMINAMATH_CALUDE_max_value_linear_program_l3577_357710


namespace NUMINAMATH_CALUDE_lb_medium_is_control_l3577_357792

/-- Represents an experiment setup -/
structure ExperimentSetup where
  name : String
  experimental_medium : String
  control_medium : String

/-- Represents the purpose of a medium in an experiment -/
inductive MediumPurpose
  | Experimental
  | Control

/-- The purpose of preparing LB full-nutrient medium in the given experiment -/
def lb_medium_purpose (setup : ExperimentSetup) : MediumPurpose := sorry

/-- The main theorem stating the purpose of LB full-nutrient medium -/
theorem lb_medium_is_control
  (setup : ExperimentSetup)
  (h1 : setup.name = "Separation of Microorganisms in Soil Using Urea as a Nitrogen Source")
  (h2 : setup.experimental_medium = "Urea as only nitrogen source")
  (h3 : setup.control_medium = "LB full-nutrient")
  : lb_medium_purpose setup = MediumPurpose.Control := sorry

end NUMINAMATH_CALUDE_lb_medium_is_control_l3577_357792


namespace NUMINAMATH_CALUDE_video_votes_l3577_357760

theorem video_votes (total_votes : ℕ) (score : ℤ) (like_percentage : ℚ) : 
  score = 140 ∧ 
  like_percentage = 70 / 100 ∧
  (like_percentage * total_votes : ℚ).num * 1 + 
    ((1 - like_percentage) * total_votes : ℚ).num * (-1) = score ∧
  (like_percentage * total_votes : ℚ).den = 1 ∧
  ((1 - like_percentage) * total_votes : ℚ).den = 1
  → total_votes = 350 := by
sorry

end NUMINAMATH_CALUDE_video_votes_l3577_357760


namespace NUMINAMATH_CALUDE_peacock_count_l3577_357713

theorem peacock_count (total_legs total_heads : ℕ) 
  (peacock_legs peacock_heads rabbit_legs rabbit_heads : ℕ) :
  total_legs = 32 →
  total_heads = 12 →
  peacock_legs = 2 →
  peacock_heads = 1 →
  rabbit_legs = 4 →
  rabbit_heads = 1 →
  ∃ (num_peacocks num_rabbits : ℕ),
    num_peacocks * peacock_legs + num_rabbits * rabbit_legs = total_legs ∧
    num_peacocks * peacock_heads + num_rabbits * rabbit_heads = total_heads ∧
    num_peacocks = 8 :=
by sorry

end NUMINAMATH_CALUDE_peacock_count_l3577_357713


namespace NUMINAMATH_CALUDE_goose_survival_fraction_l3577_357768

theorem goose_survival_fraction :
  ∀ (total_eggs : ℕ)
    (hatched_fraction : ℚ)
    (first_month_survival_fraction : ℚ)
    (first_year_survivors : ℕ),
  total_eggs = 500 →
  hatched_fraction = 2 / 3 →
  first_month_survival_fraction = 3 / 4 →
  first_year_survivors = 100 →
  (total_eggs : ℚ) * hatched_fraction * first_month_survival_fraction > (first_year_survivors : ℚ) →
  (((total_eggs : ℚ) * hatched_fraction * first_month_survival_fraction - first_year_survivors) /
   ((total_eggs : ℚ) * hatched_fraction * first_month_survival_fraction)) = 3 / 5 := by
sorry

end NUMINAMATH_CALUDE_goose_survival_fraction_l3577_357768


namespace NUMINAMATH_CALUDE_unique_k_with_infinite_k_numbers_l3577_357700

/-- Definition of a k-number -/
def is_k_number (k n : ℕ) : Prop :=
  ∃ (r m : ℕ), r > 0 ∧ m > 0 ∧ n = r * (r + k) ∧ n = m^2 - k

/-- There are infinitely many k-numbers -/
def infinitely_many_k_numbers (k : ℕ) : Prop :=
  ∀ N : ℕ, ∃ n : ℕ, n > N ∧ is_k_number k n

/-- Theorem: k = 4 is the only positive integer with infinitely many k-numbers -/
theorem unique_k_with_infinite_k_numbers :
  ∀ k : ℕ, k > 0 → (infinitely_many_k_numbers k ↔ k = 4) :=
sorry

end NUMINAMATH_CALUDE_unique_k_with_infinite_k_numbers_l3577_357700


namespace NUMINAMATH_CALUDE_distance_equals_scientific_notation_l3577_357776

/-- Represents the distance in kilometers -/
def distance : ℝ := 30000000

/-- Represents the scientific notation of the distance -/
def scientific_notation : ℝ := 3 * (10 ^ 7)

/-- Theorem stating that the distance is equal to its scientific notation representation -/
theorem distance_equals_scientific_notation : distance = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_distance_equals_scientific_notation_l3577_357776


namespace NUMINAMATH_CALUDE_value_of_b_l3577_357729

theorem value_of_b (a b c : ℤ) 
  (eq1 : a + 5 = b) 
  (eq2 : 5 + b = c) 
  (eq3 : b + c = a) : 
  b = -10 := by
  sorry

end NUMINAMATH_CALUDE_value_of_b_l3577_357729


namespace NUMINAMATH_CALUDE_largest_multiple_seven_l3577_357733

theorem largest_multiple_seven (n : ℤ) : n = 147 ↔ 
  (∃ k : ℤ, n = 7 * k) ∧ 
  (-n > -150) ∧ 
  (∀ m : ℤ, (∃ j : ℤ, m = 7 * j) → (-m > -150) → m ≤ n) := by
sorry

end NUMINAMATH_CALUDE_largest_multiple_seven_l3577_357733


namespace NUMINAMATH_CALUDE_equation_solution_l3577_357704

theorem equation_solution : 
  {x : ℝ | x^3 + x = 1/x^3 + 1/x} = {-1, 1} := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l3577_357704


namespace NUMINAMATH_CALUDE_fib_100_102_minus_101_squared_l3577_357709

/-- Fibonacci sequence -/
def fib : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Determinant property of Fibonacci relationship -/
axiom fib_det_property (n : ℕ) : 
  fib (n + 1) * fib (n - 1) - fib n ^ 2 = (-1) ^ n

/-- Theorem: F₁₀₀ F₁₀₂ - F₁₀₁² = -1 -/
theorem fib_100_102_minus_101_squared : 
  fib 100 * fib 102 - fib 101 ^ 2 = -1 := by sorry

end NUMINAMATH_CALUDE_fib_100_102_minus_101_squared_l3577_357709


namespace NUMINAMATH_CALUDE_problem_solution_l3577_357781

-- Define the conversion rate from paise to rupees
def paise_to_rupees (paise : ℚ) : ℚ := paise / 100

-- Define the given conditions
def condition_a (a : ℚ) : Prop := 0.005 * a = paise_to_rupees 80
def condition_b (b : ℚ) : Prop := 0.0025 * b = paise_to_rupees 60
def condition_c (a b c : ℚ) : Prop := c = 0.5 * a - 0.1 * b

-- Theorem statement
theorem problem_solution (a b c : ℚ) 
  (ha : condition_a a) 
  (hb : condition_b b) 
  (hc : condition_c a b c) : 
  a = 160 ∧ b = 240 ∧ c = 56 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3577_357781


namespace NUMINAMATH_CALUDE_unique_function_property_l3577_357717

theorem unique_function_property (f : ℤ → ℤ) 
  (h1 : f 0 = 1)
  (h2 : ∀ (n : ℕ), f (f n) = n)
  (h3 : ∀ (n : ℕ), f (f (n + 2) + 2) = n) :
  ∀ (n : ℤ), f n = 1 - n :=
by sorry

end NUMINAMATH_CALUDE_unique_function_property_l3577_357717


namespace NUMINAMATH_CALUDE_equation_solution_l3577_357714

theorem equation_solution :
  ∃ y : ℚ, (4 / 7) * (1 / 5) * y - 2 = 14 ∧ y = 140 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3577_357714


namespace NUMINAMATH_CALUDE_M_equals_N_l3577_357722

def M : Set ℝ := {x | ∃ k : ℤ, x = (2 * k + 1) * Real.pi}
def N : Set ℝ := {x | ∃ k : ℤ, x = (2 * k - 1) * Real.pi}

theorem M_equals_N : M = N := by
  sorry

end NUMINAMATH_CALUDE_M_equals_N_l3577_357722


namespace NUMINAMATH_CALUDE_hyperbola_vertices_distance_l3577_357744

-- Define the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 144 - y^2 / 49 = 1

-- Define the distance between vertices
def distance_between_vertices : ℝ := 24

-- Theorem statement
theorem hyperbola_vertices_distance :
  ∀ x y : ℝ, hyperbola_equation x y →
  distance_between_vertices = 24 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_vertices_distance_l3577_357744


namespace NUMINAMATH_CALUDE_smallest_n_for_radio_profit_l3577_357708

theorem smallest_n_for_radio_profit (n d : ℕ) : 
  d > 0 → 
  (n : ℚ) * ((n : ℚ) - 11) = (d : ℚ) / 8 → 
  (∃ k : ℕ, d = 8 * k) →
  n ≥ 11 →
  (∀ m : ℕ, m < n → m ≥ 11 → (m : ℚ) * ((m : ℚ) - 11) ≠ (d : ℚ) / 8 ∨ ¬(∃ k : ℕ, d = 8 * k)) →
  n = 12 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_radio_profit_l3577_357708
