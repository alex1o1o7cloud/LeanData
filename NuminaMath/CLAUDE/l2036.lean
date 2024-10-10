import Mathlib

namespace rhombus_diagonal_length_l2036_203648

/-- A rhombus with given area and diagonal ratio has a specific longer diagonal length -/
theorem rhombus_diagonal_length 
  (area : ℝ) 
  (diagonal_ratio : ℚ) 
  (h_area : area = 150) 
  (h_ratio : diagonal_ratio = 4 / 3) : 
  ∃ (d1 d2 : ℝ), d1 > d2 ∧ d1 / d2 = diagonal_ratio ∧ area = (d1 * d2) / 2 ∧ d1 = 20 := by
  sorry

end rhombus_diagonal_length_l2036_203648


namespace glucose_solution_volume_l2036_203684

/-- The concentration of glucose in the solution in grams per 100 cubic centimeters -/
def glucose_concentration : ℝ := 10

/-- The volume of solution in cubic centimeters that contains 100 grams of glucose -/
def reference_volume : ℝ := 100

/-- The amount of glucose in grams poured into the container -/
def glucose_in_container : ℝ := 4.5

/-- The volume of solution poured into the container in cubic centimeters -/
def volume_poured : ℝ := 45

theorem glucose_solution_volume :
  (glucose_concentration / reference_volume) * volume_poured = glucose_in_container :=
sorry

end glucose_solution_volume_l2036_203684


namespace papayas_needed_l2036_203679

/-- The number of papayas Jake can eat in a week -/
def jake_weekly : ℕ := 3

/-- The number of papayas Jake's brother can eat in a week -/
def brother_weekly : ℕ := 5

/-- The number of papayas Jake's father can eat in a week -/
def father_weekly : ℕ := 4

/-- The number of weeks to account for -/
def num_weeks : ℕ := 4

/-- The total number of papayas needed for the given number of weeks -/
def total_papayas : ℕ := (jake_weekly + brother_weekly + father_weekly) * num_weeks

theorem papayas_needed : total_papayas = 48 := by
  sorry

end papayas_needed_l2036_203679


namespace election_votes_calculation_l2036_203659

theorem election_votes_calculation (total_votes : ℕ) : 
  (85 : ℚ) / 100 * ((85 : ℚ) / 100 * total_votes) = 404600 →
  total_votes = 560000 := by
  sorry

end election_votes_calculation_l2036_203659


namespace root_implies_b_eq_neg_20_l2036_203693

-- Define the polynomial
def f (a b : ℚ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + 16

-- State the theorem
theorem root_implies_b_eq_neg_20 (a b : ℚ) :
  f a b (Real.sqrt 5 + 3) = 0 → b = -20 := by
  sorry

end root_implies_b_eq_neg_20_l2036_203693


namespace currency_conversion_weight_conversion_gram_to_kg_weight_conversion_kg_to_ton_length_conversion_l2036_203669

-- Define conversion rates
def yuan_to_jiao : ℚ := 10
def yuan_to_fen : ℚ := 100
def kg_to_gram : ℚ := 1000
def ton_to_kg : ℚ := 1000
def meter_to_cm : ℚ := 100

-- Define the conversion functions
def jiao_to_yuan (j : ℚ) : ℚ := j / yuan_to_jiao
def fen_to_yuan (f : ℚ) : ℚ := f / yuan_to_fen
def gram_to_kg (g : ℚ) : ℚ := g / kg_to_gram
def kg_to_ton (k : ℚ) : ℚ := k / ton_to_kg
def cm_to_meter (c : ℚ) : ℚ := c / meter_to_cm

-- Theorem statements
theorem currency_conversion :
  5 + jiao_to_yuan 4 + fen_to_yuan 8 = 5.48 := by sorry

theorem weight_conversion_gram_to_kg :
  gram_to_kg 80 = 0.08 := by sorry

theorem weight_conversion_kg_to_ton :
  kg_to_ton 73 = 0.073 := by sorry

theorem length_conversion :
  1 + cm_to_meter 5 = 1.05 := by sorry

end currency_conversion_weight_conversion_gram_to_kg_weight_conversion_kg_to_ton_length_conversion_l2036_203669


namespace not_both_odd_l2036_203677

theorem not_both_odd (m n : ℕ) (h : (1 : ℚ) / m + (1 : ℚ) / n = (1 : ℚ) / 2020) :
  Even m ∨ Even n :=
sorry

end not_both_odd_l2036_203677


namespace jane_score_is_14_l2036_203651

/-- Represents a mathematics competition with a scoring system. -/
structure MathCompetition where
  totalQuestions : ℕ
  correctAnswers : ℕ
  incorrectAnswers : ℕ
  unansweredQuestions : ℕ
  correctPoints : ℚ
  incorrectPenalty : ℚ

/-- Calculates the total score for a given math competition. -/
def calculateScore (comp : MathCompetition) : ℚ :=
  comp.correctAnswers * comp.correctPoints - comp.incorrectAnswers * comp.incorrectPenalty

/-- Theorem stating that Jane's score in the competition is 14 points. -/
theorem jane_score_is_14 (comp : MathCompetition)
  (h1 : comp.totalQuestions = 35)
  (h2 : comp.correctAnswers = 17)
  (h3 : comp.incorrectAnswers = 12)
  (h4 : comp.unansweredQuestions = 6)
  (h5 : comp.correctPoints = 1)
  (h6 : comp.incorrectPenalty = 1/4)
  (h7 : comp.totalQuestions = comp.correctAnswers + comp.incorrectAnswers + comp.unansweredQuestions) :
  calculateScore comp = 14 := by
  sorry

end jane_score_is_14_l2036_203651


namespace exists_solution_a4_eq_b3_plus_c2_l2036_203629

theorem exists_solution_a4_eq_b3_plus_c2 : 
  ∃ (a b c : ℕ+), (a : ℝ)^4 = (b : ℝ)^3 + (c : ℝ)^2 := by
  sorry

end exists_solution_a4_eq_b3_plus_c2_l2036_203629


namespace river_depth_calculation_l2036_203606

theorem river_depth_calculation (depth_mid_may : ℝ) : 
  let depth_mid_june := depth_mid_may + 10
  let depth_june_20 := depth_mid_june - 5
  let depth_july_5 := depth_june_20 + 8
  let depth_mid_july := depth_july_5
  depth_mid_july = 45 → depth_mid_may = 32 := by sorry

end river_depth_calculation_l2036_203606


namespace negation_of_all_x_squared_nonnegative_l2036_203624

theorem negation_of_all_x_squared_nonnegative :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x : ℝ, x^2 < 0) := by sorry

end negation_of_all_x_squared_nonnegative_l2036_203624


namespace extremum_of_f_under_constraint_l2036_203661

-- Define the function f
def f (x y : ℝ) : ℝ := x^2 + y^2 - 2*x - y

-- Define the constraint function φ
def φ (x y : ℝ) : ℝ := x + y - 1

-- State the theorem
theorem extremum_of_f_under_constraint :
  ∃ (x y : ℝ),
    φ x y = 0 ∧
    (∀ (x' y' : ℝ), φ x' y' = 0 → f x' y' ≥ f x y) ∧
    x = 3/4 ∧ y = 1/4 ∧ f x y = -9/8 :=
sorry

end extremum_of_f_under_constraint_l2036_203661


namespace smallest_third_term_of_geometric_progression_l2036_203686

/-- Given an arithmetic progression with first term 7, if adding 3 to the second term and 25 to the third term
    results in a geometric progression, then the smallest possible value for the third term of the geometric
    progression is -0.62. -/
theorem smallest_third_term_of_geometric_progression (d : ℝ) :
  let a₁ := 7
  let a₂ := a₁ + d
  let a₃ := a₁ + 2*d
  let g₁ := a₁
  let g₂ := a₂ + 3
  let g₃ := a₃ + 25
  (g₂^2 = g₁ * g₃) →
  ∃ (d' : ℝ), g₃ ≥ -0.62 ∧ (∀ (d'' : ℝ),
    let g₁' := 7
    let g₂' := (7 + d'') + 3
    let g₃' := (7 + 2*d'') + 25
    (g₂'^2 = g₁' * g₃') → g₃' ≥ g₃) :=
by sorry

end smallest_third_term_of_geometric_progression_l2036_203686


namespace matching_color_probability_l2036_203635

-- Define the number of jelly beans for each person
def abe_green : ℕ := 1
def abe_red : ℕ := 2
def bob_green : ℕ := 2
def bob_yellow : ℕ := 1
def bob_red : ℕ := 1

-- Define the total number of jelly beans for each person
def abe_total : ℕ := abe_green + abe_red
def bob_total : ℕ := bob_green + bob_yellow + bob_red

-- Define the probability of matching colors
def prob_match : ℚ := (abe_green * bob_green + abe_red * bob_red) / (abe_total * bob_total)

-- Theorem statement
theorem matching_color_probability :
  prob_match = 1 / 3 := by sorry

end matching_color_probability_l2036_203635


namespace x_intercept_ratio_l2036_203654

-- Define the slopes and y-intercept
def m₁ : ℚ := 8
def m₂ : ℚ := 4
def b : ℚ := 5

-- Define the x-intercepts
def s : ℚ := -b / m₁
def t : ℚ := -b / m₂

-- Theorem statement
theorem x_intercept_ratio :
  s / t = 1 / 2 := by sorry

end x_intercept_ratio_l2036_203654


namespace product_price_relationship_l2036_203649

/-- Proves the relationship between fall and spring prices of a product given specific conditions -/
theorem product_price_relationship (fall_amount : ℝ) (total_cost : ℝ) (spring_difference : ℝ) : 
  fall_amount = 550 ∧ 
  total_cost = 825 ∧ 
  spring_difference = 220 →
  ∃ (spring_price : ℝ),
    spring_price = total_cost / (fall_amount - spring_difference) ∧
    spring_price = total_cost / fall_amount + 1 ∧
    spring_price = 1.5 :=
by sorry

end product_price_relationship_l2036_203649


namespace instantaneous_velocity_at_3_seconds_l2036_203668

-- Define the displacement function
def s (t : ℝ) : ℝ := 4 - 2*t + t^2

-- State the theorem
theorem instantaneous_velocity_at_3_seconds :
  (deriv s) 3 = 4 := by sorry

end instantaneous_velocity_at_3_seconds_l2036_203668


namespace zero_exponent_l2036_203658

theorem zero_exponent (x : ℝ) (hx : x ≠ 0) : x^0 = 1 := by
  sorry

end zero_exponent_l2036_203658


namespace dining_bill_calculation_l2036_203639

theorem dining_bill_calculation (total_bill : ℝ) (tax_rate : ℝ) (tip_rate : ℝ) 
  (h1 : total_bill = 198)
  (h2 : tax_rate = 0.1)
  (h3 : tip_rate = 0.2) : 
  ∃ (food_price : ℝ), 
    food_price * (1 + tax_rate) * (1 + tip_rate) = total_bill ∧ 
    food_price = 150 := by
  sorry

end dining_bill_calculation_l2036_203639


namespace abc_sum_bound_l2036_203681

theorem abc_sum_bound (a b c : ℝ) (h : a + b + c = 1) :
  ∃ (x : ℝ), x ≤ 1/2 ∧ ∃ (a' b' c' : ℝ), a' + b' + c' = 1 ∧ a'*b' + a'*c' + b'*c' = x :=
sorry

end abc_sum_bound_l2036_203681


namespace mans_current_age_l2036_203625

/-- Given a man and his son, where the man is thrice as old as his son now,
    and after 12 years he will be twice as old as his son,
    prove that the man's current age is 36 years. -/
theorem mans_current_age (man_age son_age : ℕ) : 
  man_age = 3 * son_age →
  man_age + 12 = 2 * (son_age + 12) →
  man_age = 36 := by
  sorry

end mans_current_age_l2036_203625


namespace problem_1_problem_2_l2036_203663

theorem problem_1 (a : ℝ) : (-a^3)^2 * (-a^2)^3 / a = -a^11 := by sorry

theorem problem_2 (m n : ℝ) : (m - n)^3 * (n - m)^4 * (n - m)^5 = -(n - m)^12 := by sorry

end problem_1_problem_2_l2036_203663


namespace f_increasing_on_interval_l2036_203653

def f (x : ℝ) : ℝ := -x^2 + 2*x

theorem f_increasing_on_interval :
  ∀ (x₁ x₂ : ℝ), x₁ < x₂ → x₁ < 1 → x₂ < 1 → f x₁ < f x₂ := by
  sorry

end f_increasing_on_interval_l2036_203653


namespace square_plus_reciprocal_square_l2036_203623

theorem square_plus_reciprocal_square (n : ℝ) (h : n + 1/n = 10) : n^2 + 1/n^2 + 4 = 102 := by
  sorry

end square_plus_reciprocal_square_l2036_203623


namespace expression_undefined_at_eight_l2036_203613

/-- The expression is undefined when x = 8 -/
theorem expression_undefined_at_eight :
  ∀ x : ℝ, x = 8 → (x^2 - 16*x + 64 = 0) := by sorry

end expression_undefined_at_eight_l2036_203613


namespace sum_of_powers_of_two_l2036_203620

theorem sum_of_powers_of_two (n : ℕ) : 
  (1 : ℚ) / 2^10 + 1 / 2^9 + 1 / 2^8 = n / 2^10 → n = 7 := by
  sorry

end sum_of_powers_of_two_l2036_203620


namespace triangle_tangent_ratio_l2036_203644

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a * cos(B) - b * cos(A) = (4/5) * c, then tan(A) / tan(B) = 9 -/
theorem triangle_tangent_ratio (a b c : ℝ) (A B C : Real) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  a * Real.cos B - b * Real.cos A = (4/5) * c →
  Real.tan A / Real.tan B = 9 := by
  sorry

end triangle_tangent_ratio_l2036_203644


namespace lcm_75_120_l2036_203619

theorem lcm_75_120 : Nat.lcm 75 120 = 600 := by
  sorry

end lcm_75_120_l2036_203619


namespace similar_triangles_side_length_l2036_203692

/-- Two triangles XYZ and PQR are similar with a shared angle of 150 degrees. 
    Given the side lengths XY = 10, XZ = 20, and PR = 12, prove that PQ = 2.5. -/
theorem similar_triangles_side_length 
  (XY : ℝ) (XZ : ℝ) (PR : ℝ) (PQ : ℝ) 
  (h1 : XY = 10) 
  (h2 : XZ = 20) 
  (h3 : PR = 12) 
  (h4 : ∃ θ : ℝ, θ = 150 * π / 180) -- 150 degrees in radians
  (h5 : XY / PQ = XZ / PR) : -- similarity condition
  PQ = 2.5 := by
sorry

end similar_triangles_side_length_l2036_203692


namespace triangle_determinant_l2036_203675

theorem triangle_determinant (A B C : Real) (h1 : A = 45 * π / 180)
    (h2 : B = 75 * π / 180) (h3 : C = 60 * π / 180) :
  let M : Matrix (Fin 3) (Fin 3) Real :=
    ![![Real.tan A, 1, 1],
      ![1, Real.tan B, 1],
      ![1, 1, Real.tan C]]
  Matrix.det M = -1 := by
  sorry

end triangle_determinant_l2036_203675


namespace min_value_of_function_max_sum_with_constraint_l2036_203666

-- Part 1
theorem min_value_of_function (x : ℝ) (h : x > -1) :
  ∃ (min_y : ℝ), min_y = 9 ∧ ∀ y, y = (x^2 + 7*x + 10) / (x + 1) → y ≥ min_y :=
sorry

-- Part 2
theorem max_sum_with_constraint (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2*x + 8*y - x*y = 0) :
  x + y ≤ 18 :=
sorry

end min_value_of_function_max_sum_with_constraint_l2036_203666


namespace kolya_role_is_collection_agency_l2036_203688

-- Define the actors in the scenario
inductive Actor : Type
| Katya : Actor
| Vasya : Actor
| Kolya : Actor

-- Define the possible roles
inductive Role : Type
| FinancialPyramid : Role
| CollectionAgency : Role
| Bank : Role
| InsuranceCompany : Role

-- Define the scenario
structure BookLendingScenario where
  lender : Actor
  borrower : Actor
  mediator : Actor
  books_lent : ℕ
  return_period : ℕ
  books_not_returned : Bool
  mediator_reward : ℕ

-- Define the characteristics of a collection agency
def is_collection_agency (r : Role) : Prop :=
  r = Role.CollectionAgency

-- Define the function to determine the role based on the scenario
def determine_role (s : BookLendingScenario) : Role :=
  Role.CollectionAgency

-- Theorem statement
theorem kolya_role_is_collection_agency (s : BookLendingScenario) :
  s.lender = Actor.Katya ∧
  s.borrower = Actor.Vasya ∧
  s.mediator = Actor.Kolya ∧
  s.books_lent > 0 ∧
  s.return_period > 0 ∧
  s.books_not_returned = true ∧
  s.mediator_reward > 0 →
  is_collection_agency (determine_role s) :=
sorry

end kolya_role_is_collection_agency_l2036_203688


namespace min_sum_of_slopes_l2036_203682

/-- Parabola equation -/
def parabola (x y : ℝ) : Prop :=
  x^2 - 4*(x+y) + y^2 = 2*x*y + 8

/-- Tangent line to the parabola at point (a, b) -/
def tangent_line (a b x y : ℝ) : Prop :=
  y - b = ((b - a + 2) / (b - a - 2)) * (x - a)

/-- Intersection point of the tangent lines -/
def intersection_point (p q : ℝ) : Prop :=
  p + q = -32

theorem min_sum_of_slopes :
  ∃ (a b p q : ℝ),
    parabola a b ∧
    parabola b a ∧
    intersection_point p q ∧
    tangent_line a b p q ∧
    tangent_line b a p q ∧
    ((b - a + 2) / (b - a - 2) + (a - b + 2) / (a - b - 2) ≥ 62 / 29) :=
by sorry

end min_sum_of_slopes_l2036_203682


namespace part_one_part_two_l2036_203640

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x < 4}
def B : Set ℝ := {x | 3 * x - 7 ≥ 8 - 2 * x}

-- Part 1
theorem part_one :
  let a := 2
  (A a ∩ B = {x | 3 ≤ x ∧ x < 4}) ∧
  ((Set.univ \ A a) ∪ (Set.univ \ B) = {x | x < 3 ∨ x ≥ 4}) := by sorry

-- Part 2
theorem part_two (a : ℝ) :
  A a ∩ B = A a → a ≥ 3 := by sorry

end part_one_part_two_l2036_203640


namespace cos_two_alpha_minus_pi_sixth_l2036_203646

theorem cos_two_alpha_minus_pi_sixth (α : ℝ) 
  (h1 : 0 < α) (h2 : α < π/3) 
  (h3 : Real.sin (α + π/6) = 2 * Real.sqrt 5 / 5) : 
  Real.cos (2*α - π/6) = 4/5 := by
  sorry

end cos_two_alpha_minus_pi_sixth_l2036_203646


namespace pure_imaginary_complex_number_l2036_203628

theorem pure_imaginary_complex_number (m : ℝ) : 
  let z : ℂ := (1 + m * Complex.I) / (1 - Complex.I)
  (∃ (y : ℝ), z = Complex.I * y) → m = 1 := by
  sorry

end pure_imaginary_complex_number_l2036_203628


namespace page_number_added_twice_l2036_203630

theorem page_number_added_twice 
  (n : ℕ) 
  (h1 : 60 ≤ n ∧ n ≤ 70) 
  (h2 : ∃ k : ℕ, k ≤ n ∧ n * (n + 1) / 2 + k = 2378) : 
  ∃ k : ℕ, k ≤ n ∧ n * (n + 1) / 2 + k = 2378 ∧ k = 32 := by
  sorry

end page_number_added_twice_l2036_203630


namespace game_probabilities_l2036_203694

/-- Represents the game between players A and B -/
structure Game where
  oddProbA : ℝ
  evenProbB : ℝ
  maxRounds : ℕ

/-- Calculates the probability of the 4th round determining the winner and A winning -/
def probAWinsFourth (g : Game) : ℝ := sorry

/-- Calculates the mathematical expectation of the total number of rounds played -/
def expectedRounds (g : Game) : ℝ := sorry

/-- The main theorem about the game -/
theorem game_probabilities (g : Game) 
  (h1 : g.oddProbA = 2/3)
  (h2 : g.evenProbB = 2/3)
  (h3 : g.maxRounds = 8) :
  probAWinsFourth g = 10/81 ∧ expectedRounds g = 2968/729 := by sorry

end game_probabilities_l2036_203694


namespace favorite_movies_total_length_l2036_203627

theorem favorite_movies_total_length : 
  ∀ (michael joyce nikki ryn sam alex : ℝ),
    nikki = 30 →
    michael = nikki / 3 →
    joyce = michael + 2 →
    ryn = nikki * (4/5) →
    sam = joyce * 1.5 →
    alex = 2 * (min michael (min joyce (min nikki (min ryn sam)))) →
    michael + joyce + nikki + ryn + sam + alex = 114 :=
by
  sorry

end favorite_movies_total_length_l2036_203627


namespace laundry_water_usage_l2036_203671

/-- Calculates the total water usage for a set of laundry loads -/
def total_water_usage (heavy_wash_gallons : ℕ) (regular_wash_gallons : ℕ) (light_wash_gallons : ℕ)
  (heavy_loads : ℕ) (regular_loads : ℕ) (light_loads : ℕ) (bleached_loads : ℕ) : ℕ :=
  heavy_wash_gallons * heavy_loads +
  regular_wash_gallons * regular_loads +
  light_wash_gallons * (light_loads + bleached_loads)

/-- Proves that the total water usage for the given laundry scenario is 76 gallons -/
theorem laundry_water_usage :
  total_water_usage 20 10 2 2 3 1 2 = 76 := by sorry

end laundry_water_usage_l2036_203671


namespace inequality_proof_l2036_203612

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  (1 / (a^3 * (b + c))) + (1 / (b^3 * (c + a))) + (1 / (c^3 * (a + b))) ≥ 3/2 := by
  sorry

end inequality_proof_l2036_203612


namespace quadratic_roots_relation_l2036_203609

theorem quadratic_roots_relation (d e : ℝ) : 
  (∃ r s : ℝ, 2 * r^2 - 4 * r - 6 = 0 ∧ 2 * s^2 - 4 * s - 6 = 0 ∧
   ∀ x : ℝ, x^2 + d * x + e = 0 ↔ x = r - 3 ∨ x = s - 3) →
  e = 0 :=
by sorry

end quadratic_roots_relation_l2036_203609


namespace james_hall_of_mirrors_glass_area_l2036_203601

/-- The total area of glass needed for three walls in a hall of mirrors --/
def total_glass_area (long_wall_length long_wall_height short_wall_length short_wall_height : ℝ) : ℝ :=
  2 * (long_wall_length * long_wall_height) + (short_wall_length * short_wall_height)

/-- Theorem: The total area of glass needed for James' hall of mirrors is 960 square feet --/
theorem james_hall_of_mirrors_glass_area :
  total_glass_area 30 12 20 12 = 960 := by
  sorry

end james_hall_of_mirrors_glass_area_l2036_203601


namespace product_of_three_rationals_l2036_203642

theorem product_of_three_rationals (a b c : ℚ) :
  a * b * c < 0 → (a < 0 ∧ b ≥ 0 ∧ c ≥ 0) ∨
                   (a ≥ 0 ∧ b < 0 ∧ c ≥ 0) ∨
                   (a ≥ 0 ∧ b ≥ 0 ∧ c < 0) ∨
                   (a < 0 ∧ b < 0 ∧ c < 0) :=
by sorry

end product_of_three_rationals_l2036_203642


namespace problem_solution_l2036_203637

def M (a : ℝ) : Set ℝ := {x | x * (x - a - 1) < 0}
def N : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

theorem problem_solution :
  (M 1 = {x | 0 < x ∧ x < 2}) ∧
  ({a : ℝ | M a ⊆ N} = Set.Icc (-2) 2) := by
  sorry

end problem_solution_l2036_203637


namespace black_region_area_is_56_l2036_203696

/-- The area of the region between two squares, where a smaller square is entirely contained 
    within a larger square. -/
def black_region_area (small_side : ℝ) (large_side : ℝ) : ℝ :=
  large_side ^ 2 - small_side ^ 2

/-- Theorem stating that the area of the black region between two squares with given side lengths
    is 56 square units. -/
theorem black_region_area_is_56 :
  black_region_area 5 9 = 56 := by
  sorry

end black_region_area_is_56_l2036_203696


namespace inequality_theorem_largest_c_k_zero_largest_c_k_two_l2036_203665

theorem inequality_theorem :
  ∀ (k : ℝ), 
  (∃ (c_k : ℝ), c_k > 0 ∧ 
    (∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 → 
      (x^2 + 1) * (y^2 + 1) * (z^2 + 1) ≥ c_k * (x + y + z)^k)) ↔ 
  (0 ≤ k ∧ k ≤ 2) :=
sorry

theorem largest_c_k_zero :
  ∀ (c : ℝ), c > 0 →
  (∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 →
    (x^2 + 1) * (y^2 + 1) * (z^2 + 1) ≥ c) →
  c ≤ 1 :=
sorry

theorem largest_c_k_two :
  ∀ (c : ℝ), c > 0 →
  (∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 →
    (x^2 + 1) * (y^2 + 1) * (z^2 + 1) ≥ c * (x + y + z)^2) →
  c ≤ 8/9 :=
sorry

end inequality_theorem_largest_c_k_zero_largest_c_k_two_l2036_203665


namespace product_of_integers_l2036_203650

theorem product_of_integers (A B C D : ℕ+) : 
  A + B + C + D = 51 →
  A = 2 * C - 3 →
  B = 2 * C + 3 →
  D = 5 * C + 1 →
  A * B * C * D = 14910 := by
sorry

end product_of_integers_l2036_203650


namespace correct_stratified_sample_size_l2036_203672

/-- Represents the student population in each year -/
structure StudentPopulation where
  first_year : ℕ
  second_year : ℕ
  third_year : ℕ

/-- Calculates the stratified sample size for each year -/
def stratifiedSampleSize (population : StudentPopulation) (total_sample : ℕ) : ℕ × ℕ × ℕ :=
  sorry

/-- Theorem stating that the calculated stratified sample sizes are correct -/
theorem correct_stratified_sample_size 
  (population : StudentPopulation)
  (h1 : population.first_year = 540)
  (h2 : population.second_year = 440)
  (h3 : population.third_year = 420)
  (total_sample : ℕ)
  (h4 : total_sample = 70) :
  stratifiedSampleSize population total_sample = (27, 22, 21) :=
sorry

end correct_stratified_sample_size_l2036_203672


namespace circle_inequality_abc_inequality_l2036_203647

-- Problem I
theorem circle_inequality (x y : ℝ) (h : x^2 + y^2 = 1) : 
  -Real.sqrt 13 ≤ 2*x + 3*y ∧ 2*x + 3*y ≤ Real.sqrt 13 := by
  sorry

-- Problem II
theorem abc_inequality (a b c : ℝ) (h : a^2 + b^2 + c^2 - 2*a - 2*b - 2*c = 0) :
  2*a - b - c ≤ 3 * Real.sqrt 2 := by
  sorry

end circle_inequality_abc_inequality_l2036_203647


namespace double_points_on_quadratic_l2036_203617

/-- A double point is a point where the ordinate is twice its abscissa. -/
def is_double_point (x y : ℝ) : Prop := y = 2 * x

/-- The quadratic function y = x² + 2mx - m -/
def quadratic_function (m : ℝ) (x : ℝ) : ℝ := x^2 + 2*m*x - m

theorem double_points_on_quadratic (m : ℝ) :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    is_double_point x₁ y₁ ∧
    is_double_point x₂ y₂ ∧
    y₁ = quadratic_function m x₁ ∧
    y₂ = quadratic_function m x₂ ∧
    x₁ < 1 ∧ 1 < x₂ →
    m < 1 :=
sorry

end double_points_on_quadratic_l2036_203617


namespace geometric_series_sum_l2036_203676

theorem geometric_series_sum (a : ℝ) (r : ℝ) (n : ℕ) (h1 : a = 3) (h2 : r = -2) (h3 : a * r^(n-1) = -1536) :
  (a * (1 - r^n)) / (1 - r) = -1023 := by
  sorry

end geometric_series_sum_l2036_203676


namespace power_boat_travel_time_l2036_203674

/-- Represents the scenario of a power boat and raft traveling on a river --/
structure RiverJourney where
  r : ℝ  -- Speed of the river current (km/h)
  p : ℝ  -- Speed of the power boat relative to the river (km/h)
  t : ℝ  -- Time for power boat to travel from A to B (hours)
  s : ℝ  -- Stopping time at dock B (hours)

/-- The theorem stating that the time for the power boat to travel from A to B is 5 hours --/
theorem power_boat_travel_time 
  (journey : RiverJourney) 
  (h1 : journey.r > 0)  -- River speed is positive
  (h2 : journey.p > journey.r)  -- Power boat is faster than river current
  (h3 : journey.s = 1)  -- Stopping time is 1 hour
  (h4 : (journey.p + journey.r) * journey.t + (journey.p - journey.r) * (12 - journey.t - journey.s) = 12 * journey.r)  -- Distance equation
  : journey.t = 5 := by
  sorry

end power_boat_travel_time_l2036_203674


namespace divisibility_by_five_l2036_203645

theorem divisibility_by_five (a b : ℕ+) : 
  (5 ∣ (a * b)) → (5 ∣ a) ∨ (5 ∣ b) := by
  sorry

end divisibility_by_five_l2036_203645


namespace zoo_arrangements_l2036_203641

/-- The number of letters in the word "ZOO₁M₁O₂M₂O₃" -/
def word_length : ℕ := 7

/-- The number of distinct arrangements of the letters in "ZOO₁M₁O₂M₂O₃" -/
def num_arrangements : ℕ := Nat.factorial word_length

theorem zoo_arrangements :
  num_arrangements = 5040 := by sorry

end zoo_arrangements_l2036_203641


namespace line_equation_l2036_203690

/-- Given a line L with slope -3 and y-intercept 7, its equation is y = -3x + 7 -/
theorem line_equation (L : Set (ℝ × ℝ)) (slope : ℝ) (y_intercept : ℝ)
  (h1 : slope = -3)
  (h2 : y_intercept = 7)
  (h3 : ∀ (x y : ℝ), (x, y) ∈ L ↔ y = slope * x + y_intercept) :
  ∀ (x y : ℝ), (x, y) ∈ L ↔ y = -3 * x + 7 :=
by sorry

end line_equation_l2036_203690


namespace least_n_satisfying_inequality_l2036_203621

theorem least_n_satisfying_inequality : 
  (∀ k : ℕ, k > 0 ∧ k < 10 → (1 : ℚ) / k - (1 : ℚ) / (k + 1) ≥ (1 : ℚ) / 100) ∧
  ((1 : ℚ) / 10 - (1 : ℚ) / 11 < (1 : ℚ) / 100) :=
by sorry

end least_n_satisfying_inequality_l2036_203621


namespace largest_coefficient_binomial_expansion_l2036_203631

theorem largest_coefficient_binomial_expansion :
  let n : ℕ := 7
  let expansion := fun (k : ℕ) => Nat.choose n k
  (∃ k : ℕ, k ≤ n ∧ expansion k = Finset.sup (Finset.range (n + 1)) expansion) →
  Finset.sup (Finset.range (n + 1)) expansion = 35 :=
by sorry

end largest_coefficient_binomial_expansion_l2036_203631


namespace great_white_shark_teeth_l2036_203614

/-- The number of teeth of different shark species -/
def shark_teeth : ℕ → ℕ
| 0 => 180  -- tiger shark
| 1 => shark_teeth 0 / 6  -- hammerhead shark
| 2 => 2 * (shark_teeth 0 + shark_teeth 1)  -- great white shark
| _ => 0  -- other sharks (not relevant for this problem)

theorem great_white_shark_teeth : shark_teeth 2 = 420 := by
  sorry

end great_white_shark_teeth_l2036_203614


namespace platform_length_l2036_203643

/-- Given a train of length 200 meters that crosses a platform in 50 seconds
    and a signal pole in 42 seconds, the length of the platform is 38 meters. -/
theorem platform_length (train_length : ℝ) (platform_crossing_time : ℝ) (pole_crossing_time : ℝ)
  (h1 : train_length = 200)
  (h2 : platform_crossing_time = 50)
  (h3 : pole_crossing_time = 42) :
  let train_speed := train_length / pole_crossing_time
  let platform_length := train_speed * platform_crossing_time - train_length
  platform_length = 38 := by
sorry

end platform_length_l2036_203643


namespace inscribed_circle_radius_rhombus_l2036_203633

/-- The radius of a circle inscribed in a rhombus with given diagonals -/
theorem inscribed_circle_radius_rhombus (d1 d2 : ℝ) (h1 : d1 = 12) (h2 : d2 = 16) :
  let a := Real.sqrt ((d1/2)^2 + (d2/2)^2)
  let r := (d1 * d2) / (4 * a)
  r = 24 / 5 := by
  sorry

end inscribed_circle_radius_rhombus_l2036_203633


namespace monotonic_range_of_a_l2036_203605

def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + a*x^2 - x - 2

theorem monotonic_range_of_a :
  ∀ a : ℝ, (∀ x : ℝ, Monotone (f a)) ↔ a ∈ Set.Icc (-Real.sqrt 3) (Real.sqrt 3) :=
sorry

end monotonic_range_of_a_l2036_203605


namespace percentage_saved_approx_11_percent_l2036_203662

def original_price : ℝ := 30
def amount_saved : ℝ := 3
def amount_spent : ℝ := 24

theorem percentage_saved_approx_11_percent :
  let actual_price := amount_spent + amount_saved
  let percentage_saved := (amount_saved / actual_price) * 100
  ∃ ε > 0, abs (percentage_saved - 11) < ε :=
by sorry

end percentage_saved_approx_11_percent_l2036_203662


namespace exactly_three_red_marbles_l2036_203602

def total_marbles : ℕ := 15
def red_marbles : ℕ := 8
def blue_marbles : ℕ := 7
def trials : ℕ := 6
def target_red : ℕ := 3

def probability_red : ℚ := red_marbles / total_marbles
def probability_blue : ℚ := blue_marbles / total_marbles

theorem exactly_three_red_marbles :
  (Nat.choose trials target_red : ℚ) *
  probability_red ^ target_red *
  probability_blue ^ (trials - target_red) =
  6881280 / 38107875 :=
sorry

end exactly_three_red_marbles_l2036_203602


namespace certification_cost_certification_cost_proof_l2036_203697

/-- The cost of certification for John's seeing-eye dog --/
theorem certification_cost : ℝ → Prop :=
  fun c =>
    let adoption_fee : ℝ := 150
    let training_cost_per_week : ℝ := 250
    let training_weeks : ℝ := 12
    let insurance_coverage_percent : ℝ := 90
    let total_out_of_pocket : ℝ := 3450
    let total_cost_before_certification : ℝ := adoption_fee + training_cost_per_week * training_weeks
    let out_of_pocket_certification : ℝ := c * (100 - insurance_coverage_percent) / 100
    total_out_of_pocket = total_cost_before_certification + out_of_pocket_certification →
    c = 3000

/-- Proof of the certification cost --/
theorem certification_cost_proof : certification_cost 3000 := by
  sorry

end certification_cost_certification_cost_proof_l2036_203697


namespace camel_count_l2036_203626

/-- The cost of an elephant in rupees -/
def elephant_cost : ℚ := 12000

/-- The cost of an ox in rupees -/
def ox_cost : ℚ := 8000

/-- The cost of a horse in rupees -/
def horse_cost : ℚ := 2000

/-- The cost of a camel in rupees -/
def camel_cost : ℚ := 4800

/-- The number of camels -/
def num_camels : ℚ := 10

theorem camel_count :
  (6 * ox_cost = 4 * elephant_cost) →
  (16 * horse_cost = 4 * ox_cost) →
  (24 * horse_cost = num_camels * camel_cost) →
  (10 * elephant_cost = 120000) →
  num_camels = 10 := by
  sorry

end camel_count_l2036_203626


namespace range_of_a_in_acute_triangle_l2036_203632

/-- Given an acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if b^2 - a^2 = ac and c = 2, then 2/3 < a < 2 -/
theorem range_of_a_in_acute_triangle (a b c : ℝ) (A B C : ℝ) :
  0 < A ∧ A < π / 2 →
  0 < B ∧ B < π / 2 →
  0 < C ∧ C < π / 2 →
  a > 0 ∧ b > 0 ∧ c > 0 →
  b^2 - a^2 = a * c →
  c = 2 →
  2/3 < a ∧ a < 2 := by sorry

end range_of_a_in_acute_triangle_l2036_203632


namespace closest_integer_to_cube_root_l2036_203680

theorem closest_integer_to_cube_root : 
  ∃ (n : ℤ), n = 7 ∧ 
  ∀ (m : ℤ), |n - (5^3 + 7^3)^(1/3)| ≤ |m - (5^3 + 7^3)^(1/3)| := by
  sorry

end closest_integer_to_cube_root_l2036_203680


namespace solve_equation_l2036_203634

theorem solve_equation (x : ℝ) : (x / 5) + 3 = 4 → x = 5 := by
  sorry

end solve_equation_l2036_203634


namespace printing_machines_equation_l2036_203656

theorem printing_machines_equation (x : ℝ) : x > 0 → 
  (1000 / 15 : ℝ) + 1000 / x = 1000 / 5 ↔ 1 / 15 + 1 / x = 1 / 5 := by sorry

end printing_machines_equation_l2036_203656


namespace quadratic_factorization_l2036_203611

theorem quadratic_factorization (x : ℝ) : 2*x^2 - 4*x + 2 = 2*(x-1)^2 := by
  sorry

end quadratic_factorization_l2036_203611


namespace f_strictly_increasing_l2036_203618

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

-- Define our function f
noncomputable def f (x : ℝ) : ℝ :=
  (floor x : ℝ) + Real.sqrt (x - floor x)

-- State the theorem
theorem f_strictly_increasing :
  ∀ (x₁ x₂ : ℝ), x₁ < x₂ → f x₁ < f x₂ :=
by
  sorry

end f_strictly_increasing_l2036_203618


namespace volume_added_equals_expression_l2036_203600

/-- Represents a cylindrical tank lying on its side -/
structure CylindricalTank where
  radius : ℝ
  length : ℝ

/-- Calculates the volume of water added to the tank -/
def volumeAdded (tank : CylindricalTank) (initialDepth finalDepth : ℝ) : ℝ := sorry

/-- The main theorem to prove -/
theorem volume_added_equals_expression (tank : CylindricalTank) :
  tank.radius = 10 →
  tank.length = 30 →
  volumeAdded tank 5 (10 + 5 * Real.sqrt 2) = 1250 * Real.pi + 1500 + 750 * Real.sqrt 3 := by
  sorry

end volume_added_equals_expression_l2036_203600


namespace zero_discriminant_implies_geometric_progression_l2036_203685

-- Define the quadratic equation ax^2 - 6bx + 9c = 0
def quadratic_equation (a b c : ℝ) (x : ℝ) : Prop :=
  a * x^2 - 6 * b * x + 9 * c = 0

-- Define the discriminant of the quadratic equation
def discriminant (a b c : ℝ) : ℝ :=
  36 * b^2 - 36 * a * c

-- Define a geometric progression
def is_geometric_progression (a b c : ℝ) : Prop :=
  b^2 = a * c

-- Theorem statement
theorem zero_discriminant_implies_geometric_progression
  (a b c : ℝ) (h : discriminant a b c = 0) :
  is_geometric_progression a b c := by
sorry

end zero_discriminant_implies_geometric_progression_l2036_203685


namespace combined_savings_equal_separate_savings_l2036_203691

/-- Represents the store's window offer -/
structure WindowOffer where
  price : ℕ  -- Price per window
  buy : ℕ    -- Number of windows to buy
  free : ℕ   -- Number of free windows

/-- Calculates the cost for a given number of windows under the offer -/
def calculateCost (offer : WindowOffer) (windowsNeeded : ℕ) : ℕ :=
  let groups := windowsNeeded / (offer.buy + offer.free)
  let remainder := windowsNeeded % (offer.buy + offer.free)
  (groups * offer.buy + min remainder offer.buy) * offer.price

/-- Calculates the savings for a given number of windows under the offer -/
def calculateSavings (offer : WindowOffer) (windowsNeeded : ℕ) : ℕ :=
  windowsNeeded * offer.price - calculateCost offer windowsNeeded

theorem combined_savings_equal_separate_savings 
  (offer : WindowOffer)
  (davesWindows : ℕ)
  (dougsWindows : ℕ)
  (h1 : offer.price = 150)
  (h2 : offer.buy = 8)
  (h3 : offer.free = 2)
  (h4 : davesWindows = 10)
  (h5 : dougsWindows = 16) :
  calculateSavings offer (davesWindows + dougsWindows) = 
  calculateSavings offer davesWindows + calculateSavings offer dougsWindows :=
by sorry

end combined_savings_equal_separate_savings_l2036_203691


namespace percentage_of_pistachios_with_shells_l2036_203655

theorem percentage_of_pistachios_with_shells 
  (total_pistachios : ℕ)
  (opened_shell_ratio : ℚ)
  (opened_shell_count : ℕ)
  (h1 : total_pistachios = 80)
  (h2 : opened_shell_ratio = 3/4)
  (h3 : opened_shell_count = 57) :
  (↑opened_shell_count / (↑total_pistachios * opened_shell_ratio) : ℚ) = 95/100 :=
by sorry

end percentage_of_pistachios_with_shells_l2036_203655


namespace max_xy_under_constraint_l2036_203638

theorem max_xy_under_constraint (x y : ℕ+) (h : 27 * x.val + 35 * y.val ≤ 945) :
  x.val * y.val ≤ 234 :=
by sorry

end max_xy_under_constraint_l2036_203638


namespace files_remaining_after_deletion_l2036_203616

theorem files_remaining_after_deletion (music_files video_files deleted_files : ℕ) 
  (h1 : music_files = 16)
  (h2 : video_files = 48)
  (h3 : deleted_files = 30) :
  music_files + video_files - deleted_files = 34 := by
  sorry

end files_remaining_after_deletion_l2036_203616


namespace odd_power_sum_divisible_l2036_203652

theorem odd_power_sum_divisible (x y : ℤ) :
  ∀ n : ℕ, n > 0 → Odd n → (x^n + y^n) % (x + y) = 0 :=
by
  sorry

end odd_power_sum_divisible_l2036_203652


namespace geometric_sequence_a5_l2036_203615

/-- A geometric sequence with common ratio 2 and all positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ (∀ n, a (n + 1) = 2 * a n)

theorem geometric_sequence_a5 (a : ℕ → ℝ) (h : GeometricSequence a) (h_prod : a 3 * a 11 = 16) :
  a 5 = 1 := by
  sorry

end geometric_sequence_a5_l2036_203615


namespace cube_root_of_27_l2036_203687

theorem cube_root_of_27 (x : ℝ) (h : (Real.sqrt x) ^ 3 = 27) : x = 9 := by
  sorry

end cube_root_of_27_l2036_203687


namespace f_min_value_l2036_203678

/-- The function f as defined in the problem -/
def f (x y : ℝ) : ℝ := x^3 + y^3 + x^2*y + x*y^2 - 3*(x^2 + y^2 + x*y) + 3*(x + y)

/-- Theorem stating that f(x,y) ≥ 1 for all x,y ≥ 1/2 -/
theorem f_min_value (x y : ℝ) (hx : x ≥ 1/2) (hy : y ≥ 1/2) : f x y ≥ 1 := by
  sorry

end f_min_value_l2036_203678


namespace lemon_heads_distribution_l2036_203610

/-- Given 72 Lemon Heads distributed equally among 6 friends, prove that each friend receives 12 Lemon Heads. -/
theorem lemon_heads_distribution (total : ℕ) (friends : ℕ) (each : ℕ) 
  (h1 : total = 72) 
  (h2 : friends = 6) 
  (h3 : total = friends * each) : 
  each = 12 := by
  sorry

end lemon_heads_distribution_l2036_203610


namespace square_of_gcd_product_l2036_203667

theorem square_of_gcd_product (x y z : ℕ) (h : x > 0 ∧ y > 0 ∧ z > 0) 
  (eq : (1 : ℚ) / x - (1 : ℚ) / y = (1 : ℚ) / z) : 
  ∃ (k : ℕ), Nat.gcd x (Nat.gcd y z) * x * y * z = k ^ 2 := by
sorry

end square_of_gcd_product_l2036_203667


namespace parkway_elementary_girls_not_soccer_l2036_203657

theorem parkway_elementary_girls_not_soccer (total_students : ℕ) (boys : ℕ) (soccer_players : ℕ) 
  (h1 : total_students = 420)
  (h2 : boys = 296)
  (h3 : soccer_players = 250)
  (h4 : (86 : ℚ) / 100 * soccer_players = ↑(boys_playing_soccer))
  (boys_playing_soccer : ℕ) :
  total_students - soccer_players - (boys - boys_playing_soccer) = 89 := by
  sorry

#check parkway_elementary_girls_not_soccer

end parkway_elementary_girls_not_soccer_l2036_203657


namespace sum_of_x_equals_two_l2036_203664

/-- The function f(x) = |x+1| + |x-3| -/
def f (x : ℝ) : ℝ := |x + 1| + |x - 3|

/-- Theorem stating that if there exist two distinct real numbers x₁ and x₂ 
    such that f(x₁) = f(x₂) = 101, then their sum is 2 -/
theorem sum_of_x_equals_two (x₁ x₂ : ℝ) (h₁ : x₁ ≠ x₂) (h₂ : f x₁ = 101) (h₃ : f x₂ = 101) :
  x₁ + x₂ = 2 := by
  sorry

end sum_of_x_equals_two_l2036_203664


namespace distribute_negative_three_l2036_203673

theorem distribute_negative_three (a b : ℝ) : -3 * (a - b) = -3 * a + 3 * b := by
  sorry

end distribute_negative_three_l2036_203673


namespace production_time_reduction_l2036_203683

/-- Represents the time taken to complete a production order given a number of machines -/
def completion_time (num_machines : ℕ) (base_time : ℕ) : ℚ :=
  (num_machines * base_time : ℚ) / num_machines

theorem production_time_reduction :
  let base_machines := 3
  let base_time := 44
  let new_machines := 4
  (completion_time base_machines base_time - completion_time new_machines base_time : ℚ) = 11 := by
  sorry

end production_time_reduction_l2036_203683


namespace crow_votes_l2036_203695

/-- Represents the number of votes for each singer -/
structure Votes where
  rooster : ℕ
  crow : ℕ
  cuckoo : ℕ

/-- Represents the reported vote counts, which may be inaccurate -/
structure ReportedCounts where
  total : ℕ
  roosterCrow : ℕ
  crowCuckoo : ℕ
  cuckooRooster : ℕ

/-- Checks if a reported count is within the error margin of the actual count -/
def isWithinErrorMargin (reported actual : ℕ) : Prop :=
  (reported ≤ actual + 13) ∧ (actual ≤ reported + 13)

/-- The main theorem statement -/
theorem crow_votes (v : Votes) (r : ReportedCounts) : 
  (v.rooster + v.crow + v.cuckoo > 0) →
  isWithinErrorMargin r.total (v.rooster + v.crow + v.cuckoo) →
  isWithinErrorMargin r.roosterCrow (v.rooster + v.crow) →
  isWithinErrorMargin r.crowCuckoo (v.crow + v.cuckoo) →
  isWithinErrorMargin r.cuckooRooster (v.cuckoo + v.rooster) →
  r.total = 59 →
  r.roosterCrow = 15 →
  r.crowCuckoo = 18 →
  r.cuckooRooster = 20 →
  v.crow = 13 := by
  sorry

end crow_votes_l2036_203695


namespace triangle_problem_l2036_203699

theorem triangle_problem (A B C a b c p : ℝ) :
  -- Triangle ABC with angles A, B, C corresponding to sides a, b, c
  (0 < A ∧ A < π) → (0 < B ∧ B < π) → (0 < C ∧ C < π) →
  (A + B + C = π) →
  (a > 0) → (b > 0) → (c > 0) →
  -- Given conditions
  (Real.sin A + Real.sin C = p * Real.sin B) →
  (a * c = (1/4) * b^2) →
  -- Part I
  (p = 5/4 ∧ b = 1) →
  ((a = 1 ∧ c = 1/4) ∨ (a = 1/4 ∧ c = 1)) ∧
  -- Part II
  (0 < B ∧ B < π/2) →
  (Real.sqrt 6 / 2 < p ∧ p < Real.sqrt 2) :=
by sorry

end triangle_problem_l2036_203699


namespace two_cars_meeting_on_highway_l2036_203604

/-- Theorem: Two cars meeting on a highway --/
theorem two_cars_meeting_on_highway 
  (highway_length : ℝ) 
  (time : ℝ) 
  (speed_car2 : ℝ) 
  (h1 : highway_length = 105) 
  (h2 : time = 3) 
  (h3 : speed_car2 = 20) : 
  ∃ (speed_car1 : ℝ), 
    speed_car1 * time + speed_car2 * time = highway_length ∧ 
    speed_car1 = 15 := by
  sorry

end two_cars_meeting_on_highway_l2036_203604


namespace large_planks_nails_l2036_203607

/-- The number of nails needed for large planks in John's house wall construction -/
def nails_for_large_planks (total_nails : ℕ) (nails_for_small_planks : ℕ) : ℕ :=
  total_nails - nails_for_small_planks

/-- Theorem stating that the number of nails for large planks is 15 -/
theorem large_planks_nails :
  nails_for_large_planks 20 5 = 15 := by
  sorry

end large_planks_nails_l2036_203607


namespace max_sum_of_entries_l2036_203670

def numbers : List ℕ := [1, 2, 4, 5, 7, 8]

def is_valid_arrangement (a b c d e f : ℕ) : Prop :=
  a ∈ numbers ∧ b ∈ numbers ∧ c ∈ numbers ∧
  d ∈ numbers ∧ e ∈ numbers ∧ f ∈ numbers ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
  d ≠ e ∧ d ≠ f ∧
  e ≠ f

def sum_of_entries (a b c d e f : ℕ) : ℕ := (a + b + c) * (d + e + f)

theorem max_sum_of_entries :
  ∀ a b c d e f : ℕ,
    is_valid_arrangement a b c d e f →
    sum_of_entries a b c d e f ≤ 182 :=
by sorry

end max_sum_of_entries_l2036_203670


namespace noah_large_paintings_l2036_203660

/-- Represents the number of large paintings sold last month -/
def L : ℕ := sorry

/-- Price of a large painting -/
def large_price : ℕ := 60

/-- Price of a small painting -/
def small_price : ℕ := 30

/-- Number of small paintings sold last month -/
def small_paintings_last_month : ℕ := 4

/-- Total sales this month -/
def sales_this_month : ℕ := 1200

/-- Theorem stating that Noah sold 8 large paintings last month -/
theorem noah_large_paintings : L = 8 := by
  sorry

end noah_large_paintings_l2036_203660


namespace hyperbola_eccentricity_l2036_203689

/-- A hyperbola with foci on the x-axis -/
structure Hyperbola where
  /-- The slope of the asymptotes -/
  asymptote_slope : ℝ
  /-- Condition that the asymptote slope is positive -/
  asymptote_slope_pos : asymptote_slope > 0

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ :=
  sorry

/-- Theorem: The eccentricity of a hyperbola with asymptote slope √2/2 is √6/2 -/
theorem hyperbola_eccentricity (h : Hyperbola) 
    (h_asymptote : h.asymptote_slope = Real.sqrt 2 / 2) : 
    eccentricity h = Real.sqrt 6 / 2 := by
  sorry

end hyperbola_eccentricity_l2036_203689


namespace min_value_fraction_equality_condition_l2036_203636

theorem min_value_fraction (x : ℝ) (h : x > 8) : x^2 / (x - 8) ≥ 32 :=
by sorry

theorem equality_condition (x : ℝ) (h : x > 8) : 
  x^2 / (x - 8) = 32 ↔ x = 16 :=
by sorry

end min_value_fraction_equality_condition_l2036_203636


namespace nested_radical_simplification_l2036_203608

theorem nested_radical_simplification (a b m n : ℝ) 
  (ha : a > 0) (hb : b > 0) (hpos : a + 2 * Real.sqrt b > 0)
  (hm : m > 0) (hn : n > 0) (hmn_sum : m + n = a) (hmn_prod : m * n = b) :
  Real.sqrt (a + 2 * Real.sqrt b) = Real.sqrt m + Real.sqrt n ∧
  Real.sqrt (a - 2 * Real.sqrt b) = |Real.sqrt m - Real.sqrt n| :=
by sorry

end nested_radical_simplification_l2036_203608


namespace last_two_digits_of_nine_to_h_l2036_203603

def a : ℕ := 1
def b : ℕ := 2^a
def c : ℕ := 3^b
def d : ℕ := 4^c
def e : ℕ := 5^d
def f : ℕ := 6^e
def g : ℕ := 7^f
def h : ℕ := 8^g

theorem last_two_digits_of_nine_to_h (a b c d e f g h : ℕ) 
  (ha : a = 1)
  (hb : b = 2^a)
  (hc : c = 3^b)
  (hd : d = 4^c)
  (he : e = 5^d)
  (hf : f = 6^e)
  (hg : g = 7^f)
  (hh : h = 8^g) :
  9^h % 100 = 21 := by
  sorry

end last_two_digits_of_nine_to_h_l2036_203603


namespace triangle_similarity_problem_l2036_203622

-- Define the triangles and their properties
structure Triangle :=
  (side1 : ℝ)
  (side2 : ℝ)
  (height : ℝ)

-- Define the similarity relation
def similar (t1 t2 : Triangle) : Prop := sorry

-- State the theorem
theorem triangle_similarity_problem 
  (FGH IJH : Triangle)
  (h_similar : similar FGH IJH)
  (h_GH : FGH.side1 = 25)
  (h_JH : IJH.side1 = 15)
  (h_height : FGH.height = 15) :
  IJH.side2 = 9 := by sorry

end triangle_similarity_problem_l2036_203622


namespace shooter_probability_l2036_203698

theorem shooter_probability (p10 p9 p8 : ℝ) (h1 : p10 = 0.24) (h2 : p9 = 0.28) (h3 : p8 = 0.19) :
  1 - p10 - p9 = 0.48 := by
  sorry

end shooter_probability_l2036_203698
