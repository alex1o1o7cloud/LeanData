import Mathlib

namespace line_through_center_line_bisecting_chord_l2697_269793

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 16

-- Define point P
def point_P : ℝ × ℝ := (2, 2)

-- Define the equation of line l passing through P and center of C
def line_l_through_center (x y : ℝ) : Prop := 2*x - y - 2 = 0

-- Define the equation of line l bisecting chord AB
def line_l_bisecting_chord (x y : ℝ) : Prop := x + 2*y - 6 = 0

-- Theorem 1: Line l passing through P and center of C
theorem line_through_center : 
  ∀ x y : ℝ, circle_C x y → line_l_through_center x y → 
  ∃ t : ℝ, x = 2 + t ∧ y = 2 + 2*t :=
sorry

-- Theorem 2: Line l passing through P and bisecting chord AB
theorem line_bisecting_chord :
  ∀ x y : ℝ, circle_C x y → line_l_bisecting_chord x y →
  ∃ t : ℝ, x = 2 + t ∧ y = 2 - t/2 :=
sorry

end line_through_center_line_bisecting_chord_l2697_269793


namespace division_multiplication_negatives_l2697_269788

theorem division_multiplication_negatives : (-100) / (-25) * (-6) = -24 := by
  sorry

end division_multiplication_negatives_l2697_269788


namespace inequality_implies_a_range_l2697_269799

open Real

theorem inequality_implies_a_range (a : ℝ) : 
  (∀ θ : ℝ, 0 ≤ θ ∧ θ < π / 2 → 
    sqrt 2 * (2 * a + 3) * cos (θ - π / 4) + 6 / (sin θ + cos θ) - 2 * sin (2 * θ) < 3 * a + 6) →
  a > 3 := by
sorry

end inequality_implies_a_range_l2697_269799


namespace smallest_positive_multiple_of_32_l2697_269746

theorem smallest_positive_multiple_of_32 :
  ∀ n : ℕ, n > 0 → 32 * 1 ≤ 32 * n :=
by
  sorry

end smallest_positive_multiple_of_32_l2697_269746


namespace parking_spaces_remaining_l2697_269770

theorem parking_spaces_remaining (total_spaces : ℕ) (caravan_spaces : ℕ) (parked_caravans : ℕ) : 
  total_spaces = 30 → caravan_spaces = 2 → parked_caravans = 3 → 
  total_spaces - (caravan_spaces * parked_caravans) = 24 := by
  sorry

end parking_spaces_remaining_l2697_269770


namespace shortest_altitude_of_triangle_l2697_269728

/-- Given a triangle with sides 9, 12, and 15, the shortest altitude has length 7.2 -/
theorem shortest_altitude_of_triangle (a b c h : ℝ) : 
  a = 9 → b = 12 → c = 15 →
  (a^2 + b^2 = c^2) →
  (h * c = 2 * (a * b / 2)) →
  h = 7.2 := by sorry

end shortest_altitude_of_triangle_l2697_269728


namespace total_earnings_is_5800_l2697_269734

/-- Represents the investment and return information for three investors -/
structure InvestmentInfo where
  investment_ratio : Fin 3 → ℕ
  return_ratio : Fin 3 → ℕ
  earnings_difference : ℕ

/-- Calculates the total earnings based on the given investment information -/
def calculate_total_earnings (info : InvestmentInfo) : ℕ :=
  sorry

/-- Theorem stating that the total earnings are 5800 given the specified conditions -/
theorem total_earnings_is_5800 (info : InvestmentInfo) 
  (h1 : info.investment_ratio = ![3, 4, 5])
  (h2 : info.return_ratio = ![6, 5, 4])
  (h3 : info.earnings_difference = 200) :
  calculate_total_earnings info = 5800 :=
sorry

end total_earnings_is_5800_l2697_269734


namespace alvez_family_has_three_children_l2697_269724

/-- Represents the Alvez family structure -/
structure AlvezFamily where
  num_children : ℕ
  mother_age : ℕ
  children_ages : Fin num_children → ℕ

/-- The average age of a list of ages -/
def average_age (ages : List ℕ) : ℚ :=
  (ages.sum : ℚ) / ages.length

/-- The Alvez family satisfies the given conditions -/
def satisfies_conditions (family : AlvezFamily) : Prop :=
  let total_members := family.num_children + 2
  let all_ages := family.mother_age :: 50 :: (List.ofFn family.children_ages)
  average_age all_ages = 22 ∧
  average_age (family.mother_age :: (List.ofFn family.children_ages)) = 15

/-- The main theorem: There are exactly 3 children in the Alvez family -/
theorem alvez_family_has_three_children :
  ∃ (family : AlvezFamily), satisfies_conditions family ∧ family.num_children = 3 :=
sorry

end alvez_family_has_three_children_l2697_269724


namespace zander_sand_lorries_l2697_269777

/-- Represents the construction materials purchase scenario --/
structure ConstructionPurchase where
  total_payment : ℕ
  cement_bags : ℕ
  cement_price_per_bag : ℕ
  sand_tons_per_lorry : ℕ
  sand_price_per_ton : ℕ

/-- Calculates the number of lorries of sand purchased --/
def sand_lorries (purchase : ConstructionPurchase) : ℕ :=
  let cement_cost := purchase.cement_bags * purchase.cement_price_per_bag
  let sand_cost := purchase.total_payment - cement_cost
  let sand_price_per_lorry := purchase.sand_tons_per_lorry * purchase.sand_price_per_ton
  sand_cost / sand_price_per_lorry

/-- Theorem stating that for the given purchase scenario, the number of sand lorries is 20 --/
theorem zander_sand_lorries :
  let purchase := ConstructionPurchase.mk 13000 500 10 10 40
  sand_lorries purchase = 20 := by
  sorry

end zander_sand_lorries_l2697_269777


namespace older_friend_age_l2697_269708

theorem older_friend_age (A B C : ℝ) 
  (h1 : A - B = 2.5)
  (h2 : A - C = 3.75)
  (h3 : A + B + C = 110.5)
  (h4 : B = 2 * C) :
  A = 104.25 := by
  sorry

end older_friend_age_l2697_269708


namespace inserted_numbers_sum_l2697_269767

theorem inserted_numbers_sum (a b : ℝ) : 
  (∃ d : ℝ, a = 4 + d ∧ b = 4 + 2*d) →  -- arithmetic progression condition
  (∃ r : ℝ, b = a*r ∧ 16 = b*r) →       -- geometric progression condition
  a + b = 6*Real.sqrt 3 + 8 := by
sorry

end inserted_numbers_sum_l2697_269767


namespace egyptian_fraction_decomposition_l2697_269717

theorem egyptian_fraction_decomposition (n : ℕ) (h : n ≥ 5 ∧ Odd n) :
  (2 : ℚ) / 11 = 1 / 6 + 1 / 66 ∧
  (2 : ℚ) / n = 1 / ((n + 1) / 2) + 1 / (n * (n + 1) / 2) :=
by sorry

end egyptian_fraction_decomposition_l2697_269717


namespace min_t_for_inequality_l2697_269795

theorem min_t_for_inequality (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  (∀ x y, x ≥ 0 → y ≥ 0 → Real.sqrt (x * y) ≤ (1 / (2 * Real.sqrt 6)) * (2 * x + 3 * y)) ∧
  (∀ ε > 0, ∃ x y, x ≥ 0 ∧ y ≥ 0 ∧ Real.sqrt (x * y) > (1 / (2 * Real.sqrt 6) - ε) * (2 * x + 3 * y)) :=
sorry

end min_t_for_inequality_l2697_269795


namespace f_simplification_inverse_sum_value_l2697_269776

noncomputable def f (α : Real) : Real :=
  (Real.sin (2 * Real.pi - α) * Real.cos (Real.pi + α) * Real.cos (Real.pi / 2 - α) * Real.cos (11 * Real.pi / 2 - α)) /
  (Real.sin (3 * Real.pi - α) * Real.cos (Real.pi / 2 + α) * Real.sin (9 * Real.pi / 2 + α)) +
  Real.cos (2 * Real.pi - α)

theorem f_simplification (α : Real) : f α = Real.sin α + Real.cos α := by sorry

theorem inverse_sum_value (α : Real) (h : f α = Real.sqrt 10 / 5) :
  1 / Real.sin α + 1 / Real.cos α = -4 * Real.sqrt 10 / 3 := by sorry

end f_simplification_inverse_sum_value_l2697_269776


namespace father_son_age_sum_l2697_269764

/-- Represents the ages of a father and son pair -/
structure FatherSonAges where
  father : ℕ
  son : ℕ

/-- The sum of father's and son's ages after a given number of years -/
def ageSum (ages : FatherSonAges) (years : ℕ) : ℕ :=
  ages.father + ages.son + 2 * years

theorem father_son_age_sum :
  ∀ (ages : FatherSonAges),
    ages.father + ages.son = 55 →
    ages.father = 37 →
    ages.son = 18 →
    ageSum ages (ages.father - ages.son) = 93 :=
by
  sorry

end father_son_age_sum_l2697_269764


namespace mikey_has_125_jelly_beans_l2697_269738

/-- The number of jelly beans each person has -/
structure JellyBeans where
  napoleon : ℕ
  sedrich : ℕ
  daphne : ℕ
  alondra : ℕ
  mikey : ℕ

/-- The conditions of the jelly bean problem -/
def jelly_bean_conditions (jb : JellyBeans) : Prop :=
  jb.napoleon = 56 ∧
  jb.sedrich = 3 * jb.napoleon + 9 ∧
  jb.daphne = 2 * (jb.sedrich - jb.napoleon) ∧
  jb.alondra = (jb.napoleon + jb.sedrich + jb.daphne) / 3 - 8 ∧
  jb.napoleon + jb.sedrich + jb.daphne + jb.alondra = 5 * jb.mikey

/-- The theorem stating that under the given conditions, Mikey has 125 jelly beans -/
theorem mikey_has_125_jelly_beans (jb : JellyBeans) 
  (h : jelly_bean_conditions jb) : jb.mikey = 125 := by
  sorry


end mikey_has_125_jelly_beans_l2697_269738


namespace min_lines_theorem_l2697_269781

/-- A plane -/
structure Plane where

/-- A point in a plane -/
structure Point (α : Plane) where

/-- A line in a plane -/
structure Line (α : Plane) where

/-- A ray in a plane -/
structure Ray (α : Plane) where

/-- Predicate for a line not passing through a point -/
def LineNotThroughPoint (α : Plane) (l : Line α) (P : Point α) : Prop :=
  sorry

/-- Predicate for a ray intersecting a line -/
def RayIntersectsLine (α : Plane) (r : Ray α) (l : Line α) : Prop :=
  sorry

/-- The minimum number of lines theorem -/
theorem min_lines_theorem (α : Plane) (P : Point α) (k : ℕ) (h : k > 0) :
  ∃ (n : ℕ),
    (∀ (m : ℕ),
      (∃ (lines : Fin m → Line α),
        (∀ i, LineNotThroughPoint α (lines i) P) ∧
        (∀ r : Ray α, ∃ (S : Finset (Fin m)), S.card ≥ k ∧ ∀ i ∈ S, RayIntersectsLine α r (lines i)))
      → m ≥ n) ∧
    (∃ (lines : Fin (2 * k + 1) → Line α),
      (∀ i, LineNotThroughPoint α (lines i) P) ∧
      (∀ r : Ray α, ∃ (S : Finset (Fin (2 * k + 1))), S.card ≥ k ∧ ∀ i ∈ S, RayIntersectsLine α r (lines i))) :=
  sorry

end min_lines_theorem_l2697_269781


namespace no_integer_solutions_l2697_269783

theorem no_integer_solutions :
  ¬∃ (x y : ℤ), x^3 + 4*x^2 - 11*x + 30 = 8*y^3 + 24*y^2 + 18*y + 7 := by
  sorry

end no_integer_solutions_l2697_269783


namespace inequality_solution_set_l2697_269768

theorem inequality_solution_set (x : ℝ) :
  x * (2 * x^2 - 3 * x + 1) ≤ 0 ↔ x ≤ 0 ∨ (1/2 ≤ x ∧ x ≤ 1) :=
by sorry

end inequality_solution_set_l2697_269768


namespace max_parts_properties_l2697_269769

/-- The maximum number of parts that can be produced from n blanks -/
def max_parts (n : ℕ) : ℕ :=
  let rec aux (blanks remaining : ℕ) : ℕ :=
    if remaining = 0 then blanks
    else
      let new_blanks := remaining / 3
      aux (blanks + remaining) new_blanks
  aux 0 n

theorem max_parts_properties :
  (max_parts 9 = 13) ∧
  (max_parts 14 = 20) ∧
  (max_parts 27 = 40 ∧ ∀ m < 27, max_parts m < 40) := by
  sorry

end max_parts_properties_l2697_269769


namespace cubic_equation_roots_l2697_269785

theorem cubic_equation_roots (x : ℝ) : 
  let r1 := 2 * Real.sin (2 * Real.pi / 9)
  let r2 := 2 * Real.sin (8 * Real.pi / 9)
  let r3 := 2 * Real.sin (14 * Real.pi / 9)
  (x - r1) * (x - r2) * (x - r3) = x^3 - 3*x + Real.sqrt 3 :=
by sorry

end cubic_equation_roots_l2697_269785


namespace sqrt_18_minus_sqrt_8_equals_sqrt_2_l2697_269759

theorem sqrt_18_minus_sqrt_8_equals_sqrt_2 : Real.sqrt 18 - Real.sqrt 8 = Real.sqrt 2 := by
  sorry

end sqrt_18_minus_sqrt_8_equals_sqrt_2_l2697_269759


namespace sum_of_coefficients_l2697_269740

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x : ℝ, (1 - 2*x)^7 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = -1 :=
by
  sorry

end sum_of_coefficients_l2697_269740


namespace agri_products_theorem_l2697_269722

/-- Represents the prices and quantities of agricultural products A and B --/
structure AgriProducts where
  price_A : ℝ
  price_B : ℝ
  quantity_A : ℝ
  quantity_B : ℝ

/-- Represents the problem constraints and conditions --/
def problem_constraints (p : AgriProducts) : Prop :=
  2 * p.price_A + 3 * p.price_B = 690 ∧
  p.price_A + 4 * p.price_B = 720 ∧
  p.quantity_A + p.quantity_B = 40 ∧
  p.price_A * p.quantity_A + p.price_B * p.quantity_B ≤ 5400 ∧
  p.quantity_A ≤ 3 * p.quantity_B

/-- Calculates the profit given the prices and quantities --/
def profit (p : AgriProducts) : ℝ :=
  (160 - p.price_A) * p.quantity_A + (200 - p.price_B) * p.quantity_B

/-- The main theorem to prove --/
theorem agri_products_theorem (p : AgriProducts) :
  problem_constraints p →
  p.price_A = 120 ∧ p.price_B = 150 ∧
  ∀ q : AgriProducts, problem_constraints q →
    profit q ≤ profit { price_A := 120, price_B := 150, quantity_A := 20, quantity_B := 20 } :=
by sorry

end agri_products_theorem_l2697_269722


namespace sum_squared_expression_lower_bound_l2697_269765

theorem sum_squared_expression_lower_bound 
  (x y z : ℝ) 
  (hx : x ≠ 0) 
  (hy : y ≠ 0) 
  (hz : z ≠ 0) 
  (h_sum : x + y + z = x * y * z) : 
  ((x^2 - 1) / x)^2 + ((y^2 - 1) / y)^2 + ((z^2 - 1) / z)^2 ≥ 4 := by
  sorry

end sum_squared_expression_lower_bound_l2697_269765


namespace problem_statement_l2697_269720

/-- Given that a² + ab = -2 and b² - 3ab = -3, prove that a² + 4ab - b² = 1 -/
theorem problem_statement (a b : ℝ) (h1 : a^2 + a*b = -2) (h2 : b^2 - 3*a*b = -3) :
  a^2 + 4*a*b - b^2 = 1 := by
  sorry

end problem_statement_l2697_269720


namespace sum_greater_than_product_l2697_269760

theorem sum_greater_than_product (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h : Real.arctan x + Real.arctan y + Real.arctan z < π) : 
  x + y + z > x * y * z := by
  sorry

end sum_greater_than_product_l2697_269760


namespace sandbox_capacity_increase_l2697_269733

theorem sandbox_capacity_increase (l w h : ℝ) : 
  l * w * h = 10 → (2 * l) * (2 * w) * (2 * h) = 80 := by
  sorry

end sandbox_capacity_increase_l2697_269733


namespace service_period_problem_l2697_269707

/-- Represents the problem of determining the agreed-upon period of service --/
theorem service_period_problem (total_pay : ℕ) (uniform_price : ℕ) (partial_service : ℕ) (partial_pay : ℕ) :
  let full_compensation := total_pay + uniform_price
  let partial_compensation := partial_pay + uniform_price
  (partial_service : ℚ) / (12 : ℚ) = partial_compensation / full_compensation →
  12 = (partial_service * full_compensation) / partial_compensation :=
by
  sorry

#check service_period_problem 900 100 9 650

end service_period_problem_l2697_269707


namespace opposite_of_negative_third_l2697_269750

theorem opposite_of_negative_third : 
  (fun x : ℚ => -x) (-1/3) = 1/3 := by sorry

end opposite_of_negative_third_l2697_269750


namespace gcd_228_2010_l2697_269747

theorem gcd_228_2010 : Nat.gcd 228 2010 = 6 := by
  sorry

end gcd_228_2010_l2697_269747


namespace events_mutually_exclusive_not_opposite_l2697_269714

-- Define the set of people
inductive Person : Type
| A : Person
| B : Person
| C : Person
| D : Person

-- Define the set of cards
inductive Card : Type
| Red : Card
| Black : Card
| Blue : Card
| White : Card

-- Define a distribution as a function from Person to Card
def Distribution := Person → Card

-- Define the event "Person A gets the red card"
def event_A_red (d : Distribution) : Prop := d Person.A = Card.Red

-- Define the event "Person B gets the red card"
def event_B_red (d : Distribution) : Prop := d Person.B = Card.Red

-- State the theorem
theorem events_mutually_exclusive_not_opposite :
  ∃ (d : Distribution),
    (∀ (p : Person), ∃! (c : Card), d p = c) ∧  -- Each person gets exactly one card
    (∀ (c : Card), ∃! (p : Person), d p = c) ∧  -- Each card is given to exactly one person
    (¬(event_A_red d ∧ event_B_red d)) ∧        -- Events are mutually exclusive
    ¬(event_A_red d ↔ ¬event_B_red d)           -- Events are not opposite
  := by sorry

end events_mutually_exclusive_not_opposite_l2697_269714


namespace min_distance_to_line_l2697_269786

/-- The minimum distance from the origin to the line x + y - 4 = 0 is 2√2 -/
theorem min_distance_to_line : 
  let line := {(x, y) : ℝ × ℝ | x + y = 4}
  ∃ d : ℝ, d = 2 * Real.sqrt 2 ∧ 
    ∀ P ∈ line, Real.sqrt (P.1^2 + P.2^2) ≥ d ∧
    ∃ Q ∈ line, Real.sqrt (Q.1^2 + Q.2^2) = d :=
by sorry

end min_distance_to_line_l2697_269786


namespace shortest_tangent_length_l2697_269752

-- Define the circles C₁ and C₂
def C₁ (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 16
def C₂ (x y : ℝ) : Prop := (x + 12)^2 + y^2 = 225

-- Define the shortest tangent line segment
def shortest_tangent (R S : ℝ × ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    R = (x₁, y₁) ∧ S = (x₂, y₂) ∧
    C₁ x₁ y₁ ∧ C₂ x₂ y₂ ∧
    ∀ (T U : ℝ × ℝ),
      C₁ T.1 T.2 → C₂ U.1 U.2 →
      Real.sqrt ((T.1 - U.1)^2 + (T.2 - U.2)^2) ≥ 
      Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

-- Theorem statement
theorem shortest_tangent_length :
  ∀ (R S : ℝ × ℝ),
    shortest_tangent R S →
    Real.sqrt ((R.1 - S.1)^2 + (R.2 - S.2)^2) = 
    Real.sqrt (16 - (60/19)^2) + Real.sqrt (225 - (225/19)^2) := by
  sorry

end shortest_tangent_length_l2697_269752


namespace investment_interest_rate_proof_l2697_269774

/-- Proves that for an investment of 7000 over 2 years, if the interest earned is 840 more than
    what would be earned at 12% p.a., then the interest rate is 18% p.a. -/
theorem investment_interest_rate_proof 
  (principal : ℝ) 
  (time : ℝ) 
  (interest_diff : ℝ) 
  (base_rate : ℝ) 
  (h1 : principal = 7000)
  (h2 : time = 2)
  (h3 : interest_diff = 840)
  (h4 : base_rate = 12)
  (h5 : principal * (rate / 100) * time - principal * (base_rate / 100) * time = interest_diff) :
  rate = 18 := by
  sorry

#check investment_interest_rate_proof

end investment_interest_rate_proof_l2697_269774


namespace complex_number_product_l2697_269711

theorem complex_number_product (a b c d : ℂ) : 
  (a + b + c + d = 5) →
  ((5 - a)^4 + (5 - b)^4 + (5 - c)^4 + (5 - d)^4 = 125) →
  ((a + b)^4 + (b + c)^4 + (c + d)^4 + (d + a)^4 + (a + c)^4 + (b + d)^4 = 1205) →
  (a^4 + b^4 + c^4 + d^4 = 25) →
  a * b * c * d = 70 := by
sorry

end complex_number_product_l2697_269711


namespace vexel_language_words_l2697_269736

def alphabet_size : ℕ := 26
def max_word_length : ℕ := 5

def words_with_z (n : ℕ) : ℕ :=
  alphabet_size^n - (alphabet_size - 1)^n

def total_words : ℕ :=
  (words_with_z 1) + (words_with_z 2) + (words_with_z 3) + (words_with_z 4) + (words_with_z 5)

theorem vexel_language_words :
  total_words = 2205115 :=
by sorry

end vexel_language_words_l2697_269736


namespace perpendicular_line_to_cosine_tangent_l2697_269756

open Real

/-- The equation of a line perpendicular to the tangent of y = cos x at (π/3, 1/2) --/
theorem perpendicular_line_to_cosine_tangent :
  let f : ℝ → ℝ := fun x ↦ cos x
  let p : ℝ × ℝ := (π / 3, 1 / 2)
  let tangent_slope : ℝ := -sin (π / 3)
  let perpendicular_slope : ℝ := -1 / tangent_slope
  let line_equation : ℝ → ℝ → ℝ := fun x y ↦ 2 * x - sqrt 3 * y - 2 * π / 3 + sqrt 3 / 2
  (f (π / 3) = 1 / 2) →
  (perpendicular_slope = 2 / sqrt 3) →
  (∀ x y, line_equation x y = 0 ↔ y - p.2 = perpendicular_slope * (x - p.1)) :=
by sorry

end perpendicular_line_to_cosine_tangent_l2697_269756


namespace train_speed_calculation_l2697_269792

/-- Theorem: Train Speed Calculation
Given two trains starting from the same station, traveling along parallel tracks in the same direction,
with one train traveling at 35 mph, and the distance between them after 10 hours being 250 miles,
the speed of the first train is 60 mph. -/
theorem train_speed_calculation (speed_second_train : ℝ) (time : ℝ) (distance : ℝ) :
  speed_second_train = 35 →
  time = 10 →
  distance = 250 →
  ∃ (speed_first_train : ℝ),
    speed_first_train > 0 ∧
    distance = (speed_first_train - speed_second_train) * time ∧
    speed_first_train = 60 :=
by sorry

end train_speed_calculation_l2697_269792


namespace fish_in_tank_l2697_269775

theorem fish_in_tank (total : ℕ) (blue : ℕ) (spotted : ℕ) : 
  3 * blue = total →
  2 * spotted = blue →
  spotted = 5 →
  total = 30 := by
sorry

end fish_in_tank_l2697_269775


namespace P_divisible_by_Q_l2697_269715

variable (X : ℝ)
variable (n : ℕ)

def P (n : ℕ) (X : ℝ) : ℝ := n * X^(n+2) - (n+2) * X^(n+1) + (n+2) * X - n

def Q (X : ℝ) : ℝ := (X - 1)^3

theorem P_divisible_by_Q (n : ℕ) (h : n > 0) :
  ∃ k : ℝ, P n X = k * Q X := by
  sorry

end P_divisible_by_Q_l2697_269715


namespace network_connections_l2697_269780

theorem network_connections (n : ℕ) (k : ℕ) (h1 : n = 20) (h2 : k = 3) :
  (n * k) / 2 = 30 :=
by sorry

end network_connections_l2697_269780


namespace money_redistribution_l2697_269753

theorem money_redistribution (younger_money : ℝ) :
  let elder_money := 1.25 * younger_money
  let total_money := younger_money + elder_money
  let equal_share := total_money / 2
  let transfer_amount := equal_share - younger_money
  (transfer_amount / elder_money) = 0.1 := by
sorry

end money_redistribution_l2697_269753


namespace assignment_plans_count_l2697_269702

/-- The number of students --/
def total_students : ℕ := 6

/-- The number of tasks --/
def total_tasks : ℕ := 4

/-- The number of students to be selected --/
def selected_students : ℕ := 4

/-- The number of students who cannot be assigned to a specific task --/
def restricted_students : ℕ := 2

/-- Calculates the total number of different assignment plans --/
def total_assignment_plans : ℕ := 
  (total_students.factorial / (total_students - selected_students).factorial) - 
  2 * ((total_students - 1).factorial / (total_students - selected_students).factorial)

/-- Theorem stating the total number of different assignment plans --/
theorem assignment_plans_count : total_assignment_plans = 240 := by
  sorry

end assignment_plans_count_l2697_269702


namespace rhombus_in_rectangle_perimeter_l2697_269732

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a quadrilateral -/
structure Quadrilateral :=
  (A : Point)
  (B : Point)
  (C : Point)
  (D : Point)

/-- Checks if a quadrilateral is a rhombus -/
def is_rhombus (q : Quadrilateral) : Prop := sorry

/-- Checks if a quadrilateral is a rectangle -/
def is_rectangle (q : Quadrilateral) : Prop := sorry

/-- Checks if a point is on a line segment -/
def is_on_segment (p : Point) (a : Point) (b : Point) : Prop := sorry

/-- Calculates the distance between two points -/
def distance (p1 : Point) (p2 : Point) : ℝ := sorry

/-- Calculates the perimeter of a quadrilateral -/
def perimeter (q : Quadrilateral) : ℝ := sorry

theorem rhombus_in_rectangle_perimeter 
  (W X Y Z : Point) 
  (A B C D : Point) :
  let rect := Quadrilateral.mk W X Y Z
  let rhom := Quadrilateral.mk A B C D
  is_rectangle rect →
  is_rhombus rhom →
  is_on_segment A W X →
  is_on_segment B X Y →
  is_on_segment C Y Z →
  is_on_segment D Z W →
  distance W A = 12 →
  distance X B = 9 →
  distance B D = 15 →
  distance A C = distance X Y →
  perimeter rect = 66 := by sorry

end rhombus_in_rectangle_perimeter_l2697_269732


namespace x_divisibility_l2697_269703

def x : ℕ := 48 + 64 + 192 + 256 + 384 + 768 + 1024

theorem x_divisibility :
  (∃ k : ℕ, x = 4 * k) ∧
  (∃ k : ℕ, x = 16 * k) ∧
  ¬(∀ k : ℕ, x = 64 * k) ∧
  ¬(∀ k : ℕ, x = 128 * k) := by
  sorry

end x_divisibility_l2697_269703


namespace tom_fruit_purchase_total_l2697_269743

/-- Calculate the total amount Tom paid for fruits with discount and tax --/
theorem tom_fruit_purchase_total : 
  let apple_cost : ℝ := 8 * 70
  let mango_cost : ℝ := 9 * 90
  let grape_cost : ℝ := 5 * 150
  let total_before_discount : ℝ := apple_cost + mango_cost + grape_cost
  let discount_rate : ℝ := 0.10
  let tax_rate : ℝ := 0.05
  let discounted_amount : ℝ := total_before_discount * (1 - discount_rate)
  let final_amount : ℝ := discounted_amount * (1 + tax_rate)
  final_amount = 2003.4 := by sorry

end tom_fruit_purchase_total_l2697_269743


namespace dads_first_half_speed_is_28_l2697_269742

/-- The speed of Jake's dad during the first half of the journey to the water park -/
def dads_first_half_speed : ℝ := by sorry

/-- The total journey time for Jake's dad in hours -/
def total_journey_time : ℝ := 0.5

/-- Jake's biking speed in miles per hour -/
def jake_bike_speed : ℝ := 11

/-- Time it takes Jake to bike to the water park in hours -/
def jake_bike_time : ℝ := 2

/-- Jake's dad's speed during the second half of the journey in miles per hour -/
def dads_second_half_speed : ℝ := 60

theorem dads_first_half_speed_is_28 :
  dads_first_half_speed = 28 := by sorry

end dads_first_half_speed_is_28_l2697_269742


namespace algae_coverage_day21_l2697_269798

-- Define the algae coverage function
def algaeCoverage (day : ℕ) : ℚ :=
  1 / 2^(24 - day)

-- State the theorem
theorem algae_coverage_day21 :
  algaeCoverage 24 = 1 ∧ (∀ d : ℕ, algaeCoverage (d + 1) = 2 * algaeCoverage d) →
  algaeCoverage 21 = 1/8 :=
by
  sorry

end algae_coverage_day21_l2697_269798


namespace smallest_divisible_by_1_to_12_l2697_269790

theorem smallest_divisible_by_1_to_12 : ∃ n : ℕ, n > 0 ∧ (∀ k : ℕ, 1 ≤ k ∧ k ≤ 12 → k ∣ n) ∧ (∀ m : ℕ, m > 0 ∧ (∀ k : ℕ, 1 ≤ k ∧ k ≤ 12 → k ∣ m) → m ≥ 27720) :=
by sorry

end smallest_divisible_by_1_to_12_l2697_269790


namespace some_number_value_l2697_269758

theorem some_number_value (a n : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 25 * n * 49) : n = 5 := by
  sorry

end some_number_value_l2697_269758


namespace count_triangles_in_dodecagon_l2697_269797

/-- The number of triangles that can be formed from the vertices of a dodecagon -/
def triangles_in_dodecagon : ℕ := 220

/-- The number of vertices in a dodecagon -/
def dodecagon_vertices : ℕ := 12

/-- The number of vertices required to form a triangle -/
def triangle_vertices : ℕ := 3

/-- Theorem: The number of triangles that can be formed by selecting 3 vertices
    from a 12-vertex polygon is equal to 220 -/
theorem count_triangles_in_dodecagon :
  Nat.choose dodecagon_vertices triangle_vertices = triangles_in_dodecagon := by
  sorry

end count_triangles_in_dodecagon_l2697_269797


namespace five_people_handshakes_l2697_269791

/-- The number of handshakes when n people meet, where each pair shakes hands exactly once -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: When 5 people meet, they shake hands a total of 10 times -/
theorem five_people_handshakes : handshakes 5 = 10 := by
  sorry

end five_people_handshakes_l2697_269791


namespace cos_double_angle_special_case_l2697_269704

theorem cos_double_angle_special_case (θ : Real) 
  (h : Real.sin (Real.pi / 2 + θ) = 3 / 5) : 
  Real.cos (2 * θ) = -7 / 25 := by
  sorry

end cos_double_angle_special_case_l2697_269704


namespace correct_emu_count_l2697_269794

/-- The number of emus in Farmer Brown's flock -/
def num_emus : ℕ := 20

/-- The number of heads per emu -/
def heads_per_emu : ℕ := 1

/-- The number of legs per emu -/
def legs_per_emu : ℕ := 2

/-- The total count of heads and legs in the flock -/
def total_count : ℕ := 60

/-- Theorem stating that the number of emus is correct given the conditions -/
theorem correct_emu_count : 
  num_emus * (heads_per_emu + legs_per_emu) = total_count :=
by sorry

end correct_emu_count_l2697_269794


namespace parabola_equation_l2697_269737

-- Define the parabola type
structure Parabola where
  -- The equation of the parabola is either y² = ax or x² = by
  a : ℝ
  b : ℝ
  along_x_axis : Bool

-- Define the properties of the parabola
def satisfies_conditions (p : Parabola) : Prop :=
  -- Vertex at origin (implied by the standard form of equation)
  -- Axis of symmetry along one of the coordinate axes (implied by the structure)
  -- Passes through the point (-2, 3)
  (p.along_x_axis ∧ 3^2 = -p.a * (-2)) ∨
  (¬p.along_x_axis ∧ (-2)^2 = p.b * 3)

-- Theorem statement
theorem parabola_equation :
  ∀ p : Parabola, satisfies_conditions p →
    (p.along_x_axis ∧ p.a = -9/2) ∨ (¬p.along_x_axis ∧ p.b = 4/3) :=
by sorry

end parabola_equation_l2697_269737


namespace father_chips_amount_l2697_269721

theorem father_chips_amount (son_chips brother_chips total_chips : ℕ) 
  (h1 : son_chips = 350)
  (h2 : brother_chips = 182)
  (h3 : total_chips = 800) :
  total_chips - (son_chips + brother_chips) = 268 := by
  sorry

end father_chips_amount_l2697_269721


namespace complement_of_B_l2697_269726

def U : Set Nat := {1, 2, 3, 4, 5, 6, 7}
def B : Set Nat := {1, 3, 5, 7}

theorem complement_of_B :
  (U \ B) = {2, 4, 6} := by
  sorry

end complement_of_B_l2697_269726


namespace coin_array_problem_l2697_269749

/-- The sum of the first n natural numbers -/
def triangular_sum (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem coin_array_problem :
  ∃ N : ℕ, triangular_sum N = 3003 ∧ sum_of_digits N = 14 := by
  sorry

end coin_array_problem_l2697_269749


namespace connect_four_shapes_l2697_269727

/-- Represents a Connect Four board configuration --/
def ConnectFourBoard := Fin 7 → Fin 9

/-- The number of unique shapes in a Connect Four board --/
def num_unique_shapes : ℕ :=
  let symmetric_shapes := 9^4
  let total_shapes := 9^7
  symmetric_shapes + (total_shapes - symmetric_shapes) / 2

/-- The sum of the first n natural numbers --/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The theorem stating that the number of unique shapes in a Connect Four board
    is equal to 9 times the sum of the first 729 natural numbers --/
theorem connect_four_shapes :
  num_unique_shapes = 9 * sum_first_n 729 := by
  sorry


end connect_four_shapes_l2697_269727


namespace price_decrease_percentage_l2697_269755

theorem price_decrease_percentage (original_price new_price : ℝ) 
  (h1 : original_price = 900)
  (h2 : new_price = 684) :
  (original_price - new_price) / original_price * 100 = 24 := by
  sorry

end price_decrease_percentage_l2697_269755


namespace boat_round_trip_average_speed_l2697_269757

/-- The average speed of a boat on a round trip, given its upstream and downstream speeds -/
theorem boat_round_trip_average_speed (distance : ℝ) (upstream_speed downstream_speed : ℝ) 
  (h1 : upstream_speed = 6)
  (h2 : downstream_speed = 3)
  (h3 : distance > 0) :
  (2 * distance) / ((distance / upstream_speed) + (distance / downstream_speed)) = 4 := by
  sorry

#check boat_round_trip_average_speed

end boat_round_trip_average_speed_l2697_269757


namespace village_population_equality_l2697_269771

/-- The initial population of Village X -/
def Px : ℕ := sorry

/-- The yearly decrease in population of Village X -/
def decrease_x : ℕ := 1200

/-- The initial population of Village Y -/
def Py : ℕ := 42000

/-- The yearly increase in population of Village Y -/
def increase_y : ℕ := 800

/-- The number of years after which the populations will be equal -/
def years : ℕ := 17

theorem village_population_equality :
  Px - years * decrease_x = Py + years * increase_y ∧ Px = 76000 := by sorry

end village_population_equality_l2697_269771


namespace product_of_binary_and_ternary_l2697_269789

-- Define the binary number 1101₂
def binary_num : ℕ := 13

-- Define the ternary number 102₃
def ternary_num : ℕ := 11

-- Theorem statement
theorem product_of_binary_and_ternary :
  binary_num * ternary_num = 143 := by sorry

end product_of_binary_and_ternary_l2697_269789


namespace largest_circle_at_A_l2697_269773

/-- Represents a pentagon with circles at its vertices -/
structure PentagonWithCircles where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  DE : ℝ
  AE : ℝ
  radA : ℝ
  radB : ℝ
  radC : ℝ
  radD : ℝ
  radE : ℝ
  circle_contact : 
    AB = radA + radB ∧
    BC = radB + radC ∧
    CD = radC + radD ∧
    DE = radD + radE ∧
    AE = radE + radA

/-- The circle centered at A has the largest radius -/
theorem largest_circle_at_A (p : PentagonWithCircles) 
  (h1 : p.AB = 16) (h2 : p.BC = 14) (h3 : p.CD = 17) (h4 : p.DE = 13) (h5 : p.AE = 14) :
  p.radA = max p.radA (max p.radB (max p.radC (max p.radD p.radE))) := by
  sorry

end largest_circle_at_A_l2697_269773


namespace positive_solution_x_l2697_269719

theorem positive_solution_x (x y z : ℝ) 
  (eq1 : x * y = 12 - 3 * x - 4 * y)
  (eq2 : y * z = 8 - 2 * y - 3 * z)
  (eq3 : x * z = 42 - 5 * x - 6 * z)
  (h_positive : x > 0) : x = 6 := by
  sorry

end positive_solution_x_l2697_269719


namespace women_per_table_l2697_269739

theorem women_per_table (tables : Nat) (men_per_table : Nat) (total_customers : Nat) :
  tables = 9 →
  men_per_table = 3 →
  total_customers = 90 →
  (total_customers - tables * men_per_table) / tables = 7 := by
  sorry

end women_per_table_l2697_269739


namespace number_relationship_l2697_269705

theorem number_relationship (A B C : ℝ) 
  (h1 : B = 10)
  (h2 : A * B = 85)
  (h3 : B * C = 115)
  (h4 : B - A = C - B) :
  B - A = 1.5 ∧ C - B = 1.5 := by
sorry

end number_relationship_l2697_269705


namespace angle_1303_equivalent_to_negative_137_l2697_269778

-- Define a function to reduce an angle to its equivalent angle between 0° and 360°
def reduce_angle (angle : Int) : Int :=
  angle % 360

-- Theorem statement
theorem angle_1303_equivalent_to_negative_137 :
  reduce_angle 1303 = reduce_angle (-137) :=
sorry

end angle_1303_equivalent_to_negative_137_l2697_269778


namespace scientific_notation_of_280000_l2697_269796

theorem scientific_notation_of_280000 : 
  280000 = 2.8 * (10 : ℝ)^5 := by sorry

end scientific_notation_of_280000_l2697_269796


namespace nested_expression_evaluation_l2697_269782

theorem nested_expression_evaluation : (2*(2*(2*(2*(2*(2*(3+2)+2)+2)+2)+2)+2)+2) = 446 := by
  sorry

end nested_expression_evaluation_l2697_269782


namespace abel_overtake_distance_l2697_269787

/-- Represents the race scenario between Abel and Kelly -/
structure RaceScenario where
  totalDistance : ℝ
  headStart : ℝ
  lossDistance : ℝ

/-- Calculates the distance Abel needs to run to overtake Kelly -/
def distanceToOvertake (race : RaceScenario) : ℝ :=
  race.totalDistance - (race.totalDistance - race.headStart + race.lossDistance)

/-- Theorem stating that Abel needs to run 98 meters to overtake Kelly -/
theorem abel_overtake_distance (race : RaceScenario) 
  (h1 : race.totalDistance = 100)
  (h2 : race.headStart = 3)
  (h3 : race.lossDistance = 0.5) :
  distanceToOvertake race = 98 := by
  sorry

#eval distanceToOvertake { totalDistance := 100, headStart := 3, lossDistance := 0.5 }

end abel_overtake_distance_l2697_269787


namespace loan_problem_l2697_269712

/-- Proves that given the conditions of the loan problem, the second part is lent for 3 years -/
theorem loan_problem (total : ℝ) (second_part : ℝ) (rate1 : ℝ) (rate2 : ℝ) (time1 : ℝ) (n : ℝ) : 
  total = 2717 →
  second_part = 1672 →
  rate1 = 0.03 →
  rate2 = 0.05 →
  time1 = 8 →
  (total - second_part) * rate1 * time1 = second_part * rate2 * n →
  n = 3 := by
sorry


end loan_problem_l2697_269712


namespace det_max_value_l2697_269745

open Real

-- Define the determinant function
noncomputable def det (θ : ℝ) : ℝ :=
  let a := 1 + sin θ
  let b := 1 + cos θ
  a * (1 - a^2) - b * (1 - a * b) + (1 - b * a)

-- State the theorem
theorem det_max_value :
  ∀ θ : ℝ, det θ ≤ -1 ∧ ∃ θ₀ : ℝ, det θ₀ = -1 :=
by sorry

end det_max_value_l2697_269745


namespace binomial_probability_two_successes_l2697_269741

/-- The probability mass function for a Binomial distribution -/
def binomial_pmf (n : ℕ) (p : ℝ) (k : ℕ) : ℝ :=
  (Nat.choose n k) * p^k * (1 - p)^(n - k)

/-- Theorem: For a random variable ξ following Binomial distribution B(3, 1/3), P(ξ=2) = 2/9 -/
theorem binomial_probability_two_successes :
  binomial_pmf 3 (1/3 : ℝ) 2 = 2/9 := by
  sorry

end binomial_probability_two_successes_l2697_269741


namespace prime_sum_square_fourth_power_l2697_269751

theorem prime_sum_square_fourth_power :
  ∀ p q r : ℕ,
  Prime p → Prime q → Prime r →
  p + q^2 = r^4 →
  p = 7 ∧ q = 3 ∧ r = 2 :=
by sorry

end prime_sum_square_fourth_power_l2697_269751


namespace count_D_eq_3_is_13_l2697_269744

/-- D(n) is the number of pairs of different adjacent digits in the binary representation of n -/
def D (n : ℕ) : ℕ := sorry

/-- The count of positive integers n ≤ 200 for which D(n) = 3 -/
def count_D_eq_3 : ℕ := sorry

theorem count_D_eq_3_is_13 : count_D_eq_3 = 13 := by sorry

end count_D_eq_3_is_13_l2697_269744


namespace distance_between_A_and_B_l2697_269713

def A : ℝ × ℝ := (-3, 1)
def B : ℝ × ℝ := (6, -4)

theorem distance_between_A_and_B : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 106 := by
  sorry

end distance_between_A_and_B_l2697_269713


namespace angle_relationship_l2697_269718

theorem angle_relationship (α β : Real) 
  (h1 : 0 < α ∧ α < π/2) 
  (h2 : 0 < β ∧ β < π/2) 
  (h3 : 2 * Real.sin α = Real.sin α * Real.cos β + Real.cos α * Real.sin β) : 
  α < β := by
sorry

end angle_relationship_l2697_269718


namespace max_sphere_volume_in_prism_l2697_269700

/-- The maximum volume of a sphere inscribed in a right triangular prism -/
theorem max_sphere_volume_in_prism (a b h : ℝ) (ha : 0 < a) (hb : 0 < b) (hh : 0 < h) :
  let r := min (h / 2) (a * b / (a + b + (a^2 + b^2).sqrt))
  (4 / 3) * π * r^3 = (9 * π) / 2 :=
by sorry

end max_sphere_volume_in_prism_l2697_269700


namespace equation_solution_l2697_269772

theorem equation_solution : ∃! x : ℝ, (1 : ℝ) / (x + 3) = 3 / (x + 9) := by
  sorry

end equation_solution_l2697_269772


namespace log_value_proof_l2697_269716

theorem log_value_proof (a : ℝ) (h1 : a > 0) (h2 : a^(1/2) = 4/9) :
  Real.log a / Real.log (2/3) = 4 := by
  sorry

end log_value_proof_l2697_269716


namespace symmetry_of_shifted_function_l2697_269731

open Real

theorem symmetry_of_shifted_function :
  ∃ α : ℝ, 0 < α ∧ α < π / 3 ∧
  ∀ x : ℝ, (sin (x + α) + Real.sqrt 3 * cos (x + α)) =
           (sin (-x + α) + Real.sqrt 3 * cos (-x + α)) := by
  sorry

end symmetry_of_shifted_function_l2697_269731


namespace house_construction_bricks_house_construction_bricks_specific_l2697_269701

/-- Calculates the number of bricks needed for house construction given specific costs and requirements. -/
theorem house_construction_bricks (land_cost_per_sqm : ℕ) (brick_cost_per_thousand : ℕ) 
  (roof_tile_cost : ℕ) (land_area : ℕ) (roof_tiles : ℕ) (total_cost : ℕ) : ℕ :=
  let land_cost := land_cost_per_sqm * land_area
  let roof_cost := roof_tile_cost * roof_tiles
  let brick_budget := total_cost - land_cost - roof_cost
  let bricks_thousands := brick_budget / brick_cost_per_thousand
  bricks_thousands * 1000

/-- Proves that given the specific conditions, the number of bricks needed is 10,000. -/
theorem house_construction_bricks_specific : 
  house_construction_bricks 50 100 10 2000 500 106000 = 10000 := by
  sorry

end house_construction_bricks_house_construction_bricks_specific_l2697_269701


namespace guest_bathroom_towel_sets_l2697_269779

theorem guest_bathroom_towel_sets :
  let master_sets : ℕ := 4
  let guest_price : ℚ := 40
  let master_price : ℚ := 50
  let discount : ℚ := 20 / 100
  let total_spent : ℚ := 224
  let discounted_guest_price : ℚ := guest_price * (1 - discount)
  let discounted_master_price : ℚ := master_price * (1 - discount)
  ∃ guest_sets : ℕ,
    guest_sets * discounted_guest_price + master_sets * discounted_master_price = total_spent ∧
    guest_sets = 2 :=
by sorry

end guest_bathroom_towel_sets_l2697_269779


namespace quadratic_function_from_roots_and_point_l2697_269725

theorem quadratic_function_from_roots_and_point (f : ℝ → ℝ) :
  (∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c) →  -- f is a quadratic function
  f 0 = 2 →                                       -- f(0) = 2
  (∃ x, f x = 0 ∧ x = -2) →                       -- -2 is a root
  (∃ x, f x = 0 ∧ x = 1) →                        -- 1 is a root
  ∀ x, f x = -x^2 - x + 2 :=                      -- Conclusion: f(x) = -x^2 - x + 2
by sorry

end quadratic_function_from_roots_and_point_l2697_269725


namespace max_value_a_l2697_269735

theorem max_value_a (a b c d : ℕ+) 
  (h1 : a < 2 * b)
  (h2 : b < 3 * c)
  (h3 : c < 2 * d)
  (h4 : d < 100) :
  a ≤ 1179 ∧ ∃ (a' b' c' d' : ℕ+), 
    a' = 1179 ∧
    a' < 2 * b' ∧
    b' < 3 * c' ∧
    c' < 2 * d' ∧
    d' < 100 :=
by sorry

end max_value_a_l2697_269735


namespace max_product_a2_a6_l2697_269748

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem stating the maximum value of a₂ * a₆ in an arithmetic sequence where a₄ = 2 -/
theorem max_product_a2_a6 (a : ℕ → ℝ) (h : ArithmeticSequence a) (h4 : a 4 = 2) :
  (∀ b c : ℝ, a 2 = b ∧ a 6 = c → b * c ≤ 4) ∧ (∃ b c : ℝ, a 2 = b ∧ a 6 = c ∧ b * c = 4) :=
sorry

end max_product_a2_a6_l2697_269748


namespace f_odd_and_increasing_l2697_269766

-- Define the function f(x) = x|x|
def f (x : ℝ) : ℝ := x * |x|

-- State the theorem
theorem f_odd_and_increasing : 
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, x < y → f x < f y) :=
sorry

end f_odd_and_increasing_l2697_269766


namespace quadratic_trinomials_equal_sum_squares_l2697_269723

/-- 
Given two quadratic trinomials f(x) = x^2 - 6x + 4a and g(x) = x^2 + ax + 6,
prove that a = -12 is the only value for which both trinomials have two roots
and the sum of the squares of the roots of f(x) equals the sum of the squares
of the roots of g(x).
-/
theorem quadratic_trinomials_equal_sum_squares (a : ℝ) : 
  (∃ x y : ℝ, x^2 - 6*x + 4*a = 0 ∧ y^2 + a*y + 6 = 0) ∧ 
  (∃ x₁ x₂ y₁ y₂ : ℝ, 
    x₁^2 - 6*x₁ + 4*a = 0 ∧ 
    x₂^2 - 6*x₂ + 4*a = 0 ∧ 
    y₁^2 + a*y₁ + 6 = 0 ∧ 
    y₂^2 + a*y₂ + 6 = 0 ∧ 
    x₁^2 + x₂^2 = y₁^2 + y₂^2) →
  a = -12 :=
by sorry

end quadratic_trinomials_equal_sum_squares_l2697_269723


namespace woody_savings_l2697_269784

/-- The amount of money Woody already has -/
def money_saved (console_cost weekly_allowance weeks_to_save : ℕ) : ℕ :=
  console_cost - weekly_allowance * weeks_to_save

theorem woody_savings : money_saved 282 24 10 = 42 := by
  sorry

end woody_savings_l2697_269784


namespace sum_of_powers_zero_l2697_269754

theorem sum_of_powers_zero : -(-1)^2006 - (-1)^2007 - 1^2008 - (-1)^2009 = 0 := by
  sorry

end sum_of_powers_zero_l2697_269754


namespace isosceles_trapezoid_side_length_l2697_269706

/-- An isosceles trapezoid with given properties -/
structure IsoscelesTrapezoid where
  area : ℝ
  base1 : ℝ
  base2 : ℝ
  side : ℝ

/-- The theorem stating the side length of the specific isosceles trapezoid -/
theorem isosceles_trapezoid_side_length 
  (t : IsoscelesTrapezoid) 
  (h_area : t.area = 44)
  (h_base1 : t.base1 = 8)
  (h_base2 : t.base2 = 14) :
  t.side = 5 := by
  sorry

#check isosceles_trapezoid_side_length

end isosceles_trapezoid_side_length_l2697_269706


namespace solution_set_equality_l2697_269730

theorem solution_set_equality : 
  {x : ℝ | x^2 - 2*x ≤ 0} = {x : ℝ | 0 ≤ x ∧ x ≤ 2} := by sorry

end solution_set_equality_l2697_269730


namespace cruz_marbles_l2697_269710

/-- 
Given:
- Three times the sum of marbles that Atticus, Jensen, and Cruz have is equal to 60.
- Atticus has half as many marbles as Jensen.
- Atticus has 4 marbles.
Prove that Cruz has 8 marbles.
-/
theorem cruz_marbles (atticus jensen cruz : ℕ) : 
  3 * (atticus + jensen + cruz) = 60 →
  atticus = jensen / 2 →
  atticus = 4 →
  cruz = 8 := by
sorry

end cruz_marbles_l2697_269710


namespace five_spheres_configuration_exists_l2697_269761

/-- Represents a sphere in 3D space -/
structure Sphere where
  center : ℝ × ℝ × ℝ
  radius : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  normal : ℝ × ℝ × ℝ
  point : ℝ × ℝ × ℝ

/-- Checks if a plane is tangent to a sphere -/
def isTangent (p : Plane) (s : Sphere) : Prop := sorry

/-- Checks if a plane passes through a point -/
def passesThrough (p : Plane) (point : ℝ × ℝ × ℝ) : Prop := sorry

/-- Theorem stating the existence of a configuration of five spheres with the required property -/
theorem five_spheres_configuration_exists : 
  ∃ (s₁ s₂ s₃ s₄ s₅ : Sphere),
    (∃ (p₁ : Plane), passesThrough p₁ s₁.center ∧ 
      isTangent p₁ s₂ ∧ isTangent p₁ s₃ ∧ isTangent p₁ s₄ ∧ isTangent p₁ s₅) ∧
    (∃ (p₂ : Plane), passesThrough p₂ s₂.center ∧ 
      isTangent p₂ s₁ ∧ isTangent p₂ s₃ ∧ isTangent p₂ s₄ ∧ isTangent p₂ s₅) ∧
    (∃ (p₃ : Plane), passesThrough p₃ s₃.center ∧ 
      isTangent p₃ s₁ ∧ isTangent p₃ s₂ ∧ isTangent p₃ s₄ ∧ isTangent p₃ s₅) ∧
    (∃ (p₄ : Plane), passesThrough p₄ s₄.center ∧ 
      isTangent p₄ s₁ ∧ isTangent p₄ s₂ ∧ isTangent p₄ s₃ ∧ isTangent p₄ s₅) ∧
    (∃ (p₅ : Plane), passesThrough p₅ s₅.center ∧ 
      isTangent p₅ s₁ ∧ isTangent p₅ s₂ ∧ isTangent p₅ s₃ ∧ isTangent p₅ s₄) :=
by
  sorry

end five_spheres_configuration_exists_l2697_269761


namespace point_division_theorem_l2697_269763

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define points A, B, and P
variable (A B P : V)

-- Define the condition that P is on the line segment AB with the given ratio
def on_segment_with_ratio (A B P : V) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • A + t • B ∧ t = 5 / 8

-- Theorem statement
theorem point_division_theorem (h : on_segment_with_ratio A B P) :
  P = (3 / 8) • A + (5 / 8) • B := by sorry

end point_division_theorem_l2697_269763


namespace inequality_proof_l2697_269762

theorem inequality_proof (n : ℕ) (x : ℝ) (h1 : n ≥ 2) (h2 : |x| < 1) :
  2^n > (1 - x)^n + (1 + x)^n := by
  sorry

end inequality_proof_l2697_269762


namespace rehabilitation_centers_fraction_l2697_269709

theorem rehabilitation_centers_fraction (L J H Ja : ℕ) (f : ℚ) : 
  L = 6 →
  J = L - f * L →
  H = 2 * J - 2 →
  Ja = 2 * H + 6 →
  L + J + H + Ja = 27 →
  f = 1/2 := by sorry

end rehabilitation_centers_fraction_l2697_269709


namespace largest_fraction_l2697_269729

theorem largest_fraction : 
  let a := (5 : ℚ) / 12
  let b := (7 : ℚ) / 16
  let c := (23 : ℚ) / 48
  let d := (99 : ℚ) / 200
  let e := (201 : ℚ) / 400
  (e ≥ a) ∧ (e ≥ b) ∧ (e ≥ c) ∧ (e ≥ d) := by sorry

end largest_fraction_l2697_269729
