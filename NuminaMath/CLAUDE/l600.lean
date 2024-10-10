import Mathlib

namespace perpendicular_line_through_point_l600_60068

/-- Given a line l₁: 3x - 6y = 9 and a point P(-2, 4), 
    prove that the line l₂: y = -2x is perpendicular to l₁ and passes through P. -/
theorem perpendicular_line_through_point (x y : ℝ) : 
  let l₁ : ℝ → ℝ → Prop := λ x y ↦ 3 * x - 6 * y = 9
  let l₂ : ℝ → ℝ := λ x ↦ -2 * x
  let P : ℝ × ℝ := (-2, 4)
  (∀ x y, l₁ x y ↔ y = 1/2 * x - 3/2) ∧  -- l₁ in slope-intercept form
  (l₂ P.1 = P.2) ∧                      -- l₂ passes through P
  ((-2) * (1/2) = -1)                   -- l₁ and l₂ are perpendicular
  :=
by sorry

end perpendicular_line_through_point_l600_60068


namespace xy_equals_zero_l600_60046

theorem xy_equals_zero (x y : ℝ) (h1 : x + y = 4) (h2 : x^3 - y^3 = 64) : x * y = 0 := by
  sorry

end xy_equals_zero_l600_60046


namespace find_k_l600_60012

def vector_a (k : ℝ) : ℝ × ℝ := (k, 3)
def vector_b : ℝ × ℝ := (1, 4)
def vector_c : ℝ × ℝ := (2, 1)

def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

theorem find_k : ∃ k : ℝ, 
  perpendicular ((2 * (vector_a k).1 - 3 * vector_b.1, 2 * (vector_a k).2 - 3 * vector_b.2)) vector_c ∧ 
  k = 3 := by
  sorry

end find_k_l600_60012


namespace tom_apple_purchase_l600_60081

/-- The price of apples per kg -/
def apple_price : ℕ := 70

/-- The amount of mangoes Tom bought in kg -/
def mango_amount : ℕ := 9

/-- The price of mangoes per kg -/
def mango_price : ℕ := 55

/-- The total amount Tom paid -/
def total_paid : ℕ := 1055

/-- Theorem stating that Tom purchased 8 kg of apples -/
theorem tom_apple_purchase :
  ∃ (x : ℕ), x * apple_price + mango_amount * mango_price = total_paid ∧ x = 8 := by
  sorry

end tom_apple_purchase_l600_60081


namespace car_repair_cost_l600_60045

/-- Calculates the total cost for a car repair given the mechanic's hourly rate,
    hours worked per day, number of days worked, and cost of parts. -/
theorem car_repair_cost (hourly_rate : ℕ) (hours_per_day : ℕ) (days_worked : ℕ) (parts_cost : ℕ) :
  hourly_rate = 60 →
  hours_per_day = 8 →
  days_worked = 14 →
  parts_cost = 2500 →
  hourly_rate * hours_per_day * days_worked + parts_cost = 9220 := by
  sorry

#check car_repair_cost

end car_repair_cost_l600_60045


namespace digits_of_product_l600_60025

theorem digits_of_product : ∃ n : ℕ, n = 3^4 * 6^8 ∧ (Nat.log 10 n).succ = 9 := by sorry

end digits_of_product_l600_60025


namespace buratino_malvina_equation_l600_60087

theorem buratino_malvina_equation (x : ℝ) : 4 * x + 15 = 15 * x + 4 → x = 1 := by
  sorry

end buratino_malvina_equation_l600_60087


namespace yellow_face_probability_l600_60019

-- Define the die
def die_sides : ℕ := 8
def yellow_faces : ℕ := 3

-- Define the probability function
def probability (favorable_outcomes : ℕ) (total_outcomes : ℕ) : ℚ :=
  (favorable_outcomes : ℚ) / (total_outcomes : ℚ)

-- Theorem statement
theorem yellow_face_probability : 
  probability yellow_faces die_sides = 3 / 8 := by
  sorry

end yellow_face_probability_l600_60019


namespace terminal_side_in_third_quadrant_l600_60099

/-- Given an angle α = 7π/5, prove that its terminal side is located in the third quadrant. -/
theorem terminal_side_in_third_quadrant (α : Real) (h : α = 7 * Real.pi / 5) :
  ∃ (x y : Real), x < 0 ∧ y < 0 ∧ (∃ (t : Real), x = t * Real.cos α ∧ y = t * Real.sin α) :=
sorry

end terminal_side_in_third_quadrant_l600_60099


namespace remaining_money_l600_60009

def initial_amount : ℕ := 53
def toy_car_cost : ℕ := 11
def toy_car_quantity : ℕ := 2
def scarf_cost : ℕ := 10
def beanie_cost : ℕ := 14

theorem remaining_money :
  initial_amount - (toy_car_cost * toy_car_quantity + scarf_cost + beanie_cost) = 7 := by
  sorry

end remaining_money_l600_60009


namespace equation_solution_l600_60078

theorem equation_solution (m n : ℚ) : 
  (m * 1 + n * 1 = 6) → 
  (m * 2 + n * (-2) = 6) → 
  m = 4.5 ∧ n = 1.5 := by
sorry

end equation_solution_l600_60078


namespace arithmetic_sequence_property_l600_60015

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of specific terms in the sequence equals 120 -/
def SumCondition (a : ℕ → ℝ) : Prop :=
  a 4 + a 6 + a 8 + a 10 + a 12 = 120

theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h1 : ArithmeticSequence a) (h2 : SumCondition a) : 
  2 * a 10 - a 12 = 24 := by
  sorry

end arithmetic_sequence_property_l600_60015


namespace difference_of_squares_l600_60075

theorem difference_of_squares (x y : ℝ) : x^2 - y^2 = (x + y) * (x - y) := by
  sorry

end difference_of_squares_l600_60075


namespace largest_n_with_unique_k_l600_60048

theorem largest_n_with_unique_k : ∃ (n : ℕ), n > 0 ∧ n ≤ 136 ∧
  (∃! (k : ℤ), (9 : ℚ)/17 < (n : ℚ)/(n + k) ∧ (n : ℚ)/(n + k) < 8/15) ∧
  (∀ (m : ℕ), m > n → ¬(∃! (k : ℤ), (9 : ℚ)/17 < (m : ℚ)/(m + k) ∧ (m : ℚ)/(m + k) < 8/15)) :=
by sorry

end largest_n_with_unique_k_l600_60048


namespace q_equals_sixteen_l600_60001

/-- The polynomial with four distinct real roots in geometric progression -/
def polynomial (p q : ℝ) (x : ℝ) : ℝ := x^4 + p*x^3 + q*x^2 - 18*x + 16

/-- The roots of the polynomial form a geometric progression -/
def roots_in_geometric_progression (p q : ℝ) : Prop :=
  ∃ (a r : ℝ), a ≠ 0 ∧ r ≠ 1 ∧
    (polynomial p q a = 0) ∧
    (polynomial p q (a*r) = 0) ∧
    (polynomial p q (a*r^2) = 0) ∧
    (polynomial p q (a*r^3) = 0)

/-- The theorem stating that q equals 16 for the given conditions -/
theorem q_equals_sixteen (p q : ℝ) :
  roots_in_geometric_progression p q → q = 16 := by
  sorry

end q_equals_sixteen_l600_60001


namespace event_relationship_l600_60004

-- Define the critical value
def critical_value : ℝ := 6.635

-- Define the confidence level
def confidence_level : ℝ := 0.99

-- Define the relationship between K^2 and the confidence level
theorem event_relationship (K : ℝ) :
  K^2 > critical_value → confidence_level = 0.99 := by
  sorry

#check event_relationship

end event_relationship_l600_60004


namespace area_between_circles_l600_60041

theorem area_between_circles (r_small : ℝ) (r_large : ℝ) : 
  r_small = 2 →
  r_large = 5 * r_small →
  (π * r_large^2 - π * r_small^2) = 96 * π := by
sorry

end area_between_circles_l600_60041


namespace calculation_proof_l600_60014

theorem calculation_proof :
  (0.001)^(-1/3) + 27^(2/3) + (1/4)^(-1/2) - (1/9)^(-3/2) = -6 ∧
  1/2 * Real.log 25 / Real.log 10 + Real.log 2 / Real.log 10 - Real.log (Real.sqrt 0.1) / Real.log 10 - 
    (Real.log 9 / Real.log 2) * (Real.log 2 / Real.log 3) = -1/2 := by
  sorry

end calculation_proof_l600_60014


namespace triangle_properties_l600_60065

noncomputable section

/-- Triangle ABC with area S and sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  S : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem stating the properties of the triangle -/
theorem triangle_properties (t : Triangle) 
  (h1 : t.S = (3/2) * t.b * t.c * Real.cos t.A)
  (h2 : t.C = π/4)
  (h3 : t.S = 24) :
  Real.cos t.B = Real.sqrt 5 / 5 ∧ t.b = 8 := by
  sorry

end

end triangle_properties_l600_60065


namespace negative_300_equals_60_l600_60028

/-- Two angles have the same terminal side if they differ by a multiple of 360° -/
def same_terminal_side (α β : ℝ) : Prop :=
  ∃ k : ℤ, α = β + 360 * k

/-- Prove that -300° and 60° have the same terminal side -/
theorem negative_300_equals_60 : same_terminal_side (-300) 60 := by
  sorry

end negative_300_equals_60_l600_60028


namespace fruit_card_probability_l600_60026

theorem fruit_card_probability (total_cards : ℕ) (fruit_cards : ℕ) 
  (h1 : total_cards = 6)
  (h2 : fruit_cards = 2) :
  (fruit_cards : ℚ) / total_cards = 1 / 3 := by
  sorry

end fruit_card_probability_l600_60026


namespace transformation_sequence_l600_60090

def rotate_x (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (x, -z, y)

def reflect_xy (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (x, y, -z)

def reflect_yz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (-x, y, z)

def rotate_y (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (z, y, -x)

def reflect_xz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (x, -y, z)

def initial_point : ℝ × ℝ × ℝ := (2, 2, 2)

theorem transformation_sequence :
  (reflect_xz ∘ rotate_y ∘ reflect_yz ∘ reflect_xy ∘ rotate_x) initial_point = (2, 2, -2) := by
  sorry

end transformation_sequence_l600_60090


namespace expenditure_ratio_l600_60055

/-- Represents a person with income and expenditure -/
structure Person where
  income : ℕ
  expenditure : ℕ

/-- The problem setup -/
def problem_setup : Prop := ∃ (p1 p2 : Person),
  -- The ratio of incomes is 5:4
  p1.income * 4 = p2.income * 5 ∧
  -- Each person saves 2200
  p1.income - p1.expenditure = 2200 ∧
  p2.income - p2.expenditure = 2200 ∧
  -- P1's income is 5500
  p1.income = 5500

/-- The theorem to prove -/
theorem expenditure_ratio (h : problem_setup) :
  ∃ (p1 p2 : Person), p1.expenditure * 2 = p2.expenditure * 3 :=
sorry

end expenditure_ratio_l600_60055


namespace wedding_attendance_percentage_l600_60047

theorem wedding_attendance_percentage 
  (total_invitations : ℕ) 
  (rsvp_rate : ℚ)
  (thank_you_cards : ℕ) 
  (no_gift_attendees : ℕ) :
  total_invitations = 200 →
  rsvp_rate = 9/10 →
  thank_you_cards = 134 →
  no_gift_attendees = 10 →
  (thank_you_cards + no_gift_attendees) / (total_invitations * rsvp_rate) = 4/5 := by
sorry

#eval (134 + 10) / (200 * (9/10)) -- This should evaluate to 4/5

end wedding_attendance_percentage_l600_60047


namespace wife_cookie_percentage_l600_60098

theorem wife_cookie_percentage (total_cookies : ℕ) (daughter_cookies : ℕ) (uneaten_cookies : ℕ) :
  total_cookies = 200 →
  daughter_cookies = 40 →
  uneaten_cookies = 50 →
  ∃ (wife_percentage : ℚ),
    wife_percentage = 30 ∧
    (total_cookies - (wife_percentage / 100) * total_cookies - daughter_cookies) / 2 = uneaten_cookies :=
by sorry

end wife_cookie_percentage_l600_60098


namespace whistlers_count_l600_60018

/-- The number of whistlers in each of Koby's boxes -/
def whistlers_per_box : ℕ := sorry

/-- The total number of fireworks Koby and Cherie have -/
def total_fireworks : ℕ := 33

/-- The number of boxes Koby has -/
def koby_boxes : ℕ := 2

/-- The number of sparklers in each of Koby's boxes -/
def koby_sparklers_per_box : ℕ := 3

/-- The number of sparklers in Cherie's box -/
def cherie_sparklers : ℕ := 8

/-- The number of whistlers in Cherie's box -/
def cherie_whistlers : ℕ := 9

theorem whistlers_count : whistlers_per_box = 5 := by
  have h1 : total_fireworks = koby_boxes * (koby_sparklers_per_box + whistlers_per_box) + cherie_sparklers + cherie_whistlers := by sorry
  sorry

end whistlers_count_l600_60018


namespace difference_of_squares_of_odd_numbers_divisible_by_eight_l600_60054

theorem difference_of_squares_of_odd_numbers_divisible_by_eight (a b : ℤ) 
  (ha : ∃ k : ℤ, a = 2 * k + 1) (hb : ∃ k : ℤ, b = 2 * k + 1) : 
  ∃ m : ℤ, a^2 - b^2 = 8 * m :=
sorry

end difference_of_squares_of_odd_numbers_divisible_by_eight_l600_60054


namespace units_digit_of_19_times_37_l600_60092

theorem units_digit_of_19_times_37 : (19 * 37) % 10 = 3 := by
  sorry

end units_digit_of_19_times_37_l600_60092


namespace ice_cream_bars_per_friend_l600_60013

/-- Proves that given the conditions of the ice cream problem, each friend wants to eat 2 bars. -/
theorem ice_cream_bars_per_friend 
  (box_cost : ℚ) 
  (bars_per_box : ℕ) 
  (num_friends : ℕ) 
  (cost_per_person : ℚ) 
  (h1 : box_cost = 15/2)
  (h2 : bars_per_box = 3)
  (h3 : num_friends = 6)
  (h4 : cost_per_person = 5) : 
  (num_friends * cost_per_person / box_cost * bars_per_box) / num_friends = 2 := by
sorry

end ice_cream_bars_per_friend_l600_60013


namespace congruence_solution_extension_l600_60042

theorem congruence_solution_extension 
  (p : ℕ) (n a : ℕ) (h_prime : Nat.Prime p) 
  (h_n : ¬ p ∣ n) (h_a : ¬ p ∣ a) 
  (h_base : ∃ x : ℕ, x^n ≡ a [MOD p]) :
  ∀ r : ℕ, ∃ y : ℕ, y^n ≡ a [MOD p^r] :=
by sorry

end congruence_solution_extension_l600_60042


namespace sqrt_sum_squares_equals_sum_l600_60052

theorem sqrt_sum_squares_equals_sum (a b c : ℝ) :
  Real.sqrt (a^2 + b^2 + c^2) = a + b + c ↔ a * b + a * c + b * c = 0 ∧ a + b + c ≥ 0 := by
  sorry

end sqrt_sum_squares_equals_sum_l600_60052


namespace series_convergence_l600_60088

theorem series_convergence 
  (u v : ℕ → ℝ) 
  (hu : Summable (fun i => (u i)^2))
  (hv : Summable (fun i => (v i)^2))
  (p : ℕ) 
  (hp : p ≥ 2) : 
  Summable (fun i => (u i - v i)^p) :=
by
  sorry

end series_convergence_l600_60088


namespace isosceles_triangulation_condition_l600_60007

/-- A regular convex polygon with n sides -/
structure RegularConvexPolygon where
  n : ℕ
  n_ge_3 : n ≥ 3

/-- A triangulation of a polygon -/
structure Triangulation (P : RegularConvexPolygon) where
  isosceles : Bool

/-- Theorem: If a regular convex polygon with n sides has a triangulation
    consisting of only isosceles triangles, then n can be written as 2^(a+1) + 2^b
    for some non-negative integers a and b -/
theorem isosceles_triangulation_condition (P : RegularConvexPolygon)
  (T : Triangulation P) (h : T.isosceles = true) :
  ∃ (a b : ℕ), P.n = 2^(a+1) + 2^b :=
sorry

end isosceles_triangulation_condition_l600_60007


namespace b_over_c_equals_27_l600_60029

-- Define the coefficients of the quadratic equations
variable (a b c : ℝ)

-- Define the roots of the second equation
variable (s₁ s₂ : ℝ)

-- Assumptions
axiom a_nonzero : a ≠ 0
axiom b_nonzero : b ≠ 0
axiom c_nonzero : c ≠ 0

-- Define the relationships between roots and coefficients
axiom vieta_sum : c = -(s₁ + s₂)
axiom vieta_product : a = s₁ * s₂

-- Define the relationship between the roots of the two equations
axiom root_relationship : a = -(3*s₁ + 3*s₂) ∧ b = 9*s₁*s₂

-- Theorem to prove
theorem b_over_c_equals_27 : b / c = 27 := by
  sorry

end b_over_c_equals_27_l600_60029


namespace adams_trivia_score_l600_60079

/-- Adam's trivia game score calculation -/
theorem adams_trivia_score :
  ∀ (first_half second_half points_per_question : ℕ),
  first_half = 8 →
  second_half = 2 →
  points_per_question = 8 →
  (first_half + second_half) * points_per_question = 80 :=
by
  sorry

end adams_trivia_score_l600_60079


namespace f_properties_l600_60034

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * x) - Real.cos (2 * x) + 1

theorem f_properties :
  (∀ x : ℝ, f (π/12 + x) = f (π/12 - x)) ∧
  (¬ ∀ x ∈ Set.Ioo (5*π/12) (11*π/12), ∀ y ∈ Set.Ioo (5*π/12) (11*π/12), x < y → f x > f y) ∧
  (∀ x : ℝ, f (π/3 + x) = f (π/3 - x)) ∧
  (∀ x : ℝ, f x ≤ 3) ∧ (∃ x : ℝ, f x = 3) :=
by sorry

end f_properties_l600_60034


namespace no_snow_probability_l600_60024

theorem no_snow_probability (p : ℚ) : 
  p = 2/3 → (1 - p)^4 = 1/81 := by sorry

end no_snow_probability_l600_60024


namespace sum_a_3000_l600_60082

def a (n : ℕ) : ℕ :=
  if n % 15 = 0 ∧ n % 18 = 0 then 15
  else if n % 18 = 0 ∧ n % 17 = 0 then 18
  else if n % 15 = 0 ∧ n % 17 = 0 then 21
  else 0

theorem sum_a_3000 :
  (Finset.range 3000).sum (fun n => a (n + 1)) = 888 := by
  sorry

end sum_a_3000_l600_60082


namespace sum_difference_theorem_l600_60008

def round_to_nearest_five (n : ℕ) : ℕ :=
  5 * ((n + 2) / 5)

def jo_sum (n : ℕ) : ℕ :=
  n * (n + 1) / 2

def lisa_sum (n : ℕ) : ℕ :=
  (Finset.range n).sum (λ i => round_to_nearest_five (i + 1))

theorem sum_difference_theorem :
  jo_sum 60 - lisa_sum 60 = 240 := by
  sorry

end sum_difference_theorem_l600_60008


namespace second_race_length_l600_60053

/-- Given a 100 m race where A beats B by 10 m and C by 13 m, and another race where B beats C by 6 m,
    prove that the length of the second race is 180 meters. -/
theorem second_race_length (vA vB vC : ℝ) (t : ℝ) (h1 : vA * t = 100)
                            (h2 : vB * t = 90) (h3 : vC * t = 87) : 
  ∃ (L : ℝ), L / vB = (L - 6) / vC ∧ L = 180 := by
  sorry

end second_race_length_l600_60053


namespace percentage_of_men_in_company_l600_60021

/-- The percentage of men in a company, given attendance rates at a company picnic -/
theorem percentage_of_men_in_company : 
  ∀ (M : ℝ), 
  (M ≥ 0) →  -- M is non-negative
  (M ≤ 1) →  -- M is at most 1
  (0.20 * M + 0.40 * (1 - M) = 0.33) →  -- Picnic attendance equation
  (M = 0.35) :=  -- Conclusion: 35% of employees are men
by sorry

end percentage_of_men_in_company_l600_60021


namespace product_divisibility_l600_60067

def die_numbers : Finset ℕ := Finset.range 8

theorem product_divisibility (visible : Finset ℕ) 
  (h1 : visible ⊆ die_numbers) 
  (h2 : visible.card = 6) : 
  96 ∣ visible.prod id :=
sorry

end product_divisibility_l600_60067


namespace largest_prime_factor_of_12321_l600_60059

theorem largest_prime_factor_of_12321 : ∃ p : ℕ, p.Prime ∧ p ∣ 12321 ∧ ∀ q : ℕ, q.Prime → q ∣ 12321 → q ≤ p :=
by sorry

end largest_prime_factor_of_12321_l600_60059


namespace marble_arrangement_count_l600_60005

def total_arrangements (n : ℕ) : ℕ := Nat.factorial n

def adjacent_arrangements (n : ℕ) : ℕ := 2 * Nat.factorial (n - 1)

theorem marble_arrangement_count :
  total_arrangements 5 - adjacent_arrangements 5 = 72 := by
  sorry

end marble_arrangement_count_l600_60005


namespace ancient_chinese_problem_correct_l600_60033

/-- Represents the system of equations for the ancient Chinese math problem --/
def ancient_chinese_problem (x y : ℤ) : Prop :=
  (y = 8*x - 3) ∧ (y = 7*x + 4)

/-- Theorem stating that the system of equations correctly represents the problem --/
theorem ancient_chinese_problem_correct (x y : ℤ) :
  (ancient_chinese_problem x y) ↔
  (x ≥ 0) ∧  -- number of people is non-negative
  (y ≥ 0) ∧  -- price is non-negative
  (8*x - y = 3) ∧  -- excess of 3 coins when each contributes 8
  (y - 7*x = 4)    -- shortage of 4 coins when each contributes 7
  := by sorry

end ancient_chinese_problem_correct_l600_60033


namespace max_sum_of_digits_12hour_clock_l600_60036

/-- Represents a time in 12-hour format -/
structure Time12Hour where
  hours : Nat
  minutes : Nat
  seconds : Nat
  hours_valid : hours ≥ 1 ∧ hours ≤ 12
  minutes_valid : minutes ≤ 59
  seconds_valid : seconds ≤ 59

/-- Calculates the sum of digits in a natural number -/
def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- Calculates the sum of digits for a given Time12Hour -/
def sumOfTimeDigits (t : Time12Hour) : Nat :=
  sumOfDigits t.hours + sumOfDigits t.minutes + sumOfDigits t.seconds

/-- The maximum sum of digits on a 12-hour format digital clock -/
theorem max_sum_of_digits_12hour_clock :
  ∃ (t : Time12Hour), ∀ (t' : Time12Hour), sumOfTimeDigits t ≥ sumOfTimeDigits t' ∧ sumOfTimeDigits t = 37 :=
sorry

end max_sum_of_digits_12hour_clock_l600_60036


namespace quadratic_equation_solution_l600_60010

theorem quadratic_equation_solution :
  ∃ x : ℝ, 3 * x^2 - 6 * x + 3 = 0 ∧ x = 1 :=
by sorry

end quadratic_equation_solution_l600_60010


namespace inverse_proposition_false_l600_60032

theorem inverse_proposition_false : ∃ a b : ℝ, (abs a = abs b) ∧ (a ≠ b) := by
  sorry

end inverse_proposition_false_l600_60032


namespace transmission_time_approx_seven_minutes_l600_60095

/-- Represents the data transmission scenario -/
structure DataTransmission where
  num_blocks : ℕ
  chunks_per_block : ℕ
  transmission_rate : ℕ
  delay_per_block : ℕ

/-- Calculates the total transmission time in minutes -/
def total_transmission_time (dt : DataTransmission) : ℚ :=
  let total_chunks := dt.num_blocks * dt.chunks_per_block
  let transmission_time := total_chunks / dt.transmission_rate
  let total_delay := dt.num_blocks * dt.delay_per_block
  (transmission_time + total_delay) / 60

/-- Theorem stating that the transmission time is approximately 7 minutes -/
theorem transmission_time_approx_seven_minutes (dt : DataTransmission) 
  (h1 : dt.num_blocks = 80)
  (h2 : dt.chunks_per_block = 600)
  (h3 : dt.transmission_rate = 150)
  (h4 : dt.delay_per_block = 1) :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1 ∧ |total_transmission_time dt - 7| < ε :=
sorry

end transmission_time_approx_seven_minutes_l600_60095


namespace min_value_theorem_l600_60016

/-- The function f(x) defined as |x+a| + |x+3a| -/
def f (a : ℝ) (x : ℝ) : ℝ := |x + a| + |x + 3*a|

/-- The theorem stating the minimum value of 1/m^2 + n^2 given conditions -/
theorem min_value_theorem (a m n : ℝ) :
  (∃ (x : ℝ), ∀ (y : ℝ), f a y ≥ f a x) →  -- minimum of f(x) exists
  (∀ (x : ℝ), f a x ≥ 2) →  -- minimum value of f(x) is 2
  (a - m) * (a + m) = 4 / n^2 →  -- given condition
  (∃ (k : ℝ), ∀ (p q : ℝ), 1 / p^2 + q^2 ≥ k ∧ (1 / m^2 + n^2 = k)) →  -- minimum of 1/m^2 + n^2 exists
  (1 / m^2 + n^2 = 9)  -- conclusion: minimum value is 9
:= by sorry

end min_value_theorem_l600_60016


namespace cake_sugar_calculation_l600_60022

/-- The amount of sugar stored in the house (in pounds) -/
def sugar_stored : ℕ := 287

/-- The amount of additional sugar needed (in pounds) -/
def sugar_additional : ℕ := 163

/-- The total amount of sugar needed for the cake (in pounds) -/
def total_sugar_needed : ℕ := sugar_stored + sugar_additional

theorem cake_sugar_calculation :
  total_sugar_needed = 450 :=
by sorry

end cake_sugar_calculation_l600_60022


namespace arithmetic_square_root_of_16_l600_60094

theorem arithmetic_square_root_of_16 : 
  ∃ x : ℝ, x ≥ 0 ∧ x^2 = 16 ∧ x = 4 := by sorry

end arithmetic_square_root_of_16_l600_60094


namespace solution_difference_l600_60069

theorem solution_difference (r s : ℝ) : 
  (∀ x : ℝ, (5 * x - 15) / (x^2 + 3 * x - 18) = x + 3 → x = r ∨ x = s) →
  r ≠ s →
  r > s →
  r - s = Real.sqrt 29 := by
sorry

end solution_difference_l600_60069


namespace percentage_problem_l600_60077

theorem percentage_problem (x : ℝ) : (0.15 * 0.30 * 0.50 * x = 99) → x = 4400 := by
  sorry

end percentage_problem_l600_60077


namespace min_distance_to_line_l600_60058

open Complex

theorem min_distance_to_line (z : ℂ) (h : abs (z - 1) = abs (z + 2*I)) :
  ∃ (min_val : ℝ), min_val = (9 * Real.sqrt 5) / 10 ∧
  ∀ (w : ℂ), abs (w - 1) = abs (w + 2*I) → abs (w - 1 - I) ≥ min_val :=
by sorry

end min_distance_to_line_l600_60058


namespace object_crosses_x_axis_l600_60043

/-- The position vector of an object moving in two dimensions -/
def position_vector (t : ℝ) : ℝ × ℝ :=
  (4 * t^2 - 9, 2 * t - 5)

/-- The time when the object crosses the x-axis -/
def crossing_time : ℝ := 2.5

/-- Theorem: The object crosses the x-axis at t = 2.5 seconds -/
theorem object_crosses_x_axis :
  (position_vector crossing_time).2 = 0 := by
  sorry

end object_crosses_x_axis_l600_60043


namespace tournament_prize_interval_l600_60006

def total_prize : ℕ := 4800
def first_place_prize : ℕ := 2000

def prize_interval (x : ℕ) : Prop :=
  first_place_prize + (first_place_prize - x) + (first_place_prize - 2*x) = total_prize

theorem tournament_prize_interval : ∃ (x : ℕ), prize_interval x ∧ x = 400 := by
  sorry

end tournament_prize_interval_l600_60006


namespace wage_cut_and_raise_l600_60061

theorem wage_cut_and_raise (original_wage : ℝ) (h : original_wage > 0) :
  let reduced_wage := 0.8 * original_wage
  let raised_wage := reduced_wage * 1.25
  raised_wage = original_wage :=
by sorry

end wage_cut_and_raise_l600_60061


namespace unique_valid_number_l600_60071

def is_valid_number (n : ℕ) : Prop :=
  let x := n / 100
  let y := (n / 10) % 10
  let z := n % 10
  (100 * x + 10 * z + y = 64 * x + 8 * z + y) ∧ 
  (100 * y + 10 * x + z = 36 * y + 6 * x + z - 16) ∧
  (100 * z + 10 * y + x = 16 * z + 4 * y + x + 18)

theorem unique_valid_number : 
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧ is_valid_number n :=
by sorry

end unique_valid_number_l600_60071


namespace kristin_reading_time_l600_60003

/-- Given that Peter reads one book in 18 hours and can read three times as fast as Kristin,
    prove that Kristin will take 540 hours to read half of her 20 books. -/
theorem kristin_reading_time 
  (peter_time : ℝ) 
  (peter_speed : ℝ) 
  (kristin_books : ℕ) 
  (h1 : peter_time = 18) 
  (h2 : peter_speed = 3) 
  (h3 : kristin_books = 20) : 
  (kristin_books / 2 : ℝ) * (peter_time * peter_speed) = 540 := by
  sorry

end kristin_reading_time_l600_60003


namespace arithmetic_sequence_first_term_l600_60030

theorem arithmetic_sequence_first_term
  (a d : ℚ)  -- First term and common difference
  (sum_60 : ℚ → ℚ → ℕ → ℚ)  -- Function to calculate sum of n terms
  (h1 : sum_60 a d 60 = 240)  -- Sum of first 60 terms
  (h2 : sum_60 (a + 60 * d) d 60 = 3240)  -- Sum of next 60 terms
  : a = -247 / 12 :=
by sorry

end arithmetic_sequence_first_term_l600_60030


namespace range_of_a_l600_60050

/-- The function f(x) = x^3 + ax^2 - a^2x -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 - a^2*x

/-- The derivative of f(x) -/
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x - a^2

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Ioo 0 1, f_deriv a x ≤ 0) ∧
  (∀ x ∈ Set.Ioi 2, f_deriv a x ≥ 0) →
  a ∈ Set.Icc (-2) (-1) ∪ Set.Icc 3 6 :=
sorry

end range_of_a_l600_60050


namespace money_redistribution_theorem_l600_60038

/-- Represents the money redistribution problem among boys and girls -/
theorem money_redistribution_theorem 
  (num_boys : ℕ) 
  (num_girls : ℕ) 
  (boy_initial : ℕ) 
  (girl_initial : ℕ) : 
  num_boys = 9 → 
  num_girls = 3 → 
  boy_initial = 12 → 
  girl_initial = 36 → 
  ∃ (boy_gives girl_gives final_amount : ℕ), 
    (∀ (b : ℕ), b < num_boys → 
      boy_initial - num_girls * boy_gives + num_girls * girl_gives = final_amount) ∧
    (∀ (g : ℕ), g < num_girls → 
      girl_initial - num_boys * girl_gives + num_boys * boy_gives = final_amount) := by
  sorry

end money_redistribution_theorem_l600_60038


namespace inverse_inequality_for_negatives_l600_60056

theorem inverse_inequality_for_negatives (a b : ℝ) : 0 > a → a > b → (1 / a) < (1 / b) := by
  sorry

end inverse_inequality_for_negatives_l600_60056


namespace total_different_groups_l600_60063

-- Define the number of marbles of each color
def red_marbles : ℕ := 1
def green_marbles : ℕ := 1
def blue_marbles : ℕ := 1
def yellow_marbles : ℕ := 3

-- Define the total number of distinct colors
def distinct_colors : ℕ := 4

-- Define the function to calculate the number of different groups
def different_groups : ℕ :=
  -- Groups with two yellow marbles
  1 +
  -- Groups with two different colors
  (distinct_colors.choose 2)

-- Theorem statement
theorem total_different_groups :
  different_groups = 7 :=
sorry

end total_different_groups_l600_60063


namespace arithmetic_sequence_product_l600_60060

/-- An increasing arithmetic sequence of integers -/
def ArithmeticSequence (b : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, d > 0 ∧ ∀ n : ℕ, b (n + 1) = b n + d

theorem arithmetic_sequence_product (b : ℕ → ℤ) :
  ArithmeticSequence b → b 4 * b 5 = 21 → b 3 * b 6 = -779 ∨ b 3 * b 6 = -11 :=
by sorry

end arithmetic_sequence_product_l600_60060


namespace scaling_transforms_line_l600_60044

-- Define the original line
def original_line (x y : ℝ) : Prop := x - 2*y = 2

-- Define the transformed line
def transformed_line (x' y' : ℝ) : Prop := 2*x' - y' = 4

-- Define the scaling transformation
def scaling_transformation (x y x' y' : ℝ) : Prop := x' = x ∧ y' = 4*y

-- Theorem statement
theorem scaling_transforms_line :
  ∀ (x y x' y' : ℝ),
    original_line x y →
    scaling_transformation x y x' y' →
    transformed_line x' y' := by
  sorry

end scaling_transforms_line_l600_60044


namespace weight_of_replaced_person_l600_60023

/-- Given a group of 8 persons, if replacing one person with a new person weighing 94 kg
    increases the average weight by 3 kg, then the weight of the replaced person is 70 kg. -/
theorem weight_of_replaced_person
  (original_count : ℕ)
  (weight_increase : ℝ)
  (new_person_weight : ℝ)
  (h1 : original_count = 8)
  (h2 : weight_increase = 3)
  (h3 : new_person_weight = 94)
  : ℝ :=
by
  sorry

#check weight_of_replaced_person

end weight_of_replaced_person_l600_60023


namespace optimal_betting_strategy_l600_60049

def num_boxes : ℕ := 100

-- The maximum factor for exactly one blue cube
def max_factor_one_blue (n : ℕ) : ℚ :=
  (2 ^ n : ℚ) / n

-- The maximum factor for at least two blue cubes
def max_factor_two_plus_blue (n : ℕ) : ℚ :=
  (2 ^ n : ℚ) / ((2 ^ n : ℚ) - (n + 1 : ℚ))

theorem optimal_betting_strategy :
  (max_factor_one_blue num_boxes = (2 ^ 98 : ℚ) / 25) ∧
  (max_factor_two_plus_blue num_boxes = (2 ^ 100 : ℚ) / ((2 ^ 100 : ℚ) - 101)) :=
by sorry

end optimal_betting_strategy_l600_60049


namespace circle_tangency_radius_l600_60084

theorem circle_tangency_radius 
  (d1 d2 r1 r2 r y : ℝ) 
  (h1 : d1 < d2) 
  (h2 : r1 = d1 / 2) 
  (h3 : r2 = d2 / 2) 
  (h4 : (r + r1)^2 = (r - 2*r2 - r1)^2 + y^2) 
  (h5 : (r + r2)^2 = (r - r2)^2 + y^2) : 
  r = ((d1 + d2) * d2) / (2 * d1) := by
sorry

end circle_tangency_radius_l600_60084


namespace range_of_a_for_fourth_quadrant_l600_60002

/-- A point in the fourth quadrant has a positive x-coordinate and a negative y-coordinate -/
def is_in_fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

/-- The x-coordinate of point P -/
def x_coord (a : ℝ) : ℝ := a + 1

/-- The y-coordinate of point P -/
def y_coord (a : ℝ) : ℝ := 2 * a - 3

/-- The theorem stating the range of a for point P to be in the fourth quadrant -/
theorem range_of_a_for_fourth_quadrant :
  ∀ a : ℝ, is_in_fourth_quadrant (x_coord a) (y_coord a) ↔ -1 < a ∧ a < 3/2 :=
sorry

end range_of_a_for_fourth_quadrant_l600_60002


namespace johnny_hourly_wage_l600_60035

/-- Johnny's hourly wage calculation -/
theorem johnny_hourly_wage :
  let hours_worked : ℝ := 6
  let total_earnings : ℝ := 28.5
  let hourly_wage := total_earnings / hours_worked
  hourly_wage = 4.75 := by
sorry

end johnny_hourly_wage_l600_60035


namespace binomial_coefficient_n_n_l600_60040

theorem binomial_coefficient_n_n (n : ℕ) : Nat.choose n n = 1 := by
  sorry

end binomial_coefficient_n_n_l600_60040


namespace floor_paving_cost_l600_60064

-- Define the room dimensions
def room_length : ℝ := 6
def room_width : ℝ := 4.75

-- Define the cost per square meter
def cost_per_sqm : ℝ := 900

-- Define the function to calculate the area of a rectangle
def area (length width : ℝ) : ℝ := length * width

-- Define the function to calculate the total cost
def total_cost (length width cost_per_sqm : ℝ) : ℝ :=
  area length width * cost_per_sqm

-- State the theorem
theorem floor_paving_cost :
  total_cost room_length room_width cost_per_sqm = 25650 := by sorry

end floor_paving_cost_l600_60064


namespace pool_tiles_l600_60051

theorem pool_tiles (blue_tiles : ℕ) (red_tiles : ℕ) (additional_tiles : ℕ) : 
  blue_tiles = 48 → red_tiles = 32 → additional_tiles = 20 →
  blue_tiles + red_tiles + additional_tiles = 100 := by
  sorry

end pool_tiles_l600_60051


namespace modular_inverse_87_mod_88_l600_60076

theorem modular_inverse_87_mod_88 : ∃ x : ℤ, 0 ≤ x ∧ x < 88 ∧ (87 * x) % 88 = 1 :=
by
  use 87
  sorry

end modular_inverse_87_mod_88_l600_60076


namespace lcm_18_30_l600_60097

theorem lcm_18_30 : Nat.lcm 18 30 = 90 := by
  sorry

end lcm_18_30_l600_60097


namespace midpoint_intersection_l600_60072

/-- Given a line segment from (1,3) to (5,11), if the line x + y = b
    intersects this segment at its midpoint, then b = 10. -/
theorem midpoint_intersection (b : ℝ) : 
  (∃ (x y : ℝ), x + y = b ∧ 
   x = (1 + 5) / 2 ∧ 
   y = (3 + 11) / 2) → 
  b = 10 := by
sorry

end midpoint_intersection_l600_60072


namespace root_difference_product_l600_60066

theorem root_difference_product (p q a b c : ℝ) : 
  (a^2 + p*a + 1 = 0) → 
  (b^2 + p*b + 1 = 0) → 
  (b^2 + q*b + 2 = 0) → 
  (c^2 + q*c + 2 = 0) → 
  (b - a) * (b - c) = p*q - 6 := by
sorry

end root_difference_product_l600_60066


namespace simplify_expression_l600_60073

theorem simplify_expression : ∃ (a b c : ℕ+), 
  (Real.sqrt 3 + 2 / Real.sqrt 3 + Real.sqrt 11 + 3 / Real.sqrt 11 = (a * Real.sqrt 3 + b * Real.sqrt 11) / c) ∧
  (∀ (a' b' c' : ℕ+), 
    Real.sqrt 3 + 2 / Real.sqrt 3 + Real.sqrt 11 + 3 / Real.sqrt 11 = (a' * Real.sqrt 3 + b' * Real.sqrt 11) / c' →
    c ≤ c') ∧
  a = 39 ∧ b = 42 ∧ c = 33 := by
sorry

end simplify_expression_l600_60073


namespace equilateral_triangle_from_polynomial_roots_l600_60085

theorem equilateral_triangle_from_polynomial_roots (a b c : ℂ) :
  (∀ z : ℂ, z^3 + 5*z + 7 = 0 ↔ z = a ∨ z = b ∨ z = c) →
  Complex.abs a ^ 2 + Complex.abs b ^ 2 + Complex.abs c ^ 2 = 300 →
  Complex.abs (a - b) = Complex.abs (b - c) ∧ 
  Complex.abs (b - c) = Complex.abs (c - a) →
  (Complex.abs (a - b))^2 = 225 :=
by sorry

end equilateral_triangle_from_polynomial_roots_l600_60085


namespace factorization_of_expression_l600_60083

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

axiom natural_prime_factorization : ∀ n : ℕ, n > 1 → ∃ (primes : List ℕ), (∀ p ∈ primes, is_prime p) ∧ n = primes.prod

theorem factorization_of_expression : 2^4 * 3^2 - 1 = 11 * 13 := by sorry

end factorization_of_expression_l600_60083


namespace beverly_bottle_caps_l600_60027

/-- The number of bottle caps in Beverly's collection -/
def total_bottle_caps (small_box_caps : ℕ) (large_box_caps : ℕ) 
                      (small_boxes : ℕ) (large_boxes : ℕ) 
                      (individual_caps : ℕ) : ℕ :=
  small_box_caps * small_boxes + large_box_caps * large_boxes + individual_caps

/-- Theorem stating the total number of bottle caps in Beverly's collection -/
theorem beverly_bottle_caps : 
  total_bottle_caps 35 75 7 3 23 = 493 := by
  sorry

end beverly_bottle_caps_l600_60027


namespace matrix_cube_l600_60070

def A : Matrix (Fin 2) (Fin 2) ℤ := !![2, -1; 1, 1]

theorem matrix_cube :
  A ^ 3 = !![3, -6; 6, -3] := by sorry

end matrix_cube_l600_60070


namespace toothpaste_cost_is_three_l600_60091

/-- Represents the shopping scenario with given conditions -/
structure Shopping where
  budget : ℕ
  showerGelPrice : ℕ
  showerGelCount : ℕ
  laundryDetergentPrice : ℕ
  remaining : ℕ

/-- Calculates the cost of toothpaste given the shopping conditions -/
def toothpasteCost (s : Shopping) : ℕ :=
  s.budget - s.remaining - (s.showerGelPrice * s.showerGelCount) - s.laundryDetergentPrice

/-- Theorem stating that the toothpaste costs $3 under the given conditions -/
theorem toothpaste_cost_is_three :
  let s : Shopping := {
    budget := 60,
    showerGelPrice := 4,
    showerGelCount := 4,
    laundryDetergentPrice := 11,
    remaining := 30
  }
  toothpasteCost s = 3 := by sorry

end toothpaste_cost_is_three_l600_60091


namespace arithmetic_sequence_problem_l600_60011

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- Given an arithmetic sequence a_n where a_1 + 3a_8 + a_15 = 120, 
    prove that 2a_6 - a_4 = 24 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
    (h_arith : ArithmeticSequence a) 
    (h_sum : a 1 + 3 * a 8 + a 15 = 120) : 
    2 * a 6 - a 4 = 24 := by
  sorry

end arithmetic_sequence_problem_l600_60011


namespace abs_product_zero_implies_one_equal_one_l600_60037

theorem abs_product_zero_implies_one_equal_one (a b : ℝ) :
  |a - 1| * |b - 1| = 0 → a = 1 ∨ b = 1 := by
sorry

end abs_product_zero_implies_one_equal_one_l600_60037


namespace estimated_y_value_l600_60000

/-- Linear regression equation -/
def linear_regression (x : ℝ) : ℝ := 0.50 * x - 0.81

theorem estimated_y_value (x : ℝ) (h : x = 25) : linear_regression x = 11.69 := by
  sorry

end estimated_y_value_l600_60000


namespace no_solution_to_inequalities_l600_60057

theorem no_solution_to_inequalities :
  ¬∃ x : ℝ, (3 * x - 2 < (x + 2)^2) ∧ ((x + 2)^2 < 8 * x - 5) := by
sorry

end no_solution_to_inequalities_l600_60057


namespace ratio_sum_problem_l600_60086

theorem ratio_sum_problem (a b c : ℝ) (h1 : a / b = 2 / 3) (h2 : b / c = 3 / 4) (h3 : a * b + b * c + c * a = 13) : b * c = 6 := by
  sorry

end ratio_sum_problem_l600_60086


namespace average_speed_two_hours_car_average_speed_l600_60093

/-- The average speed of a car traveling different distances in two hours -/
theorem average_speed_two_hours (d1 d2 : ℝ) : 
  d1 > 0 → d2 > 0 → (d1 + d2) / 2 = (d1 + d2) / 2 := by
  sorry

/-- The average speed of a car traveling 50 km in the first hour and 60 km in the second hour is 55 km/h -/
theorem car_average_speed : (50 + 60) / 2 = 55 := by
  sorry

end average_speed_two_hours_car_average_speed_l600_60093


namespace square_perimeter_9cm_l600_60096

/-- Calculates the perimeter of a square given its side length -/
def square_perimeter (side_length : ℝ) : ℝ := 4 * side_length

/-- Theorem: The perimeter of a square with side length 9 cm is 36 cm -/
theorem square_perimeter_9cm : square_perimeter 9 = 36 := by
  sorry

end square_perimeter_9cm_l600_60096


namespace point_B_value_l600_60062

def point_A : ℝ := -1

def distance_AB : ℝ := 4

theorem point_B_value : 
  ∃ (B : ℝ), (B = 3 ∨ B = -5) ∧ |B - point_A| = distance_AB :=
by sorry

end point_B_value_l600_60062


namespace coffee_cups_total_l600_60039

theorem coffee_cups_total (sandra_cups marcie_cups : ℕ) 
  (h1 : sandra_cups = 6) 
  (h2 : marcie_cups = 2) : 
  sandra_cups + marcie_cups = 8 := by
sorry

end coffee_cups_total_l600_60039


namespace square_perimeter_increase_l600_60017

theorem square_perimeter_increase (x : ℝ) (h : x > 0) :
  let original_side := x / 4
  let new_perimeter := 4 * x
  let new_side := new_perimeter / 4
  new_side / original_side = 4 := by sorry

end square_perimeter_increase_l600_60017


namespace solution_set_f_gt_7_minus_x_range_of_m_for_f_leq_abs_3m_minus_2_l600_60089

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 2| + |x - 3|

-- Theorem for part (Ⅰ)
theorem solution_set_f_gt_7_minus_x :
  {x : ℝ | f x > 7 - x} = {x : ℝ | x < -6 ∨ x > 2} := by sorry

-- Theorem for part (Ⅱ)
theorem range_of_m_for_f_leq_abs_3m_minus_2 :
  {m : ℝ | ∃ x, f x ≤ |3*m - 2|} = {m : ℝ | m ≤ -1 ∨ m ≥ 7/3} := by sorry

end solution_set_f_gt_7_minus_x_range_of_m_for_f_leq_abs_3m_minus_2_l600_60089


namespace expression_is_negative_l600_60074

theorem expression_is_negative : 
  Real.sqrt (25 * Real.sqrt 7 - 27 * Real.sqrt 6) - Real.sqrt (17 * Real.sqrt 5 - 38) < 0 := by
  sorry

end expression_is_negative_l600_60074


namespace box_product_digits_l600_60020

def box_product (n : ℕ) : ℕ := n * 100 + 28 * 4

theorem box_product_digits :
  (∀ n : ℕ, n ≤ 2 → box_product n < 1000) ∧
  (∀ n : ℕ, n ≥ 3 → box_product n ≥ 1000) :=
by sorry

end box_product_digits_l600_60020


namespace negation_of_forall_inequality_l600_60031

theorem negation_of_forall_inequality :
  (¬ ∀ x : ℝ, x^2 + 1 ≥ 2*x) ↔ (∃ x : ℝ, x^2 + 1 < 2*x) := by sorry

end negation_of_forall_inequality_l600_60031


namespace polynomial_divisibility_l600_60080

/-- A complex number ω such that ω^2 + ω + 1 = 0 -/
def ω : ℂ := sorry

/-- The polynomial x^11 + Ax^2 + B -/
def P (A B : ℝ) (x : ℂ) : ℂ := x^11 + A * x^2 + B

/-- The polynomial x^2 + x + 1 -/
def Q (x : ℂ) : ℂ := x^2 + x + 1

theorem polynomial_divisibility (A B : ℝ) :
  (∀ x, Q x = 0 → P A B x = 0) → A = -1 ∧ B = 0 := by sorry

end polynomial_divisibility_l600_60080
