import Mathlib

namespace NUMINAMATH_CALUDE_cube_surface_area_equal_volume_l1775_177547

/-- The surface area of a cube with volume equal to a 9x3x27 inch rectangular prism is 486 square inches. -/
theorem cube_surface_area_equal_volume (l w h : ℝ) (cube_edge : ℝ) : 
  l = 9 ∧ w = 3 ∧ h = 27 →
  cube_edge ^ 3 = l * w * h →
  6 * cube_edge ^ 2 = 486 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_equal_volume_l1775_177547


namespace NUMINAMATH_CALUDE_ms_hatcher_students_l1775_177536

theorem ms_hatcher_students (third_graders : ℕ) (fourth_graders : ℕ) (fifth_graders : ℕ) : 
  third_graders = 20 →
  fourth_graders = 2 * third_graders →
  fifth_graders = third_graders / 2 →
  third_graders + fourth_graders + fifth_graders = 70 := by
  sorry

end NUMINAMATH_CALUDE_ms_hatcher_students_l1775_177536


namespace NUMINAMATH_CALUDE_line_through_points_l1775_177530

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line defined by two other points -/
def lies_on_line (p : Point) (p1 : Point) (p2 : Point) : Prop :=
  (p.y - p1.y) * (p2.x - p1.x) = (p2.y - p1.y) * (p.x - p1.x)

theorem line_through_points : 
  let p1 : Point := ⟨8, 16⟩
  let p2 : Point := ⟨2, -2⟩
  let p3 : Point := ⟨5, 7⟩
  let p4 : Point := ⟨4, 4⟩
  let p5 : Point := ⟨10, 22⟩
  let p6 : Point := ⟨-2, -12⟩
  let p7 : Point := ⟨1, -5⟩
  lies_on_line p3 p1 p2 ∧
  lies_on_line p4 p1 p2 ∧
  lies_on_line p5 p1 p2 ∧
  lies_on_line p7 p1 p2 ∧
  ¬ lies_on_line p6 p1 p2 :=
by
  sorry


end NUMINAMATH_CALUDE_line_through_points_l1775_177530


namespace NUMINAMATH_CALUDE_interest_rate_proof_l1775_177535

/-- Represents the annual interest rate as a real number between 0 and 1 -/
def annual_interest_rate : ℝ := 0.05

/-- The initial principal amount in rupees -/
def principal : ℝ := 4800

/-- The final amount after 2 years in rupees -/
def final_amount : ℝ := 5292

/-- The number of years the money is invested -/
def time : ℕ := 2

/-- The number of times interest is compounded per year -/
def compounds_per_year : ℕ := 1

theorem interest_rate_proof :
  final_amount = principal * (1 + annual_interest_rate) ^ (compounds_per_year * time) :=
sorry

end NUMINAMATH_CALUDE_interest_rate_proof_l1775_177535


namespace NUMINAMATH_CALUDE_cake_serving_capacity_l1775_177501

-- Define the original cake properties
def original_radius : ℝ := 20
def original_people_served : ℕ := 4

-- Define the new cake radius
def new_radius : ℝ := 50

-- Theorem statement
theorem cake_serving_capacity :
  ∃ (new_people_served : ℕ), 
    new_people_served = 25 ∧
    (new_radius^2 / original_radius^2) * original_people_served = new_people_served :=
by
  sorry

end NUMINAMATH_CALUDE_cake_serving_capacity_l1775_177501


namespace NUMINAMATH_CALUDE_marble_ratio_theorem_l1775_177505

/-- Represents the number of marbles Elsa has at different points in the day -/
structure MarbleCount where
  initial : ℕ
  after_breakfast : ℕ
  after_lunch : ℕ
  after_mom_purchase : ℕ
  final : ℕ

/-- Represents the marble transactions throughout the day -/
structure MarbleTransactions where
  lost_at_breakfast : ℕ
  given_to_susie : ℕ
  bought_by_mom : ℕ

/-- Theorem stating the ratio of marbles Susie gave back to Elsa to the marbles Elsa gave to Susie -/
theorem marble_ratio_theorem (m : MarbleCount) (t : MarbleTransactions) : 
  m.initial = 40 →
  t.lost_at_breakfast = 3 →
  t.given_to_susie = 5 →
  t.bought_by_mom = 12 →
  m.final = 54 →
  m.after_breakfast = m.initial - t.lost_at_breakfast →
  m.after_lunch = m.after_breakfast - t.given_to_susie →
  m.after_mom_purchase = m.after_lunch + t.bought_by_mom →
  (m.final - m.after_mom_purchase) / t.given_to_susie = 2 :=
by sorry

end NUMINAMATH_CALUDE_marble_ratio_theorem_l1775_177505


namespace NUMINAMATH_CALUDE_simplify_expression_l1775_177540

theorem simplify_expression (z : ℝ) : (3 - 5 * z^2) - (4 + 3 * z^2) = -1 - 8 * z^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1775_177540


namespace NUMINAMATH_CALUDE_cones_from_twelve_cylinders_l1775_177517

/-- The number of cones that can be cast from a given number of cylinders -/
def cones_from_cylinders (num_cylinders : ℕ) : ℕ :=
  3 * num_cylinders

/-- The volume ratio between a cylinder and a cone with the same base and height -/
def cylinder_cone_volume_ratio : ℕ := 3

theorem cones_from_twelve_cylinders :
  cones_from_cylinders 12 = 36 :=
by sorry

end NUMINAMATH_CALUDE_cones_from_twelve_cylinders_l1775_177517


namespace NUMINAMATH_CALUDE_function_inequality_implies_a_range_l1775_177533

/-- An even function that is decreasing on the non-negative reals -/
def EvenDecreasingFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = f (-x)) ∧ 
  (∀ x y, 0 ≤ x → x ≤ y → f y ≤ f x)

/-- The inequality condition from the problem -/
def InequalityCondition (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, 0 < x → x ≤ Real.sqrt 2 → 
    f (-a * x + x^3 + 1) + f (a * x - x^3 - 1) ≥ 2 * f 1

theorem function_inequality_implies_a_range 
  (f : ℝ → ℝ) (a : ℝ) 
  (hf : EvenDecreasingFunction f) 
  (h_ineq : InequalityCondition f a) : 
  2 ≤ a ∧ a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_implies_a_range_l1775_177533


namespace NUMINAMATH_CALUDE_shaded_area_is_nine_eighths_pi_l1775_177594

/-- Represents a right triangle with circles at its vertices and an additional circle --/
structure TriangleWithCircles where
  -- Side lengths of the right triangle
  ac : ℝ
  ab : ℝ
  bc : ℝ
  -- Radius of circles at triangle vertices
  y : ℝ
  -- Radius of circle P
  x : ℝ
  -- Conditions
  right_triangle : ac^2 + ab^2 = bc^2
  side_ac : y + 2*x + y = ac
  side_ab : y + 4*x + y = ab
  area_ratio : (2*x)^2 = 4*x^2

/-- The shaded area in the triangle configuration is 9π/8 square units --/
theorem shaded_area_is_nine_eighths_pi (t : TriangleWithCircles)
  (h1 : t.ac = 3)
  (h2 : t.ab = 4)
  (h3 : t.bc = 5) :
  3 * (π * t.y^2 / 2) + π * t.x^2 / 2 = 9 * π / 8 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_is_nine_eighths_pi_l1775_177594


namespace NUMINAMATH_CALUDE_prob_red_then_black_our_deck_l1775_177523

/-- A customized deck of cards -/
structure CustomDeck :=
  (total_cards : ℕ)
  (red_cards : ℕ)
  (black_cards : ℕ)

/-- The probability of drawing a red card first and a black card second -/
def prob_red_then_black (deck : CustomDeck) : ℚ :=
  (deck.red_cards : ℚ) * (deck.black_cards : ℚ) / ((deck.total_cards : ℚ) * (deck.total_cards - 1 : ℚ))

/-- Our specific deck -/
def our_deck : CustomDeck :=
  { total_cards := 78
  , red_cards := 39
  , black_cards := 39 }

theorem prob_red_then_black_our_deck :
  prob_red_then_black our_deck = 507 / 2002 := by
  sorry

end NUMINAMATH_CALUDE_prob_red_then_black_our_deck_l1775_177523


namespace NUMINAMATH_CALUDE_field_area_is_625_l1775_177539

/-- Represents a square field -/
structure SquareField where
  /-- The length of one side of the square field in kilometers -/
  side : ℝ
  /-- The side length is positive -/
  side_pos : side > 0

/-- Calculates the perimeter of a square field -/
def perimeter (field : SquareField) : ℝ := 4 * field.side

/-- Calculates the area of a square field -/
def area (field : SquareField) : ℝ := field.side ^ 2

/-- The speed of the horse in km/h -/
def horse_speed : ℝ := 25

/-- The time taken by the horse to run around the field in hours -/
def lap_time : ℝ := 4

theorem field_area_is_625 (field : SquareField) 
  (h : perimeter field = horse_speed * lap_time) : 
  area field = 625 := by
  sorry

end NUMINAMATH_CALUDE_field_area_is_625_l1775_177539


namespace NUMINAMATH_CALUDE_probability_even_product_excluding_13_l1775_177529

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

def valid_integer (n : ℕ) : Prop :=
  4 ≤ n ∧ n ≤ 20 ∧ n ≠ 13

def count_valid_integers : ℕ := 16

def count_even_valid_integers : ℕ := 9

def count_odd_valid_integers : ℕ := 7

def total_combinations : ℕ := count_valid_integers.choose 2

def even_product_combinations : ℕ := 
  count_even_valid_integers.choose 2 + count_even_valid_integers * count_odd_valid_integers

theorem probability_even_product_excluding_13 :
  (even_product_combinations : ℚ) / total_combinations = 33 / 40 := by sorry

end NUMINAMATH_CALUDE_probability_even_product_excluding_13_l1775_177529


namespace NUMINAMATH_CALUDE_inequality_proof_l1775_177506

theorem inequality_proof (a b : ℝ) (n : ℕ) (ha : 0 < a) (hb : 0 < b) :
  (1 + a / b) ^ n + (1 + b / a) ^ n ≥ 2^(n + 1) ∧
  ((1 + a / b) ^ n + (1 + b / a) ^ n = 2^(n + 1) ↔ a = b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1775_177506


namespace NUMINAMATH_CALUDE_max_value_sum_fractions_l1775_177553

theorem max_value_sum_fractions (a b c : ℝ) 
  (h_nonneg_a : a ≥ 0) (h_nonneg_b : b ≥ 0) (h_nonneg_c : c ≥ 0)
  (h_sum : a + b + c = 1) :
  (a * b) / (a + b) + (a * c) / (a + c) + (b * c) / (b + c) ≤ 1 / 2 :=
sorry

end NUMINAMATH_CALUDE_max_value_sum_fractions_l1775_177553


namespace NUMINAMATH_CALUDE_xy_value_l1775_177525

theorem xy_value (x y : ℝ) 
  (h1 : (8:ℝ)^x / (4:ℝ)^(x+y) = 64)
  (h2 : (27:ℝ)^(x+y) / (9:ℝ)^(6*y) = 81) :
  x * y = 644 / 9 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l1775_177525


namespace NUMINAMATH_CALUDE_sum_of_digits_power_product_l1775_177580

def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

theorem sum_of_digits_power_product :
  sumOfDigits (2^2010 * 5^2012 * 7) = 13 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_power_product_l1775_177580


namespace NUMINAMATH_CALUDE_new_job_wage_is_15_l1775_177583

/-- Represents the wage scenario for Maisy's job options -/
structure WageScenario where
  current_hours : ℕ
  current_wage : ℕ
  new_hours : ℕ
  new_bonus : ℕ
  earnings_difference : ℕ

/-- Calculates the wage per hour for the new job -/
def new_job_wage (scenario : WageScenario) : ℕ :=
  (scenario.current_hours * scenario.current_wage + scenario.earnings_difference - scenario.new_bonus) / scenario.new_hours

/-- Theorem stating that given the specified conditions, the new job wage is $15 per hour -/
theorem new_job_wage_is_15 (scenario : WageScenario) 
  (h1 : scenario.current_hours = 8)
  (h2 : scenario.current_wage = 10)
  (h3 : scenario.new_hours = 4)
  (h4 : scenario.new_bonus = 35)
  (h5 : scenario.earnings_difference = 15) :
  new_job_wage scenario = 15 := by
  sorry

#eval new_job_wage { current_hours := 8, current_wage := 10, new_hours := 4, new_bonus := 35, earnings_difference := 15 }

end NUMINAMATH_CALUDE_new_job_wage_is_15_l1775_177583


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l1775_177531

theorem complex_fraction_equality : (5 : ℂ) / (2 - I) = 2 + I := by sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l1775_177531


namespace NUMINAMATH_CALUDE_binomial_expansion_terms_l1775_177581

theorem binomial_expansion_terms (x a : ℝ) (n : ℕ) : 
  (Nat.choose n 1 : ℝ) * x^(n-1) * a = 56 ∧
  (Nat.choose n 2 : ℝ) * x^(n-2) * a^2 = 168 ∧
  (Nat.choose n 3 : ℝ) * x^(n-3) * a^3 = 336 →
  n = 3 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_terms_l1775_177581


namespace NUMINAMATH_CALUDE_coconut_grove_yield_is_120_l1775_177514

/-- Represents the yield of x trees in a coconut grove with specific conditions -/
def coconut_grove_yield (x : ℕ) (yield_x : ℕ) : Prop :=
  let yield_xplus2 : ℕ := 30 * (x + 2)
  let yield_xminus2 : ℕ := 180 * (x - 2)
  let total_trees : ℕ := (x + 2) + x + (x - 2)
  let total_yield : ℕ := yield_xplus2 + (x * yield_x) + yield_xminus2
  (total_yield = total_trees * 100) ∧ (x = 10)

theorem coconut_grove_yield_is_120 :
  coconut_grove_yield 10 120 := by sorry

end NUMINAMATH_CALUDE_coconut_grove_yield_is_120_l1775_177514


namespace NUMINAMATH_CALUDE_rectangular_solid_surface_area_l1775_177543

/-- A rectangular solid with prime edge lengths and volume 143 has surface area 382 -/
theorem rectangular_solid_surface_area : ∀ a b c : ℕ,
  Prime a → Prime b → Prime c →
  a * b * c = 143 →
  2 * (a * b + b * c + c * a) = 382 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_surface_area_l1775_177543


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1775_177524

theorem min_value_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (x / Real.sqrt (1 - x)) + (y / Real.sqrt (1 - y)) ≥ Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1775_177524


namespace NUMINAMATH_CALUDE_lcm_product_geq_lcm_square_l1775_177576

theorem lcm_product_geq_lcm_square (k m n : ℕ) :
  Nat.lcm (Nat.lcm k m) n * Nat.lcm (Nat.lcm m n) k * Nat.lcm (Nat.lcm n k) m ≥ (Nat.lcm (Nat.lcm k m) n)^2 := by
  sorry

end NUMINAMATH_CALUDE_lcm_product_geq_lcm_square_l1775_177576


namespace NUMINAMATH_CALUDE_f_range_theorem_l1775_177578

-- Define the function f
def f (x y z : ℝ) := (z - x) * (z - y)

-- State the theorem
theorem f_range_theorem (x y z : ℝ) 
  (h1 : x + y + z = 1) 
  (h2 : x ≥ 0) 
  (h3 : y ≥ 0) 
  (h4 : z ≥ 0) :
  ∃ (w : ℝ), f x y z = w ∧ w ∈ Set.Icc (-1/8 : ℝ) 1 := by
  sorry

end NUMINAMATH_CALUDE_f_range_theorem_l1775_177578


namespace NUMINAMATH_CALUDE_sin_plus_sqrt3_cos_l1775_177518

theorem sin_plus_sqrt3_cos (x : ℝ) (h1 : x ∈ Set.Ioo 0 (π/2)) 
  (h2 : Real.cos (x + π/12) = Real.sqrt 2 / 10) : 
  Real.sin x + Real.sqrt 3 * Real.cos x = 8/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_plus_sqrt3_cos_l1775_177518


namespace NUMINAMATH_CALUDE_comparison_theorems_l1775_177500

theorem comparison_theorems :
  (∀ a b : ℝ, a - b = 4 → a > b) ∧
  (∀ a b : ℝ, a - b = -2 → a < b) ∧
  (∀ x : ℝ, x > 0 → -x + 5 > -2*x + 4) ∧
  (∀ x y : ℝ, 
    (y > x → 5*x + 13*y + 2 > 6*x + 12*y + 2) ∧
    (y = x → 5*x + 13*y + 2 = 6*x + 12*y + 2) ∧
    (y < x → 5*x + 13*y + 2 < 6*x + 12*y + 2)) :=
by sorry

end NUMINAMATH_CALUDE_comparison_theorems_l1775_177500


namespace NUMINAMATH_CALUDE_square_root_of_four_l1775_177563

theorem square_root_of_four : 
  {x : ℝ | x^2 = 4} = {2, -2} := by sorry

end NUMINAMATH_CALUDE_square_root_of_four_l1775_177563


namespace NUMINAMATH_CALUDE_extended_quadrilateral_area_l1775_177503

/-- Represents a quadrilateral with extended sides -/
structure ExtendedQuadrilateral where
  -- Original quadrilateral sides
  ab : ℝ
  bc : ℝ
  cd : ℝ
  da : ℝ
  -- Extended sides
  bb' : ℝ
  cc' : ℝ
  dd' : ℝ
  aa' : ℝ
  -- Area of original quadrilateral
  area : ℝ
  -- Conditions
  ab_eq : ab = 5
  bc_eq : bc = 8
  cd_eq : cd = 4
  da_eq : da = 7
  bb'_eq : bb' = 1.5 * ab
  cc'_eq : cc' = 1.5 * bc
  dd'_eq : dd' = 1.5 * cd
  aa'_eq : aa' = 1.5 * da
  area_eq : area = 20

/-- The area of the extended quadrilateral A'B'C'D' is 140 -/
theorem extended_quadrilateral_area (q : ExtendedQuadrilateral) :
  q.area + (q.bb' - q.ab) * q.ab / 2 + (q.cc' - q.bc) * q.bc / 2 +
  (q.dd' - q.cd) * q.cd / 2 + (q.aa' - q.da) * q.da / 2 = 140 := by
  sorry

end NUMINAMATH_CALUDE_extended_quadrilateral_area_l1775_177503


namespace NUMINAMATH_CALUDE_prob_third_given_a_wins_l1775_177589

/-- The probability of Player A winning a single game -/
def p : ℚ := 2/3

/-- The probability of Player A winning the championship -/
def prob_a_wins : ℚ := p^2 + 2*p^2*(1-p)

/-- The probability of the match going to the third game and Player A winning -/
def prob_third_and_a_wins : ℚ := 2*p^2*(1-p)

/-- The probability of the match going to the third game given that Player A wins the championship -/
theorem prob_third_given_a_wins : 
  prob_third_and_a_wins / prob_a_wins = 2/5 := by sorry

end NUMINAMATH_CALUDE_prob_third_given_a_wins_l1775_177589


namespace NUMINAMATH_CALUDE_sum_of_seventh_terms_l1775_177588

/-- Given two arithmetic sequences a and b, prove that a₇ + b₇ = 8 -/
theorem sum_of_seventh_terms (a b : ℕ → ℝ) : 
  (∀ n : ℕ, ∃ d : ℝ, a (n + 1) - a n = d) →  -- a is an arithmetic sequence
  (∀ n : ℕ, ∃ d : ℝ, b (n + 1) - b n = d) →  -- b is an arithmetic sequence
  a 2 + b 2 = 3 →                            -- given condition
  a 4 + b 4 = 5 →                            -- given condition
  a 7 + b 7 = 8 :=                           -- conclusion to prove
by sorry

end NUMINAMATH_CALUDE_sum_of_seventh_terms_l1775_177588


namespace NUMINAMATH_CALUDE_fish_market_problem_l1775_177528

theorem fish_market_problem (mackerel croaker tuna : ℕ) : 
  mackerel = 48 →
  mackerel * 11 = croaker * 6 →
  croaker * 8 = tuna →
  tuna = 704 := by
sorry

end NUMINAMATH_CALUDE_fish_market_problem_l1775_177528


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l1775_177519

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := sorry

-- State the theorem
theorem solution_set_of_inequality :
  ∀ x : ℝ, 0 < x ∧ x < 1 ↔ f (Real.exp x) < 1 :=
by
  sorry

-- Define the properties of f
axiom f_derivative (x : ℝ) (h : x > 0) : 
  deriv f x = (x - (Real.exp 1 - 1)) / x

axiom f_value_at_1 : f 1 = 1

axiom f_value_at_e : f (Real.exp 1) = 1

end NUMINAMATH_CALUDE_solution_set_of_inequality_l1775_177519


namespace NUMINAMATH_CALUDE_area_of_EFGH_l1775_177527

/-- Parallelogram with vertices E, F, G, H in 2D space -/
structure Parallelogram where
  E : ℝ × ℝ
  F : ℝ × ℝ
  G : ℝ × ℝ
  H : ℝ × ℝ

/-- Calculate the area of a parallelogram -/
def parallelogramArea (p : Parallelogram) : ℝ :=
  let base := |p.F.2 - p.E.2|
  let height := |p.G.1 - p.E.1|
  base * height

/-- The specific parallelogram EFGH from the problem -/
def EFGH : Parallelogram :=
  { E := (2, -3)
    F := (2, 2)
    G := (7, 9)
    H := (7, 2) }

theorem area_of_EFGH : parallelogramArea EFGH = 25 := by
  sorry

end NUMINAMATH_CALUDE_area_of_EFGH_l1775_177527


namespace NUMINAMATH_CALUDE_average_price_per_book_l1775_177526

theorem average_price_per_book (books1 books2 : ℕ) (price1 price2 : ℕ) 
  (h1 : books1 = 32)
  (h2 : books2 = 60)
  (h3 : price1 = 1500)
  (h4 : price2 = 340) :
  (price1 + price2) / (books1 + books2) = 20 := by
sorry

end NUMINAMATH_CALUDE_average_price_per_book_l1775_177526


namespace NUMINAMATH_CALUDE_opposite_sign_coordinates_second_quadrant_range_l1775_177508

def P (x : ℝ) : ℝ × ℝ := (x - 2, x)

theorem opposite_sign_coordinates (x : ℝ) :
  (P x).1 * (P x).2 < 0 → x = 1 := by sorry

theorem second_quadrant_range (x : ℝ) :
  (P x).1 < 0 ∧ (P x).2 > 0 → 0 < x ∧ x < 2 := by sorry

end NUMINAMATH_CALUDE_opposite_sign_coordinates_second_quadrant_range_l1775_177508


namespace NUMINAMATH_CALUDE_number_of_factors_19368_l1775_177564

theorem number_of_factors_19368 : Nat.card (Nat.divisors 19368) = 24 := by
  sorry

end NUMINAMATH_CALUDE_number_of_factors_19368_l1775_177564


namespace NUMINAMATH_CALUDE_min_value_theorem_l1775_177532

theorem min_value_theorem (x : ℝ) (h : x > 4) :
  (x + 15) / Real.sqrt (x - 4) ≥ 2 * Real.sqrt 19 ∧
  ((x + 15) / Real.sqrt (x - 4) = 2 * Real.sqrt 19 ↔ x = 23) := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1775_177532


namespace NUMINAMATH_CALUDE_calculate_savings_l1775_177562

/-- Calculates a person's savings given their income and income-to-expenditure ratio -/
theorem calculate_savings (income : ℕ) (income_ratio expenditure_ratio : ℕ) : 
  income_ratio > 0 ∧ expenditure_ratio > 0 ∧ income = 18000 ∧ income_ratio = 5 ∧ expenditure_ratio = 4 →
  income - (income * expenditure_ratio / income_ratio) = 3600 := by
sorry

end NUMINAMATH_CALUDE_calculate_savings_l1775_177562


namespace NUMINAMATH_CALUDE_nabla_two_three_l1775_177538

def nabla (a b : ℕ+) : ℕ := a.val ^ b.val * b.val ^ a.val

theorem nabla_two_three : nabla 2 3 = 72 := by sorry

end NUMINAMATH_CALUDE_nabla_two_three_l1775_177538


namespace NUMINAMATH_CALUDE_opposite_number_theorem_l1775_177567

theorem opposite_number_theorem (m : ℤ) : (m + 1 = -(-4)) → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_opposite_number_theorem_l1775_177567


namespace NUMINAMATH_CALUDE_p_greater_than_q_l1775_177596

theorem p_greater_than_q (x y : ℝ) (h1 : x < y) (h2 : y < 0) 
  (p : ℝ := (x^2 + y^2)*(x - y)) (q : ℝ := (x^2 - y^2)*(x + y)) : p > q := by
  sorry

end NUMINAMATH_CALUDE_p_greater_than_q_l1775_177596


namespace NUMINAMATH_CALUDE_meeting_point_closer_to_a_l1775_177575

/-- The distance between two points A and B -/
def total_distance : ℕ := 85

/-- The constant speed of the person starting from point A -/
def speed_a : ℕ := 5

/-- The initial speed of the person starting from point B -/
def initial_speed_b : ℕ := 4

/-- The hourly increase in speed for the person starting from point B -/
def speed_increase_b : ℕ := 1

/-- The number of hours until the two people meet -/
def meeting_time : ℕ := 6

/-- The distance walked by the person starting from point A -/
def distance_a : ℕ := speed_a * meeting_time

/-- The distance walked by the person starting from point B -/
def distance_b : ℕ := meeting_time * (initial_speed_b + (meeting_time - 1) / 2 * speed_increase_b)

/-- The difference in distances walked by the two people -/
def distance_difference : ℤ := distance_b - distance_a

theorem meeting_point_closer_to_a : distance_difference = 9 := by sorry

end NUMINAMATH_CALUDE_meeting_point_closer_to_a_l1775_177575


namespace NUMINAMATH_CALUDE_number_puzzle_l1775_177582

theorem number_puzzle : ∃ x : ℚ, (x / 5 + 4 = x / 4 - 10) ∧ x = 280 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l1775_177582


namespace NUMINAMATH_CALUDE_fraction_sum_and_product_l1775_177569

theorem fraction_sum_and_product : 
  (2 / 16 + 3 / 18 + 4 / 24) * (3 / 5) = 11 / 40 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_and_product_l1775_177569


namespace NUMINAMATH_CALUDE_unique_solution_system_l1775_177551

theorem unique_solution_system (x y z : ℝ) : 
  x + y + z = 2008 ∧
  x^2 + y^2 + z^2 = 6024^2 ∧
  1/x + 1/y + 1/z = 1/2008 →
  (x = 2008 ∧ y = 4016 ∧ z = -4016) ∨
  (x = 2008 ∧ y = -4016 ∧ z = 4016) ∨
  (x = 4016 ∧ y = 2008 ∧ z = -4016) ∨
  (x = 4016 ∧ y = -4016 ∧ z = 2008) ∨
  (x = -4016 ∧ y = 2008 ∧ z = 4016) ∨
  (x = -4016 ∧ y = 4016 ∧ z = 2008) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_system_l1775_177551


namespace NUMINAMATH_CALUDE_nonagon_configuration_count_l1775_177520

structure NonagonConfiguration where
  vertices : Fin 9 → Fin 11
  center : Fin 11
  midpoint : Fin 11
  all_different : ∀ i j, i ≠ j → 
    (vertices i ≠ vertices j) ∧ 
    (vertices i ≠ center) ∧ 
    (vertices i ≠ midpoint) ∧ 
    (center ≠ midpoint)
  equal_sums : ∀ i : Fin 9, 
    (vertices i : ℕ) + (midpoint : ℕ) + (center : ℕ) = 
    (vertices 0 : ℕ) + (midpoint : ℕ) + (center : ℕ)

def count_valid_configurations : ℕ := sorry

theorem nonagon_configuration_count :
  count_valid_configurations = 10321920 := by sorry

end NUMINAMATH_CALUDE_nonagon_configuration_count_l1775_177520


namespace NUMINAMATH_CALUDE_sixteenth_root_of_unity_l1775_177557

theorem sixteenth_root_of_unity : 
  ∃ (n : ℤ), 0 ≤ n ∧ n ≤ 15 ∧ 
  (Complex.tan (π / 8) + Complex.I) / (Complex.tan (π / 8) - Complex.I) = 
  Complex.exp (Complex.I * (2 * ↑n * π / 16)) :=
sorry

end NUMINAMATH_CALUDE_sixteenth_root_of_unity_l1775_177557


namespace NUMINAMATH_CALUDE_intersection_equals_T_l1775_177566

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_equals_T : S ∩ T = T := by sorry

end NUMINAMATH_CALUDE_intersection_equals_T_l1775_177566


namespace NUMINAMATH_CALUDE_dividend_calculation_l1775_177565

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 16) 
  (h2 : quotient = 8) 
  (h3 : remainder = 4) : 
  divisor * quotient + remainder = 132 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l1775_177565


namespace NUMINAMATH_CALUDE_cheaper_module_cost_l1775_177507

theorem cheaper_module_cost (expensive_cost : ℝ) (total_modules : ℕ) (cheap_modules : ℕ) (total_value : ℝ) :
  expensive_cost = 10 →
  total_modules = 11 →
  cheap_modules = 10 →
  total_value = 45 →
  ∃ (cheap_cost : ℝ), cheap_cost * cheap_modules + expensive_cost * (total_modules - cheap_modules) = total_value ∧ cheap_cost = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_cheaper_module_cost_l1775_177507


namespace NUMINAMATH_CALUDE_stamp_collection_percentage_l1775_177521

theorem stamp_collection_percentage (total : ℕ) (chinese_percent : ℚ) (japanese_count : ℕ) : 
  total = 100 →
  chinese_percent = 35 / 100 →
  japanese_count = 45 →
  (total - (chinese_percent * total).floor - japanese_count) / total * 100 = 20 := by
sorry

end NUMINAMATH_CALUDE_stamp_collection_percentage_l1775_177521


namespace NUMINAMATH_CALUDE_dog_distance_l1775_177511

/-- The total distance run by a dog between two people walking towards each other -/
theorem dog_distance (total_distance : ℝ) (speed_A speed_B speed_dog : ℝ) : 
  total_distance = 100 ∧ 
  speed_A = 6 ∧ 
  speed_B = 4 ∧ 
  speed_dog = 10 → 
  (total_distance / (speed_A + speed_B)) * speed_dog = 100 :=
by sorry

end NUMINAMATH_CALUDE_dog_distance_l1775_177511


namespace NUMINAMATH_CALUDE_lydia_almonds_l1775_177545

theorem lydia_almonds (lydia_almonds max_almonds : ℕ) : 
  lydia_almonds = max_almonds + 8 →
  max_almonds = lydia_almonds / 3 →
  lydia_almonds = 12 := by
sorry

end NUMINAMATH_CALUDE_lydia_almonds_l1775_177545


namespace NUMINAMATH_CALUDE_original_triangle_area_l1775_177549

theorem original_triangle_area (original_area new_area : ℝ) : 
  (∀ (side : ℝ), new_area = (5 * side)^2 / 2 → original_area = side^2 / 2) →
  new_area = 200 →
  original_area = 8 :=
by sorry

end NUMINAMATH_CALUDE_original_triangle_area_l1775_177549


namespace NUMINAMATH_CALUDE_quadratic_roots_l1775_177597

theorem quadratic_roots (p q a b : ℤ) : 
  (∀ x, x^2 + p*x + q = 0 ↔ x = a ∨ x = b) →  -- polynomial has roots a and b
  a ≠ b →                                    -- roots are distinct
  a ≠ 0 →                                    -- a is non-zero
  b ≠ 0 →                                    -- b is non-zero
  (a + p) % (q - 2*b) = 0 →                  -- a + p is divisible by q - 2b
  a = 1 ∨ a = 3 :=                           -- possible values for a
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_l1775_177597


namespace NUMINAMATH_CALUDE_geometric_arithmetic_progression_problem_l1775_177570

theorem geometric_arithmetic_progression_problem :
  ∃ (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ),
    (a₁ * c₁ = b₁^2 ∧ a₁ + c₁ = 2*(b₁ + 8) ∧ a₁ * (c₁ + 64) = (b₁ + 8)^2) ∧
    (a₂ * c₂ = b₂^2 ∧ a₂ + c₂ = 2*(b₂ + 8) ∧ a₂ * (c₂ + 64) = (b₂ + 8)^2) ∧
    (a₁ = 4/9 ∧ b₁ = -20/9 ∧ c₁ = 100/9) ∧
    (a₂ = 4 ∧ b₂ = 12 ∧ c₂ = 36) :=
by sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_progression_problem_l1775_177570


namespace NUMINAMATH_CALUDE_sphere_section_distance_l1775_177592

theorem sphere_section_distance (r : ℝ) (d : ℝ) (A : ℝ) :
  r = 2 →
  A = π →
  d = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_sphere_section_distance_l1775_177592


namespace NUMINAMATH_CALUDE_jackets_sold_after_noon_l1775_177541

theorem jackets_sold_after_noon :
  let total_jackets : ℕ := 214
  let price_before_noon : ℚ := 31.95
  let price_after_noon : ℚ := 18.95
  let total_receipts : ℚ := 5108.30
  let jackets_after_noon : ℕ := 133
  let jackets_before_noon : ℕ := total_jackets - jackets_after_noon
  (jackets_before_noon : ℚ) * price_before_noon + (jackets_after_noon : ℚ) * price_after_noon = total_receipts →
  jackets_after_noon = 133 :=
by
  sorry

end NUMINAMATH_CALUDE_jackets_sold_after_noon_l1775_177541


namespace NUMINAMATH_CALUDE_categorical_variables_are_correct_l1775_177558

-- Define the type for variables
inductive Variable
  | Smoking
  | Gender
  | Religious_Belief
  | Nationality

-- Define a function to check if a variable is categorical
def is_categorical (v : Variable) : Prop :=
  v = Variable.Gender ∨ v = Variable.Religious_Belief ∨ v = Variable.Nationality

-- Define the set of all variables
def all_variables : Set Variable :=
  {Variable.Smoking, Variable.Gender, Variable.Religious_Belief, Variable.Nationality}

-- Define the set of categorical variables
def categorical_variables : Set Variable :=
  {v ∈ all_variables | is_categorical v}

-- The theorem to prove
theorem categorical_variables_are_correct :
  categorical_variables = {Variable.Gender, Variable.Religious_Belief, Variable.Nationality} :=
by sorry

end NUMINAMATH_CALUDE_categorical_variables_are_correct_l1775_177558


namespace NUMINAMATH_CALUDE_x_plus_y_value_l1775_177556

theorem x_plus_y_value (x y : ℝ) 
  (h1 : x + Real.cos y = 3012)
  (h2 : x + 3012 * Real.sin y = 3010)
  (h3 : 0 ≤ y ∧ y ≤ Real.pi) :
  x + y = 3012 + Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_value_l1775_177556


namespace NUMINAMATH_CALUDE_part_one_part_two_l1775_177544

/-- A quadratic equation ax^2 + bx + c = 0 is a double root equation if one root is twice the other -/
def is_double_root_equation (a b c : ℝ) : Prop :=
  ∃ (x y : ℝ), x ≠ 0 ∧ y = 2*x ∧ a*x^2 + b*x + c = 0 ∧ a*y^2 + b*y + c = 0

/-- The first part of the theorem -/
theorem part_one : is_double_root_equation 1 (-3) 2 := by sorry

/-- The second part of the theorem -/
theorem part_two :
  ∀ (a b : ℝ), is_double_root_equation a b (-6) →
  (∃ (x : ℝ), x = 2 ∧ a*x^2 + b*x - 6 = 0) →
  ((a = -3/4 ∧ b = 9/2) ∨ (a = -3 ∧ b = 9)) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1775_177544


namespace NUMINAMATH_CALUDE_sue_movie_borrowing_l1775_177555

/-- The number of movies Sue initially borrowed -/
def initial_movies : ℕ := 6

/-- The number of books Sue initially borrowed -/
def initial_books : ℕ := 15

/-- The number of books Sue returned -/
def returned_books : ℕ := 8

/-- The number of additional books Sue checked out -/
def additional_books : ℕ := 9

/-- The total number of items Sue has at the end -/
def total_items : ℕ := 20

theorem sue_movie_borrowing :
  initial_movies = 6 ∧
  initial_books + initial_movies - returned_books - (initial_movies / 3) + additional_books = total_items :=
by sorry

end NUMINAMATH_CALUDE_sue_movie_borrowing_l1775_177555


namespace NUMINAMATH_CALUDE_A_inverse_l1775_177546

-- Define the matrix A
def A : Matrix (Fin 2) (Fin 2) ℚ :=
  !![4, 5;
    -2, 9]

-- Define the claimed inverse matrix A_inv
def A_inv : Matrix (Fin 2) (Fin 2) ℚ :=
  !![9/46, -5/46;
    1/23, 2/23]

-- Theorem stating that A_inv is the inverse of A
theorem A_inverse : A⁻¹ = A_inv := by
  sorry

end NUMINAMATH_CALUDE_A_inverse_l1775_177546


namespace NUMINAMATH_CALUDE_third_number_proof_l1775_177590

def digit_sum (n : ℕ) : ℕ := sorry

def has_same_remainder (a b c n : ℕ) : Prop :=
  ∃ r, a % n = r ∧ b % n = r ∧ c % n = r

theorem third_number_proof :
  ∃! x : ℕ,
    ∃ n : ℕ,
      has_same_remainder 1305 4665 x n ∧
      (∀ m : ℕ, has_same_remainder 1305 4665 x m → m ≤ n) ∧
      digit_sum n = 4 ∧
      x = 4705 :=
sorry

end NUMINAMATH_CALUDE_third_number_proof_l1775_177590


namespace NUMINAMATH_CALUDE_sqrt_77_consecutive_integers_product_l1775_177515

theorem sqrt_77_consecutive_integers_product : ∃ n : ℕ, 
  (n : ℝ) < Real.sqrt 77 ∧ 
  Real.sqrt 77 < (n + 1 : ℝ) ∧ 
  n * (n + 1) = 72 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_77_consecutive_integers_product_l1775_177515


namespace NUMINAMATH_CALUDE_race_time_proof_l1775_177585

/-- Represents the race times of two runners -/
structure RaceTimes where
  total : ℕ
  difference : ℕ

/-- Calculates the longer race time given the total time and the difference between runners -/
def longerTime (times : RaceTimes) : ℕ :=
  (times.total + times.difference) / 2

theorem race_time_proof (times : RaceTimes) (h1 : times.total = 112) (h2 : times.difference = 4) :
  longerTime times = 58 := by
  sorry

end NUMINAMATH_CALUDE_race_time_proof_l1775_177585


namespace NUMINAMATH_CALUDE_catch_in_park_l1775_177584

-- Define the square park
structure Park :=
  (side_length : ℝ)
  (has_diagonal_walkways : Bool)

-- Define the participants
structure Participant :=
  (speed : ℝ)
  (position : ℝ × ℝ)

-- Define the catching condition
def can_catch (pursuer1 pursuer2 target : Participant) (park : Park) : Prop :=
  ∃ (t : ℝ), t > 0 ∧ 
  (pursuer1.position = target.position ∨ pursuer2.position = target.position)

-- Theorem statement
theorem catch_in_park (park : Park) (pursuer1 pursuer2 target : Participant) :
  park.side_length > 0 ∧
  park.has_diagonal_walkways = true ∧
  pursuer1.speed > 0 ∧
  pursuer2.speed > 0 ∧
  target.speed = 3 * pursuer1.speed ∧
  target.speed = 3 * pursuer2.speed →
  can_catch pursuer1 pursuer2 target park :=
sorry

end NUMINAMATH_CALUDE_catch_in_park_l1775_177584


namespace NUMINAMATH_CALUDE_inequality_proof_l1775_177577

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  Real.sqrt (a^2 + a*b + b^2) + Real.sqrt (a^2 + a*c + c^2) ≥ 
  4 * Real.sqrt ((a*b / (a+b))^2 + (a*b / (a+b)) * (a*c / (a+c)) + (a*c / (a+c))^2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1775_177577


namespace NUMINAMATH_CALUDE_train_speed_with_stoppages_l1775_177559

/-- Given a train that travels at 400 km/h without stoppages and stops for 6 minutes per hour,
    its average speed with stoppages is 360 km/h. -/
theorem train_speed_with_stoppages :
  let speed_without_stoppages : ℝ := 400
  let minutes_stopped_per_hour : ℝ := 6
  let minutes_per_hour : ℝ := 60
  let speed_with_stoppages : ℝ := speed_without_stoppages * (minutes_per_hour - minutes_stopped_per_hour) / minutes_per_hour
  speed_with_stoppages = 360 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_with_stoppages_l1775_177559


namespace NUMINAMATH_CALUDE_matrix_problem_l1775_177509

-- Define 2x2 matrices A and B
variable (A B : Matrix (Fin 2) (Fin 2) ℝ)

-- State the conditions
axiom cond1 : A * B = A ^ 2 * B ^ 2 - (A * B) ^ 2
axiom cond2 : Matrix.det B = 2

-- Theorem statement
theorem matrix_problem :
  Matrix.det A = 0 ∧ Matrix.det (A + 2 • B) - Matrix.det (B + 2 • A) = 6 := by
  sorry

end NUMINAMATH_CALUDE_matrix_problem_l1775_177509


namespace NUMINAMATH_CALUDE_no_palindromes_with_two_fives_l1775_177595

def isPalindrome (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n < 2000 ∧ (n / 1000 = n % 10) ∧ ((n / 100) % 10 ≠ (n / 10) % 10)

def hasTwoFives (n : ℕ) : Prop :=
  (n / 1000 = 5) ∨ ((n / 100) % 10 = 5) ∨ ((n / 10) % 10 = 5) ∨ (n % 10 = 5)

theorem no_palindromes_with_two_fives :
  ¬∃ n : ℕ, isPalindrome n ∧ hasTwoFives n :=
sorry

end NUMINAMATH_CALUDE_no_palindromes_with_two_fives_l1775_177595


namespace NUMINAMATH_CALUDE_greg_dog_walking_earnings_l1775_177552

/-- Greg's dog walking business model and earnings calculation --/
theorem greg_dog_walking_earnings :
  let base_charge : ℕ := 20
  let per_minute_charge : ℕ := 1
  let one_dog_minutes : ℕ := 10
  let two_dogs_minutes : ℕ := 7
  let three_dogs_minutes : ℕ := 9
  let total_earnings := 
    (base_charge + per_minute_charge * one_dog_minutes) +
    2 * (base_charge + per_minute_charge * two_dogs_minutes) +
    3 * (base_charge + per_minute_charge * three_dogs_minutes)
  total_earnings = 171 := by sorry

end NUMINAMATH_CALUDE_greg_dog_walking_earnings_l1775_177552


namespace NUMINAMATH_CALUDE_find_k_l1775_177561

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define the vectors
variable (e₁ e₂ : V)

-- Define the non-collinearity of e₁ and e₂
variable (h_non_collinear : ∀ (r : ℝ), e₁ ≠ r • e₂)

-- Define the vectors AB, CD, and CB
variable (k : ℝ)
def AB := 2 • e₁ + k • e₂
def CD := 2 • e₁ - 1 • e₂
def CB := 1 • e₁ + 3 • e₂

-- Define collinearity of A, B, and D
def collinear (v w : V) : Prop := ∃ (r : ℝ), v = r • w

-- State the theorem
theorem find_k : 
  collinear (AB e₁ e₂ k) (CD e₁ e₂ - CB e₁ e₂) → k = -8 :=
sorry

end NUMINAMATH_CALUDE_find_k_l1775_177561


namespace NUMINAMATH_CALUDE_baseball_team_average_l1775_177516

theorem baseball_team_average (total_score : ℕ) (total_players : ℕ) (top_scorers : ℕ) (top_average : ℕ) (remaining_average : ℕ) : 
  total_score = 270 →
  total_players = 9 →
  top_scorers = 5 →
  top_average = 50 →
  remaining_average = 5 →
  top_scorers * top_average + (total_players - top_scorers) * remaining_average = total_score :=
by sorry

end NUMINAMATH_CALUDE_baseball_team_average_l1775_177516


namespace NUMINAMATH_CALUDE_dice_probability_l1775_177560

def num_dice : ℕ := 6
def num_sides : ℕ := 15
def num_low_sides : ℕ := 9
def num_high_sides : ℕ := 6

def prob_low : ℚ := num_low_sides / num_sides
def prob_high : ℚ := num_high_sides / num_sides

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem dice_probability : 
  (choose num_dice (num_dice / 2)) * (prob_low ^ (num_dice / 2)) * (prob_high ^ (num_dice / 2)) = 4320 / 15625 := by
  sorry

end NUMINAMATH_CALUDE_dice_probability_l1775_177560


namespace NUMINAMATH_CALUDE_mean_value_theorem_application_l1775_177599

-- Define the function f(x) = x^2 + 3
def f (x : ℝ) : ℝ := x^2 + 3

-- State the theorem
theorem mean_value_theorem_application :
  ∃ c ∈ (Set.Ioo (-1) 2), 
    (deriv f c) = (f 2 - f (-1)) / (2 - (-1)) :=
by
  sorry

end NUMINAMATH_CALUDE_mean_value_theorem_application_l1775_177599


namespace NUMINAMATH_CALUDE_sqrt_fraction_equals_two_power_fifteen_l1775_177571

theorem sqrt_fraction_equals_two_power_fifteen :
  let thirty_two := (2 : ℝ) ^ 5
  let sixteen := (2 : ℝ) ^ 4
  (((thirty_two ^ 15 + sixteen ^ 15) / (thirty_two ^ 6 + sixteen ^ 18)) ^ (1/2 : ℝ)) = (2 : ℝ) ^ 15 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fraction_equals_two_power_fifteen_l1775_177571


namespace NUMINAMATH_CALUDE_jessica_remaining_seashells_l1775_177522

/-- The number of seashells Jessica initially found -/
def initial_seashells : ℕ := 8

/-- The number of seashells Jessica gave to Joan -/
def given_seashells : ℕ := 6

/-- The number of seashells Jessica is left with -/
def remaining_seashells : ℕ := initial_seashells - given_seashells

theorem jessica_remaining_seashells : remaining_seashells = 2 := by
  sorry

end NUMINAMATH_CALUDE_jessica_remaining_seashells_l1775_177522


namespace NUMINAMATH_CALUDE_derivative_of_y_l1775_177579

noncomputable def y (x : ℝ) : ℝ := x + Real.cos x

theorem derivative_of_y (x : ℝ) : 
  deriv y x = 1 - Real.sin x := by sorry

end NUMINAMATH_CALUDE_derivative_of_y_l1775_177579


namespace NUMINAMATH_CALUDE_option_B_not_mapping_l1775_177593

-- Define the sets and mappings
def CartesianPlane : Type := ℝ × ℝ
def CircleOnPlane : Type := Unit -- Placeholder type for circles
def TriangleOnPlane : Type := Unit -- Placeholder type for triangles

-- Option A
def mappingA : CartesianPlane → CartesianPlane := id

-- Option B (not a mapping)
noncomputable def correspondenceB : CircleOnPlane → Set TriangleOnPlane := sorry

-- Option C
def mappingC : ℕ → Fin 2 := fun n => n % 2

-- Option D
def mappingD : Fin 3 → Fin 3 := fun n => n^2

-- Theorem stating that B is not a mapping while others are
theorem option_B_not_mapping :
  (∀ x : CartesianPlane, ∃! y : CartesianPlane, mappingA x = y) ∧
  (∃ c : CircleOnPlane, ¬∃! t : TriangleOnPlane, t ∈ correspondenceB c) ∧
  (∀ n : ℕ, ∃! m : Fin 2, mappingC n = m) ∧
  (∀ x : Fin 3, ∃! y : Fin 3, mappingD x = y) := by
  sorry

end NUMINAMATH_CALUDE_option_B_not_mapping_l1775_177593


namespace NUMINAMATH_CALUDE_percentage_of_juniors_l1775_177591

def total_students : ℕ := 800
def seniors : ℕ := 160

theorem percentage_of_juniors : 
  ∀ (freshmen sophomores juniors : ℕ),
  freshmen + sophomores + juniors + seniors = total_students →
  sophomores = total_students / 4 →
  freshmen = sophomores + 32 →
  (juniors : ℚ) / total_students * 100 = 26 :=
by sorry

end NUMINAMATH_CALUDE_percentage_of_juniors_l1775_177591


namespace NUMINAMATH_CALUDE_tangent_dot_product_l1775_177537

/-- The circle with center at the origin and radius 1 -/
def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- Point M outside the circle -/
def M : ℝ × ℝ := (2, 0)

/-- A point is on the circle -/
def on_circle (p : ℝ × ℝ) : Prop := unit_circle p.1 p.2

/-- A line is tangent to the circle at a point -/
def is_tangent (p q : ℝ × ℝ) : Prop :=
  on_circle p ∧ (p.1 * q.1 + p.2 * q.2 = 1)

/-- The dot product of two vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem tangent_dot_product :
  ∃ (A B : ℝ × ℝ),
    on_circle A ∧
    on_circle B ∧
    is_tangent A M ∧
    is_tangent B M ∧
    dot_product (A.1 - M.1, A.2 - M.2) (B.1 - M.1, B.2 - M.2) = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_dot_product_l1775_177537


namespace NUMINAMATH_CALUDE_solid_volume_l1775_177534

/-- A solid with a square base and specific edge lengths -/
structure Solid where
  s : ℝ
  base_side_length : s > 0
  upper_edge_length : ℝ := 3 * s
  other_edge_length : ℝ := s

/-- The volume of the solid -/
def volume (solid : Solid) : ℝ := sorry

theorem solid_volume : 
  ∀ (solid : Solid), solid.s = 8 * Real.sqrt 2 → volume solid = 5760 := by
  sorry

end NUMINAMATH_CALUDE_solid_volume_l1775_177534


namespace NUMINAMATH_CALUDE_cubic_roots_problem_l1775_177587

theorem cubic_roots_problem (a b c : ℝ) 
  (h1 : a ≤ b) (h2 : b ≤ c)
  (h3 : a + b + c = -1)
  (h4 : a * b + b * c + a * c = -4)
  (h5 : a * b * c = -2) : 
  a = -1 - Real.sqrt 3 ∧ b = -1 + Real.sqrt 3 ∧ c = 1 := by
  sorry

end NUMINAMATH_CALUDE_cubic_roots_problem_l1775_177587


namespace NUMINAMATH_CALUDE_halving_r_problem_l1775_177548

theorem halving_r_problem (r : ℝ) (n : ℝ) (a : ℝ) :
  a = (2 * r) ^ n →
  ((r / 2) ^ n = 0.125 * a) →
  n = 3 := by
sorry

end NUMINAMATH_CALUDE_halving_r_problem_l1775_177548


namespace NUMINAMATH_CALUDE_sodium_hydrogen_sulfate_effect_l1775_177510

-- Define the water ionization equilibrium
def water_ionization (temp : ℝ) : Prop :=
  temp = 25 → ∃ (K : ℝ), K > 0 ∧ ∀ (c_H2O c_H c_OH : ℝ),
    c_H * c_OH = K * c_H2O

-- Define the enthalpy change
def delta_H_positive : Prop := ∃ (ΔH : ℝ), ΔH > 0

-- Define the addition of sodium hydrogen sulfate
def add_NaHSO4 (c_H_initial c_H_final : ℝ) : Prop :=
  c_H_final > c_H_initial

-- Theorem statement
theorem sodium_hydrogen_sulfate_effect
  (h1 : water_ionization 25)
  (h2 : delta_H_positive)
  (h3 : ∃ (c_H_initial c_H_final : ℝ), add_NaHSO4 c_H_initial c_H_final) :
  ∃ (K : ℝ), K > 0 ∧
    (∀ (c_H2O c_H c_OH : ℝ), c_H * c_OH = K * c_H2O) ∧
    (∃ (c_H_initial c_H_final : ℝ), c_H_final > c_H_initial) :=
sorry

end NUMINAMATH_CALUDE_sodium_hydrogen_sulfate_effect_l1775_177510


namespace NUMINAMATH_CALUDE_function_property_l1775_177550

noncomputable def f (x : ℝ) : ℝ := x * (1 - Real.log x)

theorem function_property (x₁ x₂ : ℝ) (h1 : x₁ > 0) (h2 : x₂ > 0) (h3 : x₁ ≠ x₂) (h4 : f x₁ = f x₂) :
  x₁ + x₂ < Real.exp 1 := by
sorry

end NUMINAMATH_CALUDE_function_property_l1775_177550


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1775_177554

/-- An arithmetic sequence with first term -2015 -/
def arithmetic_sequence (n : ℕ) : ℤ := -2015 + (n - 1) * d
  where d : ℤ := 2  -- We define d here, but it should be derived in the proof

/-- Sum of the first n terms of the arithmetic sequence -/
def S (n : ℕ) : ℤ := n * (2 * (-2015) + (n - 1) * d) / 2
  where d : ℤ := 2  -- We define d here, but it should be derived in the proof

/-- Main theorem -/
theorem arithmetic_sequence_sum :
  2 * S 6 - 3 * S 4 = 24 → S 2015 = -2015 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1775_177554


namespace NUMINAMATH_CALUDE_train_journey_duration_l1775_177574

/-- Given a train journey with a distance and average speed, calculate the duration of the journey. -/
theorem train_journey_duration (distance : ℝ) (speed : ℝ) (duration : ℝ) 
  (h_distance : distance = 27) 
  (h_speed : speed = 3) 
  (h_duration : duration = distance / speed) : 
  duration = 9 := by
  sorry

end NUMINAMATH_CALUDE_train_journey_duration_l1775_177574


namespace NUMINAMATH_CALUDE_multiples_5_or_7_not_both_main_theorem_l1775_177542

def count_multiples (n : ℕ) (m : ℕ) : ℕ :=
  (n - 1) / m

theorem multiples_5_or_7_not_both (upper_bound : ℕ) 
  (h_upper_bound : upper_bound = 101) : ℕ := by
  let multiples_5 := count_multiples upper_bound 5
  let multiples_7 := count_multiples upper_bound 7
  let multiples_35 := count_multiples upper_bound 35
  exact (multiples_5 + multiples_7 - 2 * multiples_35)

theorem main_theorem : multiples_5_or_7_not_both 101 rfl = 30 := by
  sorry

end NUMINAMATH_CALUDE_multiples_5_or_7_not_both_main_theorem_l1775_177542


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1775_177568

theorem quadratic_equation_roots (k : ℝ) : 
  (∃ x : ℝ, x^2 + k*x - 2 = 0 ∧ x = -2) → 
  (∃ y : ℝ, y^2 + k*y - 2 = 0 ∧ y = 1 ∧ k = 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1775_177568


namespace NUMINAMATH_CALUDE_intersection_condition_l1775_177586

theorem intersection_condition (m n : ℝ) : 
  let A : Set ℝ := {2, m / (2 * n)}
  let B : Set ℝ := {m, n}
  (A ∩ B : Set ℝ) = {1} → n = 1/2 := by
sorry

end NUMINAMATH_CALUDE_intersection_condition_l1775_177586


namespace NUMINAMATH_CALUDE_heavy_washes_count_l1775_177502

/-- Represents the number of gallons of water used for different wash types and conditions --/
structure WashingMachine where
  heavyWashWater : ℕ
  regularWashWater : ℕ
  lightWashWater : ℕ
  bleachRinseWater : ℕ
  regularWashCount : ℕ
  lightWashCount : ℕ
  bleachedLoadsCount : ℕ
  totalWaterUsage : ℕ

/-- Calculates the number of heavy washes given the washing machine parameters --/
def calculateHeavyWashes (wm : WashingMachine) : ℕ :=
  (wm.totalWaterUsage - 
   (wm.regularWashWater * wm.regularWashCount + 
    wm.lightWashWater * wm.lightWashCount + 
    wm.bleachRinseWater * wm.bleachedLoadsCount)) / wm.heavyWashWater

/-- Theorem stating that the number of heavy washes is 2 given the specific conditions --/
theorem heavy_washes_count (wm : WashingMachine) 
  (h1 : wm.heavyWashWater = 20)
  (h2 : wm.regularWashWater = 10)
  (h3 : wm.lightWashWater = 2)
  (h4 : wm.bleachRinseWater = 2)
  (h5 : wm.regularWashCount = 3)
  (h6 : wm.lightWashCount = 1)
  (h7 : wm.bleachedLoadsCount = 2)
  (h8 : wm.totalWaterUsage = 76) :
  calculateHeavyWashes wm = 2 := by
  sorry

#eval calculateHeavyWashes {
  heavyWashWater := 20,
  regularWashWater := 10,
  lightWashWater := 2,
  bleachRinseWater := 2,
  regularWashCount := 3,
  lightWashCount := 1,
  bleachedLoadsCount := 2,
  totalWaterUsage := 76
}

end NUMINAMATH_CALUDE_heavy_washes_count_l1775_177502


namespace NUMINAMATH_CALUDE_complex_square_root_l1775_177504

theorem complex_square_root (a b : ℕ+) (h : (↑a + ↑b * Complex.I) ^ 2 = 5 + 12 * Complex.I) :
  ↑a + ↑b * Complex.I = 3 + 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_square_root_l1775_177504


namespace NUMINAMATH_CALUDE_puppy_cost_problem_l1775_177513

theorem puppy_cost_problem (total_cost : ℕ) (sale_price : ℕ) (distinct_cost1 : ℕ) (distinct_cost2 : ℕ) :
  total_cost = 2200 →
  sale_price = 180 →
  distinct_cost1 = 250 →
  distinct_cost2 = 300 →
  ∃ (remaining_price : ℕ),
    4 * sale_price + distinct_cost1 + distinct_cost2 + 2 * remaining_price = total_cost ∧
    remaining_price = 465 := by
  sorry

end NUMINAMATH_CALUDE_puppy_cost_problem_l1775_177513


namespace NUMINAMATH_CALUDE_hexagonal_circle_selection_l1775_177572

/-- Represents the number of ways to choose three consecutive circles in a direction --/
def consecutive_triples (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The total number of circles in the figure --/
def total_circles : ℕ := 33

/-- The number of circles in the longest row --/
def longest_row : ℕ := 6

/-- The number of ways to choose three consecutive circles in the first direction --/
def first_direction : ℕ := consecutive_triples longest_row

/-- The number of ways to choose three consecutive circles in each of the other two directions --/
def other_directions : ℕ := 18

/-- The total number of ways to choose three consecutive circles in all directions --/
def total_ways : ℕ := first_direction + 2 * other_directions

theorem hexagonal_circle_selection :
  total_ways = 57 :=
sorry

end NUMINAMATH_CALUDE_hexagonal_circle_selection_l1775_177572


namespace NUMINAMATH_CALUDE_integral_proof_l1775_177573

open Real

noncomputable def f (x : ℝ) : ℝ :=
  (1/16) * log (abs (x - 2)) + (15/16) * log (abs (x + 2)) + (33*x + 34) / (4*(x + 2)^2)

theorem integral_proof (x : ℝ) (hx2 : x ≠ 2) (hx_2 : x ≠ -2) :
  deriv f x = (x^3 - 6*x^2 + 13*x - 6) / ((x - 2)*(x + 2)^3) :=
by sorry

end NUMINAMATH_CALUDE_integral_proof_l1775_177573


namespace NUMINAMATH_CALUDE_roger_step_goal_time_l1775_177598

/-- Represents the time it takes Roger to reach his step goal -/
def time_to_reach_goal (steps_per_interval : ℕ) (interval_duration : ℕ) (goal_steps : ℕ) : ℕ :=
  (goal_steps * interval_duration) / steps_per_interval

/-- Proves that Roger will take 150 minutes to reach his goal of 10,000 steps -/
theorem roger_step_goal_time :
  time_to_reach_goal 2000 30 10000 = 150 := by
  sorry

end NUMINAMATH_CALUDE_roger_step_goal_time_l1775_177598


namespace NUMINAMATH_CALUDE_area_ratio_quadrupled_triangle_l1775_177512

/-- Given a triangle whose dimensions are quadrupled to form a larger triangle,
    this theorem relates the area of the larger triangle to the area of the original triangle. -/
theorem area_ratio_quadrupled_triangle (A : ℝ) :
  (4 * 4 * A = 64) → (A = 4) := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_quadrupled_triangle_l1775_177512
