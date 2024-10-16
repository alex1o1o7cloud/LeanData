import Mathlib

namespace NUMINAMATH_CALUDE_intersection_implies_sum_l893_89360

-- Define the functions
def f (a b x : ℝ) : ℝ := -|x - a| + b
def g (c d x : ℝ) : ℝ := |x - c| + d

-- State the theorem
theorem intersection_implies_sum (a b c d : ℝ) :
  (f a b 2 = 5) ∧ (f a b 8 = 3) ∧ (g c d 2 = 5) ∧ (g c d 8 = 3) →
  a + c = 10 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_sum_l893_89360


namespace NUMINAMATH_CALUDE_x_value_when_one_in_set_l893_89392

theorem x_value_when_one_in_set (x : ℝ) : 1 ∈ ({x, x^2} : Set ℝ) → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_x_value_when_one_in_set_l893_89392


namespace NUMINAMATH_CALUDE_expand_expression_l893_89322

theorem expand_expression (x : ℝ) : (x - 1) * (4 * x + 5) = 4 * x^2 + x - 5 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l893_89322


namespace NUMINAMATH_CALUDE_divisibility_of_all_ones_number_l893_89346

/-- A positive integer whose decimal representation contains only ones -/
def all_ones_number (n : ℕ) : ℕ :=
  (10^n - 1) / 9

theorem divisibility_of_all_ones_number (n : ℕ) (h : n > 0) :
  7 ∣ all_ones_number n → 13 ∣ all_ones_number n :=
by
  sorry

#check divisibility_of_all_ones_number

end NUMINAMATH_CALUDE_divisibility_of_all_ones_number_l893_89346


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l893_89343

theorem quadratic_inequality_range (m : ℝ) : 
  (∀ x : ℝ, x > 0 ∧ x ≤ 1 → x^2 - 4*x ≥ m) → m ≤ -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l893_89343


namespace NUMINAMATH_CALUDE_total_profit_equation_l893_89394

/-- Represents the initial investment of person A in rupees -/
def initial_investment_A : ℚ := 2000

/-- Represents the initial investment of person B in rupees -/
def initial_investment_B : ℚ := 4000

/-- Represents the number of months before investment change -/
def months_before_change : ℕ := 8

/-- Represents the number of months after investment change -/
def months_after_change : ℕ := 4

/-- Represents the amount A withdrew after 8 months in rupees -/
def amount_A_withdrew : ℚ := 1000

/-- Represents the amount B added after 8 months in rupees -/
def amount_B_added : ℚ := 1000

/-- Represents A's share of the profit in rupees -/
def A_profit_share : ℚ := 175

/-- Calculates the total investment of A over the year -/
def total_investment_A : ℚ :=
  initial_investment_A * months_before_change +
  (initial_investment_A - amount_A_withdrew) * months_after_change

/-- Calculates the total investment of B over the year -/
def total_investment_B : ℚ :=
  initial_investment_B * months_before_change +
  (initial_investment_B + amount_B_added) * months_after_change

/-- Theorem stating that the total profit P satisfies the equation (5/18) * P = 175 -/
theorem total_profit_equation (P : ℚ) :
  total_investment_A / (total_investment_A + total_investment_B) * P = A_profit_share := by
  sorry

end NUMINAMATH_CALUDE_total_profit_equation_l893_89394


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l893_89328

theorem sqrt_equation_solution :
  ∃! x : ℝ, Real.sqrt ((2 + Real.sqrt 3) ^ x) + Real.sqrt ((2 - Real.sqrt 3) ^ x) = 6 ∧ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l893_89328


namespace NUMINAMATH_CALUDE_ratio_difference_increases_dependence_l893_89342

/-- Represents a 2x2 contingency table -/
structure ContingencyTable where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Calculates the chi-square statistic for a 2x2 contingency table -/
def chi_square (table : ContingencyTable) : ℝ :=
  sorry

/-- Represents the probability of dependence between two variables -/
def dependence_probability (chi_square_value : ℝ) : ℝ :=
  sorry

/-- Theorem: As the difference between ratios increases, the probability of dependence increases -/
theorem ratio_difference_increases_dependence (table : ContingencyTable) :
  let ratio1 := table.a / (table.a + table.b)
  let ratio2 := table.c / (table.c + table.d)
  let diff := |ratio1 - ratio2|
  ∀ ε > 0, ∃ δ > 0,
    ∀ table' : ContingencyTable,
      let ratio1' := table'.a / (table'.a + table'.b)
      let ratio2' := table'.c / (table'.c + table'.d)
      let diff' := |ratio1' - ratio2'|
      diff' > diff + δ →
        dependence_probability (chi_square table') > dependence_probability (chi_square table) + ε :=
by
  sorry

end NUMINAMATH_CALUDE_ratio_difference_increases_dependence_l893_89342


namespace NUMINAMATH_CALUDE_tournament_outcomes_l893_89372

/-- Represents a tournament with n players --/
def Tournament (n : ℕ) := ℕ

/-- The number of possible outcomes in a tournament --/
def possibleOutcomes (t : Tournament n) : ℕ := 2^(n-1)

theorem tournament_outcomes (t : Tournament 6) : 
  possibleOutcomes t = 32 := by
  sorry

#check tournament_outcomes

end NUMINAMATH_CALUDE_tournament_outcomes_l893_89372


namespace NUMINAMATH_CALUDE_chord_equation_l893_89370

/-- Given a circle with equation x^2 + y^2 = 9 and a chord PQ with midpoint (1, 2),
    the equation of line PQ is x + 2y - 5 = 0 -/
theorem chord_equation (P Q : ℝ × ℝ) : 
  (∀ (x y : ℝ), (x, y) ∈ {p : ℝ × ℝ | p.1^2 + p.2^2 = 9} → 
    (P ∈ {p : ℝ × ℝ | p.1^2 + p.2^2 = 9} ∧ 
     Q ∈ {p : ℝ × ℝ | p.1^2 + p.2^2 = 9})) →
  ((P.1 + Q.1) / 2 = 1 ∧ (P.2 + Q.2) / 2 = 2) →
  ∃ (a b c : ℝ), a * P.1 + b * P.2 + c = 0 ∧ 
                  a * Q.1 + b * Q.2 + c = 0 ∧
                  a = 1 ∧ b = 2 ∧ c = -5 :=
by sorry

end NUMINAMATH_CALUDE_chord_equation_l893_89370


namespace NUMINAMATH_CALUDE_fencing_required_l893_89387

/-- Calculates the fencing required for a rectangular field -/
theorem fencing_required (area : ℝ) (uncovered_side : ℝ) : area = 400 ∧ uncovered_side = 20 → 
  ∃ (width : ℝ), area = uncovered_side * width ∧ uncovered_side + 2 * width = 60 := by
  sorry

end NUMINAMATH_CALUDE_fencing_required_l893_89387


namespace NUMINAMATH_CALUDE_circle_condition_tangent_circles_intersecting_circle_line_l893_89377

-- Define the equation C
def equation_C (x y m : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y + m = 0

-- Define the given circle equation
def given_circle (x y : ℝ) : Prop := x^2 + y^2 - 8*x - 12*y + 36 = 0

-- Define the line l
def line_l (x y : ℝ) : Prop := x + 2*y - 4 = 0

-- Theorem 1: For equation C to represent a circle, m < 5
theorem circle_condition (m : ℝ) : 
  (∃ x y, equation_C x y m) → m < 5 :=
sorry

-- Theorem 2: When circle C is tangent to the given circle, m = 4
theorem tangent_circles (m : ℝ) :
  (∃ x y, equation_C x y m ∧ given_circle x y) → m = 4 :=
sorry

-- Theorem 3: When circle C intersects line l at points M and N with |MN| = 4√5/5, m = 4
theorem intersecting_circle_line (m : ℝ) :
  (∃ x1 y1 x2 y2, 
    equation_C x1 y1 m ∧ equation_C x2 y2 m ∧
    line_l x1 y1 ∧ line_l x2 y2 ∧
    (x1 - x2)^2 + (y1 - y2)^2 = (4*Real.sqrt 5/5)^2) → 
  m = 4 :=
sorry

end NUMINAMATH_CALUDE_circle_condition_tangent_circles_intersecting_circle_line_l893_89377


namespace NUMINAMATH_CALUDE_alice_age_l893_89344

/-- Proves that Alice's current age is 30 years old given the specified conditions -/
theorem alice_age :
  ∀ (alice_age tom_age : ℕ),
  tom_age = alice_age - 15 →
  alice_age - 10 = 4 * (tom_age - 10) →
  alice_age = 30 :=
by
  sorry

#check alice_age

end NUMINAMATH_CALUDE_alice_age_l893_89344


namespace NUMINAMATH_CALUDE_cubic_root_of_unity_solutions_l893_89359

theorem cubic_root_of_unity_solutions (p q r s : ℂ) (m : ℂ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) :
  (p * m^2 + q * m + r = 0) ∧ (q * m^2 + r * m + s = 0) →
  (m = 1) ∨ (m = Complex.exp ((2 * Real.pi * Complex.I) / 3)) ∨ (m = Complex.exp ((-2 * Real.pi * Complex.I) / 3)) :=
by sorry

end NUMINAMATH_CALUDE_cubic_root_of_unity_solutions_l893_89359


namespace NUMINAMATH_CALUDE_isosceles_triangle_cut_l893_89327

-- Define the triangle PQR
structure Triangle :=
  (area : ℝ)
  (altitude : ℝ)

-- Define the line segment ST and resulting areas
structure Segment :=
  (length : ℝ)
  (trapezoid_area : ℝ)

-- Define the theorem
theorem isosceles_triangle_cut (PQR : Triangle) (ST : Segment) :
  PQR.area = 144 →
  PQR.altitude = 24 →
  ST.trapezoid_area = 108 →
  ST.length = 6 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_cut_l893_89327


namespace NUMINAMATH_CALUDE_exists_equilateral_DEF_l893_89353

open Real

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Represents a circle -/
structure Circle where
  center : Point
  radius : ℝ

/-- Checks if a triangle is acute-angled -/
def isAcuteAngled (t : Triangle) : Prop := sorry

/-- Checks if a point is inside a circle -/
def isInside (p : Point) (c : Circle) : Prop := sorry

/-- Gets the circumcircle of a triangle -/
def circumcircle (t : Triangle) : Circle := sorry

/-- Gets the intersection points of a ray from a point through another point with a circle -/
def rayIntersection (start : Point) (through : Point) (c : Circle) : Point := sorry

/-- Checks if a triangle is equilateral -/
def isEquilateral (t : Triangle) : Prop := sorry

/-- Main theorem -/
theorem exists_equilateral_DEF (ABC : Triangle) (c : Circle) :
  isAcuteAngled ABC →
  c = circumcircle ABC →
  ∃ P : Point,
    isInside P c ∧
    let D := rayIntersection A P c
    let E := rayIntersection B P c
    let F := rayIntersection C P c
    isEquilateral (Triangle.mk D E F) :=
by sorry

end NUMINAMATH_CALUDE_exists_equilateral_DEF_l893_89353


namespace NUMINAMATH_CALUDE_age_difference_l893_89307

/-- Represents a person's age at different points in time -/
structure AgeRelation where
  current : ℕ
  future : ℕ

/-- The age relation between two people A and B -/
def age_relation (a b : AgeRelation) : Prop :=
  a.current - b.current = b.current - 10 ∧
  a.current - b.current = 25 - a.future

theorem age_difference (a b : AgeRelation) 
  (h : age_relation a b) : a.current - b.current = 5 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l893_89307


namespace NUMINAMATH_CALUDE_sum_of_factors_of_125_l893_89365

theorem sum_of_factors_of_125 :
  ∃ (a b c : ℕ+),
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (a.val * b.val * c.val = 125) ∧
    (a.val + b.val + c.val = 31) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_factors_of_125_l893_89365


namespace NUMINAMATH_CALUDE_coins_in_first_stack_l893_89324

theorem coins_in_first_stack (total : ℕ) (stack2 : ℕ) (h1 : total = 12) (h2 : stack2 = 8) :
  total - stack2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_coins_in_first_stack_l893_89324


namespace NUMINAMATH_CALUDE_sqrt_yz_times_sqrt_xy_l893_89369

theorem sqrt_yz_times_sqrt_xy (x y z : ℝ) (hx : x = 3) (hy : y = 4) (hz : z = 5) :
  Real.sqrt (y * z) * Real.sqrt (x * y) = 4 * Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_yz_times_sqrt_xy_l893_89369


namespace NUMINAMATH_CALUDE_quadratic_root_interval_l893_89316

theorem quadratic_root_interval (a b : ℝ) (hb : b > 0) 
  (h_distinct : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + a*x₁ + b = 0 ∧ x₂^2 + a*x₂ + b = 0)
  (h_one_in_unit : ∃! x : ℝ, x^2 + a*x + b = 0 ∧ x ∈ Set.Icc (-1) 1) :
  ∃! x : ℝ, x^2 + a*x + b = 0 ∧ x ∈ Set.Ioo (-b) b :=
sorry

end NUMINAMATH_CALUDE_quadratic_root_interval_l893_89316


namespace NUMINAMATH_CALUDE_sin_cos_product_l893_89325

theorem sin_cos_product (θ : Real) (h : Real.tan (θ + Real.pi / 2) = 2) :
  Real.sin θ * Real.cos θ = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_product_l893_89325


namespace NUMINAMATH_CALUDE_math_contest_theorem_l893_89338

theorem math_contest_theorem (n m k : ℕ) (h_n : n = 200) (h_m : m = 6) (h_k : k = 120)
  (solved : Fin n → Fin m → Prop)
  (h_solved : ∀ j : Fin m, ∃ S : Finset (Fin n), S.card ≥ k ∧ ∀ i ∈ S, solved i j) :
  ∃ i₁ i₂ : Fin n, i₁ ≠ i₂ ∧ ∀ j : Fin m, solved i₁ j ∨ solved i₂ j := by
  sorry

end NUMINAMATH_CALUDE_math_contest_theorem_l893_89338


namespace NUMINAMATH_CALUDE_set_operations_l893_89390

open Set

-- Define the sets A and B
def A : Set ℝ := {x | -5 < x ∧ x < 2}
def B : Set ℝ := {x | -3 < x ∧ x ≤ 3}

-- Define the theorem
theorem set_operations :
  (A ∩ B = Ioc (-3) 2) ∧
  (A ∪ B = Ioc (-5) 3) ∧
  (Aᶜ = Iic (-5) ∪ Ici 2) ∧
  ((A ∩ B)ᶜ = Iic (-3) ∪ Ioi 2) ∧
  (Aᶜ ∩ B = Icc 2 3) :=
by sorry

end NUMINAMATH_CALUDE_set_operations_l893_89390


namespace NUMINAMATH_CALUDE_geometric_sequence_a12_l893_89333

/-- A geometric sequence (aₙ) -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_a12 (a : ℕ → ℝ) :
  geometric_sequence a → a 4 = 4 → a 8 = 8 → a 12 = 16 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a12_l893_89333


namespace NUMINAMATH_CALUDE_amoeba_survival_l893_89317

/-- Represents the state of an amoeba with pseudopods and nuclei -/
structure Amoeba where
  pseudopods : Int
  nuclei : Int

/-- Mutation function for an amoeba -/
def mutate (a : Amoeba) : Amoeba :=
  { pseudopods := 2 * a.pseudopods - a.nuclei,
    nuclei := 2 * a.nuclei - a.pseudopods }

/-- Predicate to check if an amoeba is alive -/
def isAlive (a : Amoeba) : Prop :=
  a.pseudopods ≥ 0 ∧ a.nuclei ≥ 0

/-- Theorem stating that only amoebas with equal initial pseudopods and nuclei survive indefinitely -/
theorem amoeba_survival (a : Amoeba) :
  (∀ n : ℕ, isAlive ((mutate^[n]) a)) ↔ a.pseudopods = a.nuclei :=
sorry

end NUMINAMATH_CALUDE_amoeba_survival_l893_89317


namespace NUMINAMATH_CALUDE_average_pieces_lost_l893_89389

def audrey_pieces : List ℕ := [6, 8, 4, 7, 10]
def thomas_pieces : List ℕ := [5, 6, 3, 7, 11]
def num_games : ℕ := 5

theorem average_pieces_lost (audrey_pieces : List ℕ) (thomas_pieces : List ℕ) (num_games : ℕ) :
  audrey_pieces = [6, 8, 4, 7, 10] →
  thomas_pieces = [5, 6, 3, 7, 11] →
  num_games = 5 →
  (audrey_pieces.sum + thomas_pieces.sum : ℚ) / num_games = 13.4 := by
  sorry

end NUMINAMATH_CALUDE_average_pieces_lost_l893_89389


namespace NUMINAMATH_CALUDE_problem_solution_l893_89393

-- Define proposition p
def p : Prop := ∀ (x a : ℝ), x^2 + a*x + a^2 ≥ 0

-- Define proposition q
def q : Prop := ∃ (x : ℕ), x > 0 ∧ 2*x^2 - 1 ≤ 0

-- Theorem to prove
theorem problem_solution :
  p ∧ ¬q ∧ (p ∨ q) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l893_89393


namespace NUMINAMATH_CALUDE_line_through_intersection_and_origin_l893_89311

/-- Given two lines in the plane -/
def line1 (x y : ℝ) : ℝ := 2023 * x - 2022 * y - 1
def line2 (x y : ℝ) : ℝ := 2022 * x + 2023 * y + 1

/-- The equation of the line we want to prove -/
def target_line (x y : ℝ) : ℝ := 4045 * x + y

theorem line_through_intersection_and_origin :
  ∃ (x₀ y₀ : ℝ),
    (line1 x₀ y₀ = 0 ∧ line2 x₀ y₀ = 0) ∧  -- Intersection point
    (target_line 0 0 = 0) ∧                -- Passes through origin
    (∀ (x y : ℝ), line1 x y = 0 ∧ line2 x y = 0 → target_line x y = 0)
    -- The target line passes through the intersection
:= by sorry

end NUMINAMATH_CALUDE_line_through_intersection_and_origin_l893_89311


namespace NUMINAMATH_CALUDE_train_length_l893_89386

/-- The length of a train given its speed, time to pass a platform, and platform length -/
theorem train_length (train_speed : ℝ) (passing_time : ℝ) (platform_length : ℝ) :
  train_speed = 60 * (1000 / 3600) →
  passing_time = 21.598272138228943 →
  platform_length = 240 →
  train_speed * passing_time - platform_length = 120 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l893_89386


namespace NUMINAMATH_CALUDE_land_increase_percentage_l893_89383

theorem land_increase_percentage (A B C D E : ℝ) 
  (h1 : B = 1.5 * A)
  (h2 : C = 2 * A)
  (h3 : D = 2.5 * A)
  (h4 : E = 3 * A)
  (h5 : A > 0) :
  let initial_area := A + B + C + D + E
  let increase := 0.1 * A + (1 / 15) * B + 0.05 * C + 0.04 * D + (1 / 30) * E
  increase / initial_area = 0.05 := by
sorry

end NUMINAMATH_CALUDE_land_increase_percentage_l893_89383


namespace NUMINAMATH_CALUDE_prime_fraction_solutions_l893_89384

theorem prime_fraction_solutions (x y : ℕ) :
  (x > 0 ∧ y > 0) →
  (∃ p : ℕ, Nat.Prime p ∧ x * y^2 = p * (x + y)) ↔ 
  ((x = 2 ∧ y = 2) ∨ (x = 6 ∧ y = 2)) := by
sorry

end NUMINAMATH_CALUDE_prime_fraction_solutions_l893_89384


namespace NUMINAMATH_CALUDE_odd_function_period_two_value_at_six_l893_89312

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_period_two_value_at_six
  (f : ℝ → ℝ)
  (h_odd : IsOdd f)
  (h_period : ∀ x, f (x + 2) = -f x) :
  f 6 = 0 := by
sorry

end NUMINAMATH_CALUDE_odd_function_period_two_value_at_six_l893_89312


namespace NUMINAMATH_CALUDE_curve_C_perpendicular_points_sum_l893_89385

/-- The curve C in polar coordinates -/
def C (ρ θ : ℝ) : Prop := ρ^2 = 1 / (3 * (Real.cos θ)^2 + (Real.sin θ)^2)

/-- The curve C in Cartesian coordinates -/
def C_cartesian (x y : ℝ) : Prop := 3 * x^2 + y^2 = 1

/-- Two points are perpendicular with respect to the origin -/
def perpendicular (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁ * x₂ + y₁ * y₂ = 0

theorem curve_C_perpendicular_points_sum (x₁ y₁ x₂ y₂ : ℝ) :
  C_cartesian x₁ y₁ → C_cartesian x₂ y₂ → perpendicular x₁ y₁ x₂ y₂ →
  1 / (x₁^2 + y₁^2) + 1 / (x₂^2 + y₂^2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_curve_C_perpendicular_points_sum_l893_89385


namespace NUMINAMATH_CALUDE_max_area_equilateral_triangle_in_rectangle_l893_89313

/-- The maximum area of an equilateral triangle inscribed in a 12x17 rectangle --/
theorem max_area_equilateral_triangle_in_rectangle : 
  ∃ (A : ℝ), A = 325 * Real.sqrt 3 - 612 ∧ 
  ∀ (triangle_area : ℝ), 
    (∃ (x y : ℝ), 
      0 ≤ x ∧ x ≤ 12 ∧ 
      0 ≤ y ∧ y ≤ 17 ∧ 
      triangle_area = (Real.sqrt 3 / 4) * (x^2 + y^2)) →
    triangle_area ≤ A :=
by sorry

end NUMINAMATH_CALUDE_max_area_equilateral_triangle_in_rectangle_l893_89313


namespace NUMINAMATH_CALUDE_bench_press_calculation_l893_89371

theorem bench_press_calculation (initial_weight : ℝ) (injury_decrease : ℝ) (training_increase : ℝ) : 
  initial_weight = 500 →
  injury_decrease = 0.8 →
  training_increase = 3 →
  (initial_weight * (1 - injury_decrease) * training_increase) = 300 := by
sorry

end NUMINAMATH_CALUDE_bench_press_calculation_l893_89371


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l893_89364

/-- Given an arithmetic sequence {a_n} where a_4 = 4, prove that S_7 = 28 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, S n = (n : ℝ) / 2 * (a 1 + a n)) →  -- Definition of S_n
  (∀ k m, a (k + m) - a k = m * (a 2 - a 1)) →  -- Definition of arithmetic sequence
  a 4 = 4 →  -- Given condition
  S 7 = 28 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l893_89364


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l893_89348

/-- Given a circle with equation x^2 + y^2 + 2x - 4y - 4 = 0, 
    its center is at (-1, 2) and its radius is 3. -/
theorem circle_center_and_radius :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (-1, 2) ∧ 
    radius = 3 ∧
    ∀ (x y : ℝ), x^2 + y^2 + 2*x - 4*y - 4 = 0 ↔ 
      (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l893_89348


namespace NUMINAMATH_CALUDE_problems_per_worksheet_l893_89376

/-- Given a set of worksheets with some graded and some problems left to grade,
    calculate the number of problems per worksheet. -/
theorem problems_per_worksheet
  (total_worksheets : ℕ)
  (graded_worksheets : ℕ)
  (remaining_problems : ℕ)
  (h1 : total_worksheets = 16)
  (h2 : graded_worksheets = 8)
  (h3 : remaining_problems = 32)
  : (remaining_problems / (total_worksheets - graded_worksheets) : ℚ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_problems_per_worksheet_l893_89376


namespace NUMINAMATH_CALUDE_cylinder_lateral_area_not_base_area_times_height_l893_89362

/-- The lateral area of a cylinder is not equal to the base area multiplied by the height. -/
theorem cylinder_lateral_area_not_base_area_times_height 
  (r h : ℝ) (r_pos : 0 < r) (h_pos : 0 < h) :
  2 * π * r * h ≠ (π * r^2) * h := by sorry

end NUMINAMATH_CALUDE_cylinder_lateral_area_not_base_area_times_height_l893_89362


namespace NUMINAMATH_CALUDE_power_of_power_l893_89336

theorem power_of_power (a : ℝ) : (a ^ 2) ^ 3 = a ^ 6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l893_89336


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l893_89352

theorem perfect_square_trinomial (m : ℚ) : 
  (∃ a b : ℚ, ∀ x, 4*x^2 - (2*m+1)*x + 121 = (a*x + b)^2) → 
  (m = 43/2 ∨ m = -45/2) :=
sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l893_89352


namespace NUMINAMATH_CALUDE_middle_number_in_ratio_l893_89314

theorem middle_number_in_ratio (a b c : ℝ) : 
  a / b = 3 / 2 ∧ 
  b / c = 2 / 5 ∧ 
  a^2 + b^2 + c^2 = 1862 → 
  b = 14 := by
sorry

end NUMINAMATH_CALUDE_middle_number_in_ratio_l893_89314


namespace NUMINAMATH_CALUDE_jonas_bookshelves_l893_89379

/-- Calculates the maximum number of bookshelves that can fit in a room. -/
def max_bookshelves (total_space : ℕ) (reserved_space : ℕ) (shelf_space : ℕ) : ℕ :=
  (total_space - reserved_space) / shelf_space

/-- Proves that given the specific conditions, the maximum number of bookshelves is 3. -/
theorem jonas_bookshelves :
  max_bookshelves 400 160 80 = 3 := by
  sorry

end NUMINAMATH_CALUDE_jonas_bookshelves_l893_89379


namespace NUMINAMATH_CALUDE_price_reduction_l893_89319

theorem price_reduction (original_price final_price : ℝ) (x : ℝ) 
  (h1 : original_price = 120)
  (h2 : final_price = 85)
  (h3 : x > 0 ∧ x < 1) -- Assuming x is a valid percentage
  (h4 : final_price = original_price * (1 - x)^2) :
  120 * (1 - x)^2 = 85 := by sorry

end NUMINAMATH_CALUDE_price_reduction_l893_89319


namespace NUMINAMATH_CALUDE_kerrys_age_l893_89334

/-- Given the conditions of Kerry's birthday candles, prove his age --/
theorem kerrys_age (num_cakes : ℕ) (candles_per_box : ℕ) (cost_per_box : ℚ) (total_cost : ℚ) :
  num_cakes = 5 →
  candles_per_box = 22 →
  cost_per_box = 9/2 →
  total_cost = 27 →
  ∃ (age : ℕ), age = 26 ∧ (num_cakes * age : ℚ) ≤ (total_cost / cost_per_box * candles_per_box) :=
by sorry

end NUMINAMATH_CALUDE_kerrys_age_l893_89334


namespace NUMINAMATH_CALUDE_smiley_red_smile_l893_89396

def tulip_smiley (red_smile : ℕ) : Prop :=
  let red_eyes : ℕ := 8 * 2
  let yellow_background : ℕ := 9 * red_smile
  red_eyes + red_smile + yellow_background = 196

theorem smiley_red_smile :
  ∃ (red_smile : ℕ), tulip_smiley red_smile ∧ red_smile = 18 :=
by sorry

end NUMINAMATH_CALUDE_smiley_red_smile_l893_89396


namespace NUMINAMATH_CALUDE_tangent_line_and_critical_point_l893_89300

/-- The function f(x) = (1/2)x^2 - ax - ln(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - a*x - Real.log x

/-- The derivative of f(x) -/
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := x - a - 1/x

theorem tangent_line_and_critical_point (a : ℝ) (h : a ≥ 0) :
  /- The equation of the tangent line to f(x) at x=1 when a=1 is y = -x + 1/2 -/
  (let y : ℝ → ℝ := fun x ↦ -x + 1/2
   f 1 1 = y 1 ∧ f' 1 1 = -1) ∧
  /- For any critical point x₀ of f(x), f(x₀) ≤ 1/2 -/
  ∀ x₀ > 0, f' a x₀ = 0 → f a x₀ ≤ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_and_critical_point_l893_89300


namespace NUMINAMATH_CALUDE_circle_to_square_impossible_l893_89375

/-- Represents a piece of paper with a boundary --/
structure PaperPiece where
  boundary : Set ℝ × ℝ

/-- Represents a cut on a paper piece --/
inductive Cut
  | StraightLine : (ℝ × ℝ) → (ℝ × ℝ) → Cut
  | CircularArc : (ℝ × ℝ) → ℝ → ℝ → ℝ → Cut

/-- Represents a transformation of paper pieces --/
def Transform := List PaperPiece → List PaperPiece

/-- Checks if a shape is a circle --/
def is_circle (p : PaperPiece) : Prop := sorry

/-- Checks if a shape is a square --/
def is_square (p : PaperPiece) : Prop := sorry

/-- Calculates the area of a paper piece --/
def area (p : PaperPiece) : ℝ := sorry

/-- Theorem stating the impossibility of transforming a circle to a square of equal area --/
theorem circle_to_square_impossible 
  (initial : PaperPiece) 
  (cuts : List Cut) 
  (transform : Transform) :
  is_circle initial →
  (∃ final, is_square final ∧ area final = area initial ∧ 
    transform [initial] = final :: (transform [initial]).tail) →
  False := by
  sorry

#check circle_to_square_impossible

end NUMINAMATH_CALUDE_circle_to_square_impossible_l893_89375


namespace NUMINAMATH_CALUDE_fudge_piece_size_l893_89345

/-- Given a rectangular pan of fudge with dimensions 18 inches by 29 inches,
    containing 522 square pieces, prove that each piece has a side length of 1 inch. -/
theorem fudge_piece_size (pan_length : ℝ) (pan_width : ℝ) (num_pieces : ℕ) 
    (h1 : pan_length = 18) 
    (h2 : pan_width = 29) 
    (h3 : num_pieces = 522) : 
  (pan_length * pan_width) / num_pieces = 1 := by
  sorry

#check fudge_piece_size

end NUMINAMATH_CALUDE_fudge_piece_size_l893_89345


namespace NUMINAMATH_CALUDE_second_last_digit_of_power_of_three_is_even_l893_89303

/-- The second-to-last digit of a natural number -/
def secondLastDigit (n : ℕ) : ℕ := (n / 10) % 10

/-- A natural number is even if it's divisible by 2 -/
def isEven (n : ℕ) : Prop := ∃ k, n = 2 * k

theorem second_last_digit_of_power_of_three_is_even (n : ℕ) (h : n > 2) :
  isEven (secondLastDigit (3^n)) := by sorry

end NUMINAMATH_CALUDE_second_last_digit_of_power_of_three_is_even_l893_89303


namespace NUMINAMATH_CALUDE_subsets_and_sum_of_M_l893_89357

def M : Finset Nat := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

theorem subsets_and_sum_of_M :
  (Finset.powerset M).card = 2^10 ∧
  (Finset.powerset M).sum (λ s => s.sum id) = 55 * 2^9 := by
  sorry

end NUMINAMATH_CALUDE_subsets_and_sum_of_M_l893_89357


namespace NUMINAMATH_CALUDE_base7_subtraction_l893_89321

/-- Converts a base 7 number represented as a list of digits to its decimal equivalent -/
def base7ToDecimal (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 7 * acc + d) 0

/-- Converts a decimal number to its base 7 representation as a list of digits -/
def decimalToBase7 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
  aux n []

/-- The statement to be proved -/
theorem base7_subtraction :
  let a := base7ToDecimal [2, 5, 3, 4]
  let b := base7ToDecimal [1, 4, 6, 6]
  decimalToBase7 (a - b) = [1, 0, 6, 5] := by
  sorry

end NUMINAMATH_CALUDE_base7_subtraction_l893_89321


namespace NUMINAMATH_CALUDE_remainder_equality_l893_89363

theorem remainder_equality (A B D S S' s s' : ℕ) : 
  A > B →
  (A + 3) % D = S →
  (B - 2) % D = S' →
  ((A + 3) * (B - 2)) % D = s →
  (S * S') % D = s' →
  s = s' := by sorry

end NUMINAMATH_CALUDE_remainder_equality_l893_89363


namespace NUMINAMATH_CALUDE_cylinder_volume_change_l893_89358

/-- Given a cylinder with volume 15 cubic meters, if its radius is tripled
    and its height is doubled, then its new volume is 270 cubic meters. -/
theorem cylinder_volume_change (r h : ℝ) (h1 : r > 0) (h2 : h > 0) :
  π * r^2 * h = 15 → π * (3*r)^2 * (2*h) = 270 := by sorry

end NUMINAMATH_CALUDE_cylinder_volume_change_l893_89358


namespace NUMINAMATH_CALUDE_intersection_distance_squared_l893_89340

/-- Given two circles in a 2D plane, one centered at (1,1) with radius 5 
and another centered at (1,-8) with radius √26, this theorem states that 
the square of the distance between their intersection points is 3128/81. -/
theorem intersection_distance_squared : 
  ∃ (C D : ℝ × ℝ), 
    ((C.1 - 1)^2 + (C.2 - 1)^2 = 25) ∧ 
    ((D.1 - 1)^2 + (D.2 - 1)^2 = 25) ∧
    ((C.1 - 1)^2 + (C.2 + 8)^2 = 26) ∧ 
    ((D.1 - 1)^2 + (D.2 + 8)^2 = 26) ∧
    ((C.1 - D.1)^2 + (C.2 - D.2)^2 = 3128 / 81) :=
by sorry


end NUMINAMATH_CALUDE_intersection_distance_squared_l893_89340


namespace NUMINAMATH_CALUDE_green_mandm_probability_l893_89330

/-- Represents the count of M&Ms of each color -/
structure MandMCount where
  green : ℕ
  red : ℕ
  blue : ℕ
  orange : ℕ
  yellow : ℕ
  purple : ℕ
  brown : ℕ

/-- Calculates the total count of M&Ms -/
def totalCount (count : MandMCount) : ℕ :=
  count.green + count.red + count.blue + count.orange + count.yellow + count.purple + count.brown

/-- Represents the actions taken by Carter and others -/
def finalCount : MandMCount :=
  let initial := MandMCount.mk 35 25 10 15 0 0 0
  let afterCarter := MandMCount.mk (initial.green - 20) (initial.red - 8) initial.blue initial.orange 0 0 0
  let afterSister := MandMCount.mk afterCarter.green (afterCarter.red / 2) (afterCarter.blue - 5) afterCarter.orange 14 0 0
  let afterAlex := MandMCount.mk afterSister.green afterSister.red afterSister.blue (afterSister.orange - 7) (afterSister.yellow - 3) 8 0
  MandMCount.mk afterAlex.green afterAlex.red 0 afterAlex.orange afterAlex.yellow afterAlex.purple 10

/-- The main theorem to prove -/
theorem green_mandm_probability :
  (finalCount.green : ℚ) / (totalCount finalCount : ℚ) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_green_mandm_probability_l893_89330


namespace NUMINAMATH_CALUDE_song_circle_l893_89341

theorem song_circle (S : Finset Nat) (covers : Finset Nat → Finset Nat)
  (h_card : S.card = 12)
  (h_cover_10 : ∀ T ⊆ S, T.card = 10 → (covers T).card = 20)
  (h_cover_8 : ∀ T ⊆ S, T.card = 8 → (covers T).card = 16) :
  (covers S).card = 24 := by
  sorry

end NUMINAMATH_CALUDE_song_circle_l893_89341


namespace NUMINAMATH_CALUDE_smallest_890_multiple_of_18_l893_89318

def is_digit_890 (d : ℕ) : Prop := d = 8 ∨ d = 9 ∨ d = 0

def all_digits_890 (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → is_digit_890 d

theorem smallest_890_multiple_of_18 :
  ∃! m : ℕ, m > 0 ∧ m % 18 = 0 ∧ all_digits_890 m ∧
  ∀ n : ℕ, n > 0 → n % 18 = 0 → all_digits_890 n → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_890_multiple_of_18_l893_89318


namespace NUMINAMATH_CALUDE_equation_solutions_l893_89355

theorem equation_solutions :
  (∀ x : ℝ, (1/2 * x^2 = 5) ↔ (x = Real.sqrt 10 ∨ x = -Real.sqrt 10)) ∧
  (∀ x : ℝ, ((x - 1)^2 = 16) ↔ (x = 5 ∨ x = -3)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l893_89355


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l893_89315

-- Define the quadratic function
def f (a x : ℝ) : ℝ := x^2 - (2 + a) * x + 2 * a

-- Define the solution set
def solution_set (a : ℝ) : Set ℝ := {x | f a x < 0}

-- Theorem statement
theorem quadratic_inequality_solution (a : ℝ) :
  (a < 2 → solution_set a = {x | a < x ∧ x < 2}) ∧
  (a = 2 → solution_set a = ∅) ∧
  (a > 2 → solution_set a = {x | 2 < x ∧ x < a}) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l893_89315


namespace NUMINAMATH_CALUDE_altitude_divides_triangle_iff_right_angle_or_isosceles_l893_89373

/-- Triangle ABC with altitude h_a from vertex A to side BC -/
structure Triangle :=
  (A B C : Point)
  (h_a : Point)

/-- The altitude h_a divides triangle ABC into two similar triangles -/
def divides_into_similar_triangles (t : Triangle) : Prop :=
  sorry

/-- Angle A is a right angle -/
def is_right_angle_at_A (t : Triangle) : Prop :=
  sorry

/-- Triangle ABC is isosceles with AB = AC -/
def is_isosceles (t : Triangle) : Prop :=
  sorry

/-- Theorem: The altitude h_a of triangle ABC divides it into two similar triangles
    if and only if either angle A is a right angle or AB = AC -/
theorem altitude_divides_triangle_iff_right_angle_or_isosceles (t : Triangle) :
  divides_into_similar_triangles t ↔ (is_right_angle_at_A t ∨ is_isosceles t) :=
sorry

end NUMINAMATH_CALUDE_altitude_divides_triangle_iff_right_angle_or_isosceles_l893_89373


namespace NUMINAMATH_CALUDE_gcd_linear_combination_l893_89304

theorem gcd_linear_combination (a b : ℤ) : 
  Int.gcd (5*a + 3*b) (13*a + 8*b) = Int.gcd a b := by sorry

end NUMINAMATH_CALUDE_gcd_linear_combination_l893_89304


namespace NUMINAMATH_CALUDE_complex_coordinate_of_reciprocal_i_cubed_l893_89397

theorem complex_coordinate_of_reciprocal_i_cubed :
  let z : ℂ := (Complex.I ^ 3)⁻¹
  z = Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_coordinate_of_reciprocal_i_cubed_l893_89397


namespace NUMINAMATH_CALUDE_calculation_proof_l893_89326

theorem calculation_proof : -50 * 3 - (-2.5) / 0.1 = -125 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l893_89326


namespace NUMINAMATH_CALUDE_sphere_volume_increase_l893_89347

theorem sphere_volume_increase (r : ℝ) (h : r > 0) :
  let new_r := r * Real.sqrt 2
  (4 / 3 * Real.pi * new_r^3) / (4 / 3 * Real.pi * r^3) = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_increase_l893_89347


namespace NUMINAMATH_CALUDE_root_sum_squares_implies_h_abs_one_l893_89320

theorem root_sum_squares_implies_h_abs_one (h : ℝ) : 
  (∃ r s : ℝ, r^2 + 6*h*r + 8 = 0 ∧ s^2 + 6*h*s + 8 = 0 ∧ r^2 + s^2 = 20) → 
  |h| = 1 := by
sorry

end NUMINAMATH_CALUDE_root_sum_squares_implies_h_abs_one_l893_89320


namespace NUMINAMATH_CALUDE_print_statement_output_l893_89306

def print_output (a : ℕ) : String := s!"a={a}"

theorem print_statement_output (a : ℕ) (h : a = 10) : print_output a = "a=10" := by
  sorry

end NUMINAMATH_CALUDE_print_statement_output_l893_89306


namespace NUMINAMATH_CALUDE_total_methods_is_fifteen_l893_89380

/-- A two-stage test with options for each stage -/
structure TwoStageTest where
  first_stage_options : Nat
  second_stage_options : Nat

/-- Calculate the total number of testing methods for a two-stage test -/
def total_testing_methods (test : TwoStageTest) : Nat :=
  test.first_stage_options * test.second_stage_options

/-- The specific test configuration -/
def our_test : TwoStageTest :=
  { first_stage_options := 3
  , second_stage_options := 5 }

theorem total_methods_is_fifteen :
  total_testing_methods our_test = 15 := by
  sorry

#eval total_testing_methods our_test

end NUMINAMATH_CALUDE_total_methods_is_fifteen_l893_89380


namespace NUMINAMATH_CALUDE_valid_coloring_iff_even_product_l893_89391

/-- Represents a chessboard coloring where each small square not on the perimeter has exactly two sides colored. -/
def ValidColoring (m n : ℕ) := True  -- Placeholder definition

/-- Theorem stating that a valid coloring exists if and only if m * n is even -/
theorem valid_coloring_iff_even_product (m n : ℕ) :
  ValidColoring m n ↔ Even (m * n) :=
by sorry

end NUMINAMATH_CALUDE_valid_coloring_iff_even_product_l893_89391


namespace NUMINAMATH_CALUDE_ammonia_formation_l893_89381

/-- Represents the chemical reaction between Potassium hydroxide and Ammonium iodide -/
structure ChemicalReaction where
  koh : ℝ  -- moles of Potassium hydroxide
  nh4i : ℝ  -- moles of Ammonium iodide
  nh3 : ℝ  -- moles of Ammonia formed

/-- Theorem stating that the moles of Ammonia formed equals the moles of Ammonium iodide used -/
theorem ammonia_formation (reaction : ChemicalReaction) 
  (h1 : reaction.nh4i = 3)  -- 3 moles of Ammonium iodide are used
  (h2 : reaction.nh3 = 3)   -- The total moles of Ammonia formed is 3
  : reaction.nh3 = reaction.nh4i := by
  sorry


end NUMINAMATH_CALUDE_ammonia_formation_l893_89381


namespace NUMINAMATH_CALUDE_circle_radius_is_two_l893_89366

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 8*x + y^2 - 4*y + 16 = 0

/-- The radius of the circle -/
def circle_radius : ℝ := 2

/-- Theorem: The radius of the circle with the given equation is 2 -/
theorem circle_radius_is_two :
  ∃ (h k : ℝ), ∀ (x y : ℝ), circle_equation x y ↔ (x - h)^2 + (y - k)^2 = circle_radius^2 :=
sorry

end NUMINAMATH_CALUDE_circle_radius_is_two_l893_89366


namespace NUMINAMATH_CALUDE_plot_length_is_65_l893_89368

/-- Represents a rectangular plot with its dimensions and fencing cost. -/
structure RectangularPlot where
  breadth : ℝ
  length : ℝ
  fencingCostPerMeter : ℝ
  totalFencingCost : ℝ

/-- The length of the plot is 30 meters more than its breadth. -/
def lengthCondition (plot : RectangularPlot) : Prop :=
  plot.length = plot.breadth + 30

/-- The cost of fencing the plot at the given rate equals the total fencing cost. -/
def fencingCostCondition (plot : RectangularPlot) : Prop :=
  plot.fencingCostPerMeter * (2 * plot.length + 2 * plot.breadth) = plot.totalFencingCost

/-- The main theorem stating that under the given conditions, the length of the plot is 65 meters. -/
theorem plot_length_is_65 (plot : RectangularPlot) 
    (h1 : lengthCondition plot) 
    (h2 : fencingCostCondition plot) 
    (h3 : plot.fencingCostPerMeter = 26.5) 
    (h4 : plot.totalFencingCost = 5300) : 
  plot.length = 65 := by
  sorry

end NUMINAMATH_CALUDE_plot_length_is_65_l893_89368


namespace NUMINAMATH_CALUDE_weekly_coffee_cost_household_weekly_coffee_cost_l893_89335

/-- Calculates the weekly cost of coffee for a household -/
theorem weekly_coffee_cost 
  (people : ℕ) 
  (cups_per_person : ℕ) 
  (ounces_per_cup : ℚ) 
  (cost_per_ounce : ℚ) : ℚ :=
  let daily_cups := people * cups_per_person
  let daily_ounces := daily_cups * ounces_per_cup
  let weekly_ounces := daily_ounces * 7
  weekly_ounces * cost_per_ounce

/-- Proves that the weekly coffee cost for the given household is $35 -/
theorem household_weekly_coffee_cost : 
  weekly_coffee_cost 4 2 (1/2) (5/4) = 35 := by
  sorry

end NUMINAMATH_CALUDE_weekly_coffee_cost_household_weekly_coffee_cost_l893_89335


namespace NUMINAMATH_CALUDE_soda_can_ratio_l893_89349

theorem soda_can_ratio :
  let initial_cans : ℕ := 22
  let taken_cans : ℕ := 6
  let final_cans : ℕ := 24
  let remaining_cans := initial_cans - taken_cans
  let bought_cans := final_cans - remaining_cans
  (bought_cans : ℚ) / remaining_cans = 1 / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_soda_can_ratio_l893_89349


namespace NUMINAMATH_CALUDE_yellow_parrots_count_l893_89331

theorem yellow_parrots_count (total : ℕ) (red_fraction : ℚ) (green_fraction : ℚ) :
  total = 180 →
  red_fraction = 2/3 →
  green_fraction = 1/6 →
  (total : ℚ) * (1 - (red_fraction + green_fraction)) = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_yellow_parrots_count_l893_89331


namespace NUMINAMATH_CALUDE_bob_fruit_drink_cost_l893_89309

/-- The cost of Bob's fruit drink -/
def fruit_drink_cost (andy_total bob_sandwich_cost : ℕ) : ℕ :=
  andy_total - bob_sandwich_cost

theorem bob_fruit_drink_cost :
  let andy_total := 5
  let bob_sandwich_cost := 3
  fruit_drink_cost andy_total bob_sandwich_cost = 2 := by
  sorry

end NUMINAMATH_CALUDE_bob_fruit_drink_cost_l893_89309


namespace NUMINAMATH_CALUDE_nested_sqrt_range_l893_89350

theorem nested_sqrt_range :
  ∃ y : ℝ, y = Real.sqrt (4 + y) ∧ 2 ≤ y ∧ y < 3 := by
  sorry

end NUMINAMATH_CALUDE_nested_sqrt_range_l893_89350


namespace NUMINAMATH_CALUDE_cone_surface_area_l893_89367

/-- Given a cone with slant height 4 and cross-sectional area π, 
    its total surface area is 12π. -/
theorem cone_surface_area (s : ℝ) (a : ℝ) (h1 : s = 4) (h2 : a = π) :
  let r := Real.sqrt (a / π)
  let lateral_area := π * r * s
  let base_area := a
  lateral_area + base_area = 12 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_surface_area_l893_89367


namespace NUMINAMATH_CALUDE_polygonE_has_largest_area_l893_89399

/-- Represents a polygon composed of unit squares and right triangles -/
structure Polygon where
  unitSquares : ℕ
  rightTriangles : ℕ

/-- Calculates the area of a polygon -/
def areaOfPolygon (p : Polygon) : ℝ :=
  p.unitSquares + 0.5 * p.rightTriangles

/-- The polygons given in the problem -/
def polygonA : Polygon := ⟨6, 0⟩
def polygonB : Polygon := ⟨3, 2⟩
def polygonC : Polygon := ⟨4, 4⟩
def polygonD : Polygon := ⟨5, 0⟩
def polygonE : Polygon := ⟨6, 2⟩

/-- The list of all polygons -/
def allPolygons : List Polygon := [polygonA, polygonB, polygonC, polygonD, polygonE]

theorem polygonE_has_largest_area :
  ∀ p ∈ allPolygons, areaOfPolygon polygonE ≥ areaOfPolygon p :=
by sorry

#eval areaOfPolygon polygonE -- Should output 7

end NUMINAMATH_CALUDE_polygonE_has_largest_area_l893_89399


namespace NUMINAMATH_CALUDE_sin_plus_cos_for_point_l893_89305

/-- Theorem: If the terminal side of angle α passes through point P(-4,3), then sin α + cos α = -1/5 -/
theorem sin_plus_cos_for_point (α : Real) : 
  (∃ (x y : Real), x = -4 ∧ y = 3 ∧ Real.cos α = x / Real.sqrt (x^2 + y^2) ∧ Real.sin α = y / Real.sqrt (x^2 + y^2)) → 
  Real.sin α + Real.cos α = -1/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_plus_cos_for_point_l893_89305


namespace NUMINAMATH_CALUDE_familyReunionHandshakesCount_l893_89332

/-- Represents the number of handshakes at a family reunion --/
def familyReunionHandshakes : ℕ :=
  let quadrupletSets : ℕ := 12
  let quintupletSets : ℕ := 4
  let quadrupletsPerSet : ℕ := 4
  let quintupletsPerSet : ℕ := 5
  let totalQuadruplets : ℕ := quadrupletSets * quadrupletsPerSet
  let totalQuintuplets : ℕ := quintupletSets * quintupletsPerSet
  let quadrupletHandshakes : ℕ := totalQuadruplets * (totalQuadruplets - quadrupletsPerSet)
  let quintupletHandshakes : ℕ := totalQuintuplets * (totalQuintuplets - quintupletsPerSet)
  let crossHandshakes : ℕ := totalQuadruplets * 7 + totalQuintuplets * 12
  (quadrupletHandshakes + quintupletHandshakes + crossHandshakes) / 2

/-- Theorem stating that the number of handshakes at the family reunion is 1494 --/
theorem familyReunionHandshakesCount : familyReunionHandshakes = 1494 := by
  sorry

end NUMINAMATH_CALUDE_familyReunionHandshakesCount_l893_89332


namespace NUMINAMATH_CALUDE_circle_symmetry_line_l893_89395

/-- A circle in the Cartesian plane -/
structure Circle where
  equation : ℝ → ℝ → Prop

/-- A line in the Cartesian plane -/
structure Line where
  equation : ℝ → ℝ → Prop

/-- Symmetry of a circle with respect to a line -/
def isSymmetric (c : Circle) (l : Line) : Prop := sorry

/-- The main theorem -/
theorem circle_symmetry_line (a : ℝ) :
  let c : Circle := { equation := fun x y => x^2 + y^2 - 4*x - 8*y + 19 = 0 }
  let l : Line := { equation := fun x y => x + 2*y - a = 0 }
  isSymmetric c l → a = 10 := by sorry

end NUMINAMATH_CALUDE_circle_symmetry_line_l893_89395


namespace NUMINAMATH_CALUDE_pool_filling_time_l893_89329

/-- Proves the time required to fill a pool given the pool capacity, bucket size, and time per trip -/
theorem pool_filling_time 
  (pool_capacity : ℕ) 
  (bucket_size : ℕ) 
  (seconds_per_trip : ℕ) 
  (h1 : pool_capacity = 84)
  (h2 : bucket_size = 2)
  (h3 : seconds_per_trip = 20) :
  (pool_capacity / bucket_size) * seconds_per_trip / 60 = 14 := by
  sorry

#check pool_filling_time

end NUMINAMATH_CALUDE_pool_filling_time_l893_89329


namespace NUMINAMATH_CALUDE_average_score_is_106_l893_89339

/-- The average bowling score of three bowlers -/
def average_bowling_score (score1 score2 score3 : ℕ) : ℚ :=
  (score1 + score2 + score3 : ℚ) / 3

/-- Theorem: The average bowling score of three bowlers with scores 120, 113, and 85 is 106 -/
theorem average_score_is_106 :
  average_bowling_score 120 113 85 = 106 := by
  sorry

end NUMINAMATH_CALUDE_average_score_is_106_l893_89339


namespace NUMINAMATH_CALUDE_min_value_trig_expression_min_value_trig_expression_achievable_l893_89310

theorem min_value_trig_expression (α β : ℝ) :
  (2 * Real.cos α + 5 * Real.sin β - 8)^2 + (2 * Real.sin α + 5 * Real.cos β - 15)^2 ≥ 100 :=
by sorry

theorem min_value_trig_expression_achievable :
  ∃ α β : ℝ, (2 * Real.cos α + 5 * Real.sin β - 8)^2 + (2 * Real.sin α + 5 * Real.cos β - 15)^2 = 100 :=
by sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_min_value_trig_expression_achievable_l893_89310


namespace NUMINAMATH_CALUDE_miae_closer_estimate_l893_89351

def bowl_volume : ℝ := 1000  -- in milliliters
def miae_estimate : ℝ := 1100  -- in milliliters
def hyori_estimate : ℝ := 850  -- in milliliters

theorem miae_closer_estimate :
  |miae_estimate - bowl_volume| < |hyori_estimate - bowl_volume| := by
  sorry

end NUMINAMATH_CALUDE_miae_closer_estimate_l893_89351


namespace NUMINAMATH_CALUDE_polynomial_factorization_l893_89302

theorem polynomial_factorization (a b c : ℝ) :
  a^4 * (b^3 - c^3) + b^4 * (c^3 - a^3) + c^4 * (a^3 - b^3) = 
  -(a - b) * (b - c) * (c - a) * (a^2 + a*b + b^2 + b*c + c^2 + a*c) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l893_89302


namespace NUMINAMATH_CALUDE_sheela_savings_percentage_l893_89356

/-- Given Sheela's deposit and monthly income, prove the percentage of income deposited -/
theorem sheela_savings_percentage (deposit : ℝ) (monthly_income : ℝ) 
  (h1 : deposit = 5000)
  (h2 : monthly_income = 25000) :
  (deposit / monthly_income) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_sheela_savings_percentage_l893_89356


namespace NUMINAMATH_CALUDE_dress_final_price_l893_89308

/-- The final price of a dress after multiple discounts and tax -/
def finalPrice (d : ℝ) : ℝ :=
  let price1 := d * (1 - 0.45)  -- After first discount
  let price2 := price1 * (1 - 0.30)  -- After second discount
  let price3 := price2 * (1 - 0.25)  -- After third discount
  let price4 := price3 * (1 - 0.50)  -- After staff discount
  price4 * (1 + 0.10)  -- After sales tax

/-- Theorem stating the final price of the dress -/
theorem dress_final_price (d : ℝ) : finalPrice d = 0.1588125 * d := by
  sorry

end NUMINAMATH_CALUDE_dress_final_price_l893_89308


namespace NUMINAMATH_CALUDE_mashas_dolls_l893_89301

theorem mashas_dolls (n : ℕ) : 
  (n / 2 : ℚ) * 1 + (n / 4 : ℚ) * 2 + (n / 4 : ℚ) * 4 = 24 → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_mashas_dolls_l893_89301


namespace NUMINAMATH_CALUDE_complement_union_M_N_l893_89323

def I : Set ℕ := {x | x ≤ 10}
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 4, 6, 8, 10}

theorem complement_union_M_N :
  (M ∪ N)ᶜ = {0, 5, 7, 9} := by sorry

end NUMINAMATH_CALUDE_complement_union_M_N_l893_89323


namespace NUMINAMATH_CALUDE_basketball_card_price_l893_89354

/-- The price of each pack of basketball cards given Nina's shopping details -/
theorem basketball_card_price (toy_price shirt_price total_spent : ℚ)
  (num_toys num_shirts num_card_packs : ℕ)
  (h1 : toy_price = 10)
  (h2 : shirt_price = 6)
  (h3 : num_toys = 3)
  (h4 : num_shirts = 5)
  (h5 : num_card_packs = 2)
  (h6 : total_spent = 70)
  (h7 : total_spent = toy_price * num_toys + shirt_price * num_shirts + num_card_packs * card_price) :
  card_price = 5 :=
by
  sorry

#check basketball_card_price

end NUMINAMATH_CALUDE_basketball_card_price_l893_89354


namespace NUMINAMATH_CALUDE_population_closest_to_target_in_2060_l893_89337

def initial_population : ℕ := 500
def growth_rate : ℕ := 4
def years_per_growth : ℕ := 30
def target_population : ℕ := 10000
def initial_year : ℕ := 2000

def population_at_year (year : ℕ) : ℕ :=
  initial_population * growth_rate ^ ((year - initial_year) / years_per_growth)

theorem population_closest_to_target_in_2060 :
  ∀ year : ℕ, year ≤ 2060 → population_at_year year ≤ target_population ∧
  population_at_year 2060 > population_at_year (2060 - years_per_growth) ∧
  population_at_year (2060 + years_per_growth) > target_population :=
by sorry

end NUMINAMATH_CALUDE_population_closest_to_target_in_2060_l893_89337


namespace NUMINAMATH_CALUDE_gym_member_count_l893_89361

/-- Represents a gym with its pricing and revenue information -/
structure Gym where
  charge_per_half_month : ℕ
  monthly_revenue : ℕ

/-- Calculates the number of members in the gym -/
def member_count (g : Gym) : ℕ :=
  g.monthly_revenue / (2 * g.charge_per_half_month)

/-- Theorem stating that a gym with the given parameters has 300 members -/
theorem gym_member_count :
  ∃ (g : Gym), g.charge_per_half_month = 18 ∧ g.monthly_revenue = 10800 ∧ member_count g = 300 := by
  sorry

end NUMINAMATH_CALUDE_gym_member_count_l893_89361


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l893_89382

def A : Set Nat := {1, 2, 3, 5}
def B : Set Nat := {2, 3, 6}

theorem union_of_A_and_B : A ∪ B = {1, 2, 3, 5, 6} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l893_89382


namespace NUMINAMATH_CALUDE_stone_123_is_8_l893_89388

/-- Represents the number of stones in the sequence -/
def num_stones : ℕ := 15

/-- Represents the length of a complete cycle in the counting pattern -/
def cycle_length : ℕ := 2 * num_stones - 1

/-- The count we're interested in -/
def target_count : ℕ := 123

/-- The original stone number we're trying to prove -/
def original_stone : ℕ := 8

/-- Theorem stating that the 123rd count corresponds to the 8th stone -/
theorem stone_123_is_8 : target_count % cycle_length = original_stone := by
  sorry

end NUMINAMATH_CALUDE_stone_123_is_8_l893_89388


namespace NUMINAMATH_CALUDE_angle_D_measure_l893_89398

-- Define the hexagon and its properties
def ConvexHexagon (A B C D E F : ℝ) : Prop :=
  -- Angles A, B, and C are congruent
  A = B ∧ B = C
  -- Angles D, E, and F are congruent
  ∧ D = E ∧ E = F
  -- The measure of angle A is 50° less than the measure of angle D
  ∧ A + 50 = D
  -- Sum of interior angles of a hexagon is 720°
  ∧ A + B + C + D + E + F = 720

-- Theorem statement
theorem angle_D_measure (A B C D E F : ℝ) 
  (h : ConvexHexagon A B C D E F) : D = 145 := by
  sorry

end NUMINAMATH_CALUDE_angle_D_measure_l893_89398


namespace NUMINAMATH_CALUDE_sector_max_area_l893_89374

/-- Given a sector with circumference 20cm, its maximum area is 25cm² -/
theorem sector_max_area :
  ∀ r l : ℝ,
  r > 0 →
  l > 0 →
  l + 2 * r = 20 →
  ∀ A : ℝ,
  A = 1/2 * l * r →
  A ≤ 25 :=
by
  sorry

end NUMINAMATH_CALUDE_sector_max_area_l893_89374


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l893_89378

/-- Proves that for any real number m, the line (2m+1)x + (m+1)y - 7m - 4 = 0 passes through the point (3, 1) -/
theorem fixed_point_on_line (m : ℝ) : (2 * m + 1) * 3 + (m + 1) * 1 - 7 * m - 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l893_89378
