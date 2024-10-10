import Mathlib

namespace church_members_count_l3553_355377

theorem church_members_count :
  ∀ (total adults children : ℕ),
  adults = (40 * total) / 100 →
  children = total - adults →
  children = adults + 24 →
  total = 120 :=
by
  sorry

end church_members_count_l3553_355377


namespace polar_equation_perpendicular_line_l3553_355352

/-- The polar equation of a line passing through (2,0) and perpendicular to the polar axis -/
theorem polar_equation_perpendicular_line (ρ θ : ℝ) :
  (∃ (x y : ℝ), x = 2 ∧ y = 0 ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) →
  (∀ (x y : ℝ), x = 2 → y = ρ * Real.sin θ) →
  ρ * Real.cos θ = 2 :=
by sorry

end polar_equation_perpendicular_line_l3553_355352


namespace polynomial_sum_l3553_355312

theorem polynomial_sum (h k : ℝ → ℝ) :
  (∀ x, h x + k x = -3 + 2 * x) →
  (∀ x, h x = x^3 - 3 * x^2 - 2) →
  (∀ x, k x = -x^3 + 3 * x^2 + 2 * x - 1) :=
by sorry

end polynomial_sum_l3553_355312


namespace unique_integer_solution_l3553_355306

theorem unique_integer_solution (x y z : ℤ) :
  x^2 + y^2 + z^2 = x^2 * y^2 → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end unique_integer_solution_l3553_355306


namespace expression_simplification_and_evaluation_l3553_355386

theorem expression_simplification_and_evaluation :
  ∀ a : ℕ,
    a ≠ 0 →
    a ≠ 1 →
    2 * a - 3 ≤ 1 →
    a = 2 →
    (a - (2 * a - 1) / a) / ((a^2 - 1) / a) = 1 / 3 :=
by
  sorry

end expression_simplification_and_evaluation_l3553_355386


namespace inscribed_hexagon_area_l3553_355313

/-- The area of a regular hexagon inscribed in a circle with area 100π square units -/
theorem inscribed_hexagon_area :
  let circle_area : ℝ := 100 * Real.pi
  let hexagon_area : ℝ := 150 * Real.sqrt 3
  (∃ (r : ℝ), r > 0 ∧ circle_area = Real.pi * r^2) →
  (∃ (s : ℝ), s > 0 ∧ hexagon_area = 6 * (s^2 * Real.sqrt 3 / 4)) →
  hexagon_area = 150 * Real.sqrt 3 :=
by sorry

end inscribed_hexagon_area_l3553_355313


namespace rectangle_max_area_l3553_355387

theorem rectangle_max_area (perimeter : ℝ) (width : ℝ) (length : ℝ) (area : ℝ) : 
  perimeter = 40 →
  length = 2 * width →
  perimeter = 2 * (length + width) →
  area = length * width →
  area = 800 / 9 := by
sorry

end rectangle_max_area_l3553_355387


namespace solve_for_r_l3553_355354

theorem solve_for_r (k : ℝ) (r : ℝ) 
  (h1 : 5 = k * 3^r) 
  (h2 : 45 = k * 9^r) : 
  r = 2 := by
sorry

end solve_for_r_l3553_355354


namespace three_pairs_satisfy_l3553_355335

/-- The set S of elements -/
inductive S
| A₀ : S
| A₁ : S
| A₂ : S

/-- The operation ⊕ on S -/
def op (x y : S) : S :=
  match x, y with
  | S.A₀, S.A₀ => S.A₀
  | S.A₀, S.A₁ => S.A₁
  | S.A₀, S.A₂ => S.A₂
  | S.A₁, S.A₀ => S.A₁
  | S.A₁, S.A₁ => S.A₂
  | S.A₁, S.A₂ => S.A₀
  | S.A₂, S.A₀ => S.A₂
  | S.A₂, S.A₁ => S.A₀
  | S.A₂, S.A₂ => S.A₁

/-- The theorem stating that there are exactly 3 pairs satisfying the equation -/
theorem three_pairs_satisfy :
  ∃! (pairs : List (S × S)), pairs.length = 3 ∧
    ∀ (x y : S), (op (op x y) x = S.A₀) ↔ (x, y) ∈ pairs :=
by sorry

end three_pairs_satisfy_l3553_355335


namespace yah_to_bah_conversion_l3553_355303

/-- Conversion rate between bahs and rahs -/
def bah_to_rah_rate : ℚ := 30 / 20

/-- Conversion rate between rahs and yahs -/
def rah_to_yah_rate : ℚ := 20 / 12

/-- The number of yahs we want to convert -/
def target_yahs : ℕ := 500

/-- The equivalent number of bahs -/
def equivalent_bahs : ℕ := 200

theorem yah_to_bah_conversion :
  (target_yahs : ℚ) / (rah_to_yah_rate * bah_to_rah_rate) = equivalent_bahs := by
  sorry

end yah_to_bah_conversion_l3553_355303


namespace second_number_is_30_l3553_355388

theorem second_number_is_30 (a b c : ℚ) 
  (sum_eq : a + b + c = 98)
  (ratio_ab : a / b = 2 / 3)
  (ratio_bc : b / c = 5 / 8) :
  b = 30 := by
  sorry

end second_number_is_30_l3553_355388


namespace optimal_bouquet_l3553_355337

/-- Represents the number of flowers Kyle picked last year -/
def last_year_flowers : ℕ := 12

/-- Represents the total number of flowers needed this year -/
def total_flowers : ℕ := 2 * last_year_flowers

/-- Represents the number of roses Kyle picked from his garden this year -/
def picked_roses : ℕ := last_year_flowers / 2

/-- Represents the cost of a rose -/
def rose_cost : ℕ := 3

/-- Represents the cost of a tulip -/
def tulip_cost : ℕ := 2

/-- Represents the cost of a daisy -/
def daisy_cost : ℕ := 1

/-- Represents Kyle's budget constraint -/
def budget : ℕ := 30

/-- Represents the number of additional flowers Kyle needs to buy -/
def flowers_to_buy : ℕ := total_flowers - picked_roses

theorem optimal_bouquet (roses tulips daisies : ℕ) :
  roses + tulips + daisies = flowers_to_buy →
  rose_cost * roses + tulip_cost * tulips + daisy_cost * daisies ≤ budget →
  roses ≤ 9 ∧
  (roses = 9 → tulips = 1 ∧ daisies = 1) :=
sorry

end optimal_bouquet_l3553_355337


namespace envelope_is_hyperbola_l3553_355366

/-- A family of straight lines forming right-angled triangles with area a^2 / 2 -/
def LineFamily (a : ℝ) := {l : Set (ℝ × ℝ) | ∃ α : ℝ, α > 0 ∧ l = {(x, y) | x + α^2 * y = α * a}}

/-- The envelope of the family of lines -/
def Envelope (a : ℝ) := {(x, y) : ℝ × ℝ | x * y = a^2 / 4}

/-- Theorem stating that the envelope of the line family is the hyperbola xy = a^2 / 4 -/
theorem envelope_is_hyperbola (a : ℝ) (h : a > 0) :
  Envelope a = {p : ℝ × ℝ | ∃ l ∈ LineFamily a, p ∈ l ∧ 
    ∀ l' ∈ LineFamily a, l ≠ l' → (∃ q ∈ l ∩ l', ∀ r ∈ l ∩ l', dist p q ≤ dist p r)} :=
sorry

end envelope_is_hyperbola_l3553_355366


namespace hyperbola_asymptotes_tangent_curve_l3553_355342

/-- The value of 'a' for which the asymptotes of the hyperbola x²/9 - y²/4 = 1 
    are precisely the two tangent lines of the curve y = ax² + 1/3 -/
theorem hyperbola_asymptotes_tangent_curve (a : ℝ) : 
  (∀ x y : ℝ, x^2/9 - y^2/4 = 1 → 
    ∃ k : ℝ, (y = k*x ∨ y = -k*x) ∧ 
    (∀ x₀ : ℝ, (k*x₀ = a*x₀^2 + 1/3 → 
      ∀ x : ℝ, k*x ≤ a*x^2 + 1/3) ∧
    (-k*x₀ = a*x₀^2 + 1/3 → 
      ∀ x : ℝ, -k*x ≤ a*x^2 + 1/3))) →
  a = 1/3 :=
sorry

end hyperbola_asymptotes_tangent_curve_l3553_355342


namespace heartsuit_calculation_l3553_355310

def heartsuit (u v : ℝ) : ℝ := (u + 2*v) * (u - v)

theorem heartsuit_calculation : heartsuit 2 (heartsuit 3 4) = -260 := by
  sorry

end heartsuit_calculation_l3553_355310


namespace min_reciprocal_sum_l3553_355392

theorem min_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 3 * b = 1) :
  (1 / a + 1 / b) ≥ 4 + 2 * Real.sqrt 3 :=
sorry

end min_reciprocal_sum_l3553_355392


namespace squirrel_stocks_l3553_355398

structure Squirrel where
  mushrooms : ℕ
  hazelnuts : ℕ
  fir_cones : ℕ

def total_items (s : Squirrel) : ℕ := s.mushrooms + s.hazelnuts + s.fir_cones

theorem squirrel_stocks :
  ∃ (zrzecka pizizubka krivoousko : Squirrel),
    -- Each squirrel has 48 mushrooms
    zrzecka.mushrooms = 48 ∧ pizizubka.mushrooms = 48 ∧ krivoousko.mushrooms = 48 ∧
    -- Zrzečka has twice as many hazelnuts as Pizizubka
    zrzecka.hazelnuts = 2 * pizizubka.hazelnuts ∧
    -- Křivoouško has 20 more hazelnuts than Pizizubka
    krivoousko.hazelnuts = pizizubka.hazelnuts + 20 ∧
    -- Together, they have 180 fir cones and 180 hazelnuts
    zrzecka.fir_cones + pizizubka.fir_cones + krivoousko.fir_cones = 180 ∧
    zrzecka.hazelnuts + pizizubka.hazelnuts + krivoousko.hazelnuts = 180 ∧
    -- All squirrels have the same total number of items
    total_items zrzecka = total_items pizizubka ∧
    total_items pizizubka = total_items krivoousko ∧
    -- The correct distribution of items
    zrzecka = { mushrooms := 48, hazelnuts := 80, fir_cones := 40 } ∧
    pizizubka = { mushrooms := 48, hazelnuts := 40, fir_cones := 80 } ∧
    krivoousko = { mushrooms := 48, hazelnuts := 60, fir_cones := 60 } :=
by
  sorry

end squirrel_stocks_l3553_355398


namespace distance_AK_equals_sqrt2_plus_1_l3553_355395

/-- Given a quadrilateral ABCD with vertices A(0, 0), B(0, -1), C(1, 0), D(√2/2, √2/2),
    and K is the intersection point of lines AB and CD,
    prove that the distance AK = √2 + 1 -/
theorem distance_AK_equals_sqrt2_plus_1 :
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (0, -1)
  let C : ℝ × ℝ := (1, 0)
  let D : ℝ × ℝ := (Real.sqrt 2 / 2, Real.sqrt 2 / 2)
  let K : ℝ × ℝ := (0, -(Real.sqrt 2 + 1))  -- Intersection point of AB and CD
  -- Distance formula
  let distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  distance A K = Real.sqrt 2 + 1 := by sorry

end distance_AK_equals_sqrt2_plus_1_l3553_355395


namespace emily_shopping_expense_l3553_355320

def total_spent (art_supplies_cost skirt_cost number_of_skirts : ℕ) : ℕ :=
  art_supplies_cost + skirt_cost * number_of_skirts

theorem emily_shopping_expense :
  let art_supplies_cost : ℕ := 20
  let skirt_cost : ℕ := 15
  let number_of_skirts : ℕ := 2
  total_spent art_supplies_cost skirt_cost number_of_skirts = 50 := by
  sorry

end emily_shopping_expense_l3553_355320


namespace seven_trapezoid_solutions_l3553_355382

/-- The number of solutions for the trapezoidal park problem -/
def trapezoid_solutions : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ =>
    p.1 % 5 = 0 ∧
    p.2 % 5 = 0 ∧
    75 * (p.1 + p.2) / 2 = 2250 ∧
    p.1 ≤ p.2)
    (Finset.product (Finset.range 61) (Finset.range 61))).card

/-- The theorem stating that there are exactly seven solutions -/
theorem seven_trapezoid_solutions : trapezoid_solutions = 7 := by
  sorry

end seven_trapezoid_solutions_l3553_355382


namespace melanie_dimes_problem_l3553_355349

/-- The number of dimes Melanie gave her mother -/
def dimes_given_to_mother (initial dimes_from_dad final : ℕ) : ℕ :=
  initial + dimes_from_dad - final

theorem melanie_dimes_problem (initial dimes_from_dad final : ℕ) 
  (h1 : initial = 7)
  (h2 : dimes_from_dad = 8)
  (h3 : final = 11) :
  dimes_given_to_mother initial dimes_from_dad final = 4 := by
sorry

end melanie_dimes_problem_l3553_355349


namespace ceiling_floor_product_l3553_355338

theorem ceiling_floor_product (y : ℝ) : 
  y < 0 → ⌈y⌉ * ⌊y⌋ = 72 → -9 < y ∧ y < -8 := by
  sorry

end ceiling_floor_product_l3553_355338


namespace solve_equation_l3553_355333

theorem solve_equation (x : ℝ) :
  3 * (x - 5) = 3 * (18 - 5) → x = 18 := by
  sorry

end solve_equation_l3553_355333


namespace odd_composite_quotient_l3553_355383

theorem odd_composite_quotient : 
  let first_four := [9, 15, 21, 25]
  let next_four := [27, 33, 35, 39]
  (first_four.prod : ℚ) / (next_four.prod : ℚ) = 25 / 429 := by
  sorry

end odd_composite_quotient_l3553_355383


namespace algebraic_expression_value_l3553_355317

theorem algebraic_expression_value (a b : ℝ) 
  (h1 : a - b = 5) 
  (h2 : a * b = 3) : 
  a * b^2 - a^2 * b = -15 := by
sorry

end algebraic_expression_value_l3553_355317


namespace last_remaining_number_l3553_355301

/-- Represents the state of the number sequence --/
structure SequenceState where
  numbers : List Nat
  markStart : Nat

/-- Marks every third number in the sequence --/
def markEveryThird (state : SequenceState) : SequenceState := sorry

/-- Reverses the remaining numbers in the sequence --/
def reverseRemaining (state : SequenceState) : SequenceState := sorry

/-- Performs one round of marking and reversing --/
def performRound (state : SequenceState) : SequenceState := sorry

/-- Continues the process until only one number remains --/
def processUntilOne (state : SequenceState) : Nat := sorry

/-- The main theorem to be proved --/
theorem last_remaining_number :
  processUntilOne { numbers := List.range 120, markStart := 1 } = 57 := by sorry

end last_remaining_number_l3553_355301


namespace rotation_equivalence_l3553_355308

theorem rotation_equivalence (y : ℝ) : 
  (450 % 360 : ℝ) = (360 - y) % 360 → y < 360 → y = 270 := by
  sorry

end rotation_equivalence_l3553_355308


namespace rectangle_width_proof_l3553_355375

theorem rectangle_width_proof (w : ℝ) (h1 : w > 0) : 
  (∃ l : ℝ, l > 0 ∧ l = 3 * w ∧ l + w = 3 * (l * w)) → w = 4/9 := by
  sorry

end rectangle_width_proof_l3553_355375


namespace ball_drawing_theorem_l3553_355350

/-- The number of balls in the bin -/
def n : ℕ := 15

/-- The number of draws -/
def k : ℕ := 4

/-- The number of possible lists when drawing k times with replacement from n balls -/
def possible_lists (n k : ℕ) : ℕ := n ^ k

theorem ball_drawing_theorem : possible_lists n k = 50625 := by
  sorry

end ball_drawing_theorem_l3553_355350


namespace at_least_one_correct_guess_l3553_355318

/-- Represents the color of a hat -/
inductive HatColor
| Red
| Blue
| Green

/-- Converts HatColor to its corresponding integer representation -/
def hatColorToInt (color : HatColor) : Fin 3 :=
  match color with
  | HatColor.Red => 0
  | HatColor.Blue => 1
  | HatColor.Green => 2

/-- Represents the configuration of hats on the four sages -/
structure HatConfiguration where
  a : HatColor
  b : HatColor
  c : HatColor
  d : HatColor

/-- Represents a sage's guess -/
def SageGuess := Fin 3

/-- The strategy for Sage A -/
def guessA (config : HatConfiguration) : SageGuess :=
  (hatColorToInt config.b + hatColorToInt config.d) % 3

/-- The strategy for Sage B -/
def guessB (config : HatConfiguration) : SageGuess :=
  (-(hatColorToInt config.a + hatColorToInt config.c)) % 3

/-- The strategy for Sage C -/
def guessC (config : HatConfiguration) : SageGuess :=
  (hatColorToInt config.b - hatColorToInt config.d) % 3

/-- The strategy for Sage D -/
def guessD (config : HatConfiguration) : SageGuess :=
  (hatColorToInt config.c - hatColorToInt config.a) % 3

/-- Theorem stating that the strategy guarantees at least one correct guess -/
theorem at_least_one_correct_guess (config : HatConfiguration) :
  (guessA config = hatColorToInt config.a) ∨
  (guessB config = hatColorToInt config.b) ∨
  (guessC config = hatColorToInt config.c) ∨
  (guessD config = hatColorToInt config.d) := by
  sorry

end at_least_one_correct_guess_l3553_355318


namespace train_speed_l3553_355315

theorem train_speed 
  (n : ℝ) 
  (a : ℝ) 
  (b : ℝ) 
  (c : ℝ) 
  (h1 : n > 0) 
  (h2 : a > c) 
  (h3 : b > 0) : 
  ∃ (speed : ℝ), speed = (b * (n + 1)) / (a - c) := by
  sorry

end train_speed_l3553_355315


namespace intersection_A_B_l3553_355328

-- Define set A
def A : Set ℝ := {y | ∃ x, y = Real.sin x}

-- Define set B
def B : Set ℝ := {x | x^2 - x < 0}

-- Theorem statement
theorem intersection_A_B : A ∩ B = Set.Ioo 0 1 := by sorry

end intersection_A_B_l3553_355328


namespace min_area_rectangle_l3553_355329

theorem min_area_rectangle (l w : ℕ) : 
  l > 0 ∧ w > 0 ∧ 2 * (l + w) = 84 → l * w ≥ 41 := by
  sorry

end min_area_rectangle_l3553_355329


namespace sum_of_fifth_powers_l3553_355322

theorem sum_of_fifth_powers (a b c : ℝ) 
  (h1 : a + b + c = 1)
  (h2 : a^2 + b^2 + c^2 = 3)
  (h3 : a^3 + b^3 + c^3 = 4)
  (h4 : a^4 + b^4 + c^4 = 5) :
  a^5 + b^5 + c^5 = 6 := by
sorry

end sum_of_fifth_powers_l3553_355322


namespace sin_45_eq_neg_cos_135_l3553_355336

theorem sin_45_eq_neg_cos_135 : Real.sin (π / 4) = - Real.cos (3 * π / 4) := by
  sorry

end sin_45_eq_neg_cos_135_l3553_355336


namespace total_prom_cost_is_correct_l3553_355302

/-- Calculates the total cost of prom services for Keesha -/
def total_prom_cost : ℝ :=
  let updo_cost : ℝ := 50
  let updo_discount : ℝ := 0.1
  let manicure_cost : ℝ := 30
  let pedicure_cost : ℝ := 35
  let pedicure_discount : ℝ := 0.5
  let makeup_cost : ℝ := 40
  let makeup_tax : ℝ := 0.07
  let facial_cost : ℝ := 60
  let facial_discount : ℝ := 0.15
  let tip_rate : ℝ := 0.2

  let hair_total : ℝ := (updo_cost * (1 - updo_discount)) * (1 + tip_rate)
  let nails_total : ℝ := (manicure_cost + pedicure_cost * pedicure_discount) * (1 + tip_rate)
  let makeup_total : ℝ := (makeup_cost * (1 + makeup_tax)) * (1 + tip_rate)
  let facial_total : ℝ := (facial_cost * (1 - facial_discount)) * (1 + tip_rate)

  hair_total + nails_total + makeup_total + facial_total

/-- Theorem stating that the total cost of prom services for Keesha is $223.56 -/
theorem total_prom_cost_is_correct : total_prom_cost = 223.56 := by
  sorry

end total_prom_cost_is_correct_l3553_355302


namespace ellipse_equation_l3553_355305

/-- Given an ellipse with foci on the x-axis, sum of major and minor axes equal to 10,
    and focal distance equal to 4√5, prove that its equation is x²/36 + y²/16 = 1. -/
theorem ellipse_equation (a b c : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : a + b = 10) (h4 : 2 * c = 4 * Real.sqrt 5) (h5 : a^2 - b^2 = c^2) :
  ∀ x y : ℝ, (x^2 / 36 + y^2 / 16 = 1) ↔ 
  (x^2 / a^2 + y^2 / b^2 = 1) :=
sorry

end ellipse_equation_l3553_355305


namespace three_in_range_of_f_l3553_355391

/-- The function f(x) = x^2 + bx - 1 -/
def f (b : ℝ) (x : ℝ) : ℝ := x^2 + b*x - 1

/-- Theorem: For all real b, 3 is in the range of f(x) = x^2 + bx - 1 -/
theorem three_in_range_of_f (b : ℝ) : ∃ x, f b x = 3 := by
  sorry

end three_in_range_of_f_l3553_355391


namespace shifted_parabola_equation_l3553_355343

/-- Represents a parabola in 2D space -/
structure Parabola where
  equation : ℝ → ℝ

/-- Shifts a parabola horizontally and vertically -/
def shift_parabola (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { equation := fun x => p.equation (x - h) + v }

/-- The original parabola y = 3x² -/
def original_parabola : Parabola :=
  { equation := fun x => 3 * x^2 }

/-- The shifted parabola -/
def shifted_parabola : Parabola :=
  shift_parabola original_parabola 1 2

theorem shifted_parabola_equation :
  shifted_parabola.equation = fun x => 3 * (x - 1)^2 + 2 := by
  sorry

end shifted_parabola_equation_l3553_355343


namespace fraction_simplification_l3553_355326

theorem fraction_simplification (x : ℝ) : 
  (x + 2) / 4 + (3 - 4*x) / 5 + (7*x - 1) / 10 = (3*x + 20) / 20 := by
  sorry

end fraction_simplification_l3553_355326


namespace carpet_shaded_area_l3553_355368

/-- The total shaded area of a carpet with a circle and squares -/
theorem carpet_shaded_area (carpet_side : ℝ) (circle_diameter : ℝ) (square_side : ℝ) : 
  carpet_side = 12 →
  carpet_side / circle_diameter = 4 →
  circle_diameter / square_side = 4 →
  (π * (circle_diameter / 2)^2) + (8 * square_side^2) = (9 * π / 4) + (9 / 2) :=
by sorry

end carpet_shaded_area_l3553_355368


namespace sin_cos_derivative_l3553_355323

theorem sin_cos_derivative (x : ℝ) : 
  deriv (λ x => Real.sin x * Real.cos x) x = Real.cos x ^ 2 - Real.sin x ^ 2 := by
  sorry

end sin_cos_derivative_l3553_355323


namespace matrix_sum_equality_l3553_355351

def A : Matrix (Fin 2) (Fin 2) ℤ := !![4, -3; 0, 5]
def B : Matrix (Fin 2) (Fin 2) ℤ := !![-6, 8; 7, -10]

theorem matrix_sum_equality : A + B = !![(-2 : ℤ), 5; 7, -5] := by sorry

end matrix_sum_equality_l3553_355351


namespace quadratic_inequality_solution_l3553_355325

open Set
open Real

-- Define the solution set
def solution_set : Set ℝ := {x : ℝ | 1 < x ∧ x < 3}

-- Define the quadratic function
def f (a c x : ℝ) := a * x^2 + x + c

-- Define the modified quadratic function
def g (a c x : ℝ) := a * x^2 + 2*x + 4*c

-- Define the linear function
def h (m x : ℝ) := x + m

theorem quadratic_inequality_solution (a c : ℝ) :
  (∀ x, x ∈ solution_set ↔ f a c x > 0) →
  (∀ x, g a c x > 0 → h m x > 0) →
  (∃ x, h m x > 0 ∧ g a c x ≤ 0) →
  (a = -1/4 ∧ c = -3/4) ∧ (∀ m', m' ≥ -2 ↔ m' ≥ m) :=
sorry

end quadratic_inequality_solution_l3553_355325


namespace two_face_painted_count_l3553_355339

/-- Represents a cube that has been painted on all faces and cut into smaller cubes --/
structure PaintedCube where
  /-- The number of smaller cubes along each edge of the original cube --/
  edge_count : Nat
  /-- Assumption that the cube is fully painted before cutting --/
  is_fully_painted : Bool

/-- Counts the number of smaller cubes painted on exactly two faces --/
def count_two_face_painted_cubes (cube : PaintedCube) : Nat :=
  sorry

/-- Theorem stating that a cube cut into 27 smaller cubes has 12 cubes painted on two faces --/
theorem two_face_painted_count (cube : PaintedCube) 
  (h1 : cube.edge_count = 3)
  (h2 : cube.is_fully_painted = true) : 
  count_two_face_painted_cubes cube = 12 :=
sorry

end two_face_painted_count_l3553_355339


namespace circle_tangency_l3553_355369

theorem circle_tangency (a : ℝ) : 
  (∃! p : ℝ × ℝ, p.1^2 + p.2^2 = 1 ∧ (p.1 + 4)^2 + (p.2 - a)^2 = 25) →
  (a = 2 * Real.sqrt 5 ∨ a = -2 * Real.sqrt 5 ∨ a = 0) :=
by sorry

end circle_tangency_l3553_355369


namespace distinct_arrangements_statistics_l3553_355380

def word_length : ℕ := 10
def letter_counts : List ℕ := [3, 2, 2, 1, 1]

theorem distinct_arrangements_statistics :
  (word_length.factorial) / ((letter_counts.map Nat.factorial).prod) = 75600 := by
  sorry

end distinct_arrangements_statistics_l3553_355380


namespace triangle_and_function_problem_l3553_355341

open Real

theorem triangle_and_function_problem 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (m α : ℝ) :
  (2 * c * cos B = 2 * a + b) →
  (∀ x, 2 * sin (2 * x + π / 6) + m * cos (2 * x) = 2 * sin (2 * (C / 2 - x) + π / 6) + m * cos (2 * (C / 2 - x))) →
  (2 * sin (α + π / 6) + m * cos α = 6 / 5) →
  cos (2 * α + C) = -7 / 25 := by
sorry

end triangle_and_function_problem_l3553_355341


namespace swimming_hours_per_month_l3553_355348

/-- Calculate the required hours per month for freestyle and sidestroke swimming --/
theorem swimming_hours_per_month 
  (total_required : ℕ) 
  (completed : ℕ) 
  (months : ℕ) 
  (h1 : total_required = 1500) 
  (h2 : completed = 180) 
  (h3 : months = 6) :
  (total_required - completed) / months = 220 :=
by sorry

end swimming_hours_per_month_l3553_355348


namespace mean_equality_implies_y_equals_three_l3553_355358

theorem mean_equality_implies_y_equals_three :
  let mean1 := (7 + 11 + 19) / 3
  let mean2 := (16 + 18 + y) / 3
  mean1 = mean2 →
  y = 3 := by
sorry

end mean_equality_implies_y_equals_three_l3553_355358


namespace purely_imaginary_complex_number_l3553_355384

theorem purely_imaginary_complex_number (m : ℝ) :
  let z : ℂ := m^2 - 1 + (m + 1) * Complex.I
  (z.re = 0 ∧ z.im ≠ 0) →
  (m = 1 ∧ (1 : ℂ) / (1 + z) = (1 : ℂ) / 5 - (2 : ℂ) / 5 * Complex.I) :=
by sorry

end purely_imaginary_complex_number_l3553_355384


namespace mutual_win_exists_l3553_355389

/-- Represents the result of a match between two teams -/
inductive MatchResult
| Win
| Draw
| Loss

/-- Calculates points for a given match result -/
def points (result : MatchResult) : Nat :=
  match result with
  | MatchResult.Win => 2
  | MatchResult.Draw => 1
  | MatchResult.Loss => 0

/-- Represents a tournament with given number of teams -/
structure Tournament (n : Nat) where
  firstRound : Fin n → Fin n → MatchResult
  secondRound : Fin n → Fin n → MatchResult

/-- Calculates total points for a team after both rounds -/
def totalPoints (t : Tournament n) (team : Fin n) : Nat :=
  sorry

/-- Checks if all teams have different points after the first round -/
def allDifferentFirstRound (t : Tournament n) : Prop :=
  sorry

/-- Checks if all teams have the same points after both rounds -/
def allSameTotal (t : Tournament n) : Prop :=
  sorry

/-- Checks if there exists a pair of teams that have each won once against each other -/
def existsMutualWin (t : Tournament n) : Prop :=
  sorry

/-- Main theorem: If all teams have different points after the first round
    and the same total points after both rounds, then there exists a pair
    of teams that have each won once against each other -/
theorem mutual_win_exists (t : Tournament 20)
    (h1 : allDifferentFirstRound t)
    (h2 : allSameTotal t) :
    existsMutualWin t := by
  sorry

end mutual_win_exists_l3553_355389


namespace f_fixed_point_exists_f_fixed_point_19_pow_86_l3553_355321

def f (A : ℕ) : ℕ :=
  let digits := Nat.digits 10 A
  List.sum (List.zipWith (·*·) (List.reverse digits) (List.map (2^·) (List.range digits.length)))

theorem f_fixed_point_exists (A : ℕ) : ∃ k : ℕ, f (f^[k] A) = f^[k] A :=
sorry

theorem f_fixed_point_19_pow_86 : ∃ k : ℕ, f^[k] (19^86) = 19 :=
sorry

end f_fixed_point_exists_f_fixed_point_19_pow_86_l3553_355321


namespace fraction_simplification_l3553_355370

theorem fraction_simplification (a b : ℝ) (h : b ≠ 0) :
  b / (a * b + b) = 1 / (a + 1) := by
  sorry

end fraction_simplification_l3553_355370


namespace fraction_equality_l3553_355394

theorem fraction_equality (m n p q : ℚ) 
  (h1 : m / n = 10)
  (h2 : p / n = 2)
  (h3 : p / q = 1 / 5) :
  m / q = 1 := by
  sorry

end fraction_equality_l3553_355394


namespace sufficient_condition_for_positive_quadratic_l3553_355359

theorem sufficient_condition_for_positive_quadratic (m : ℝ) :
  m > 1 → ∀ x : ℝ, x^2 - 2*x + m > 0 := by sorry

end sufficient_condition_for_positive_quadratic_l3553_355359


namespace congruence_problem_l3553_355347

theorem congruence_problem (n : ℤ) : 
  0 ≤ n ∧ n < 103 ∧ (100 * n) % 103 = 85 % 103 → n % 103 = 6 % 103 := by
  sorry

end congruence_problem_l3553_355347


namespace store_purchase_exists_l3553_355346

theorem store_purchase_exists :
  ∃ (P L E : ℕ), 0.45 * (P : ℝ) + 0.35 * (L : ℝ) + 0.30 * (E : ℝ) = 7.80 := by
  sorry

end store_purchase_exists_l3553_355346


namespace geometric_sequence_sum_l3553_355376

/-- Sum of the first n terms of a geometric sequence -/
def geometricSum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The problem statement -/
theorem geometric_sequence_sum :
  let a : ℚ := 1/6
  let r : ℚ := 1/2
  let n : ℕ := 6
  geometricSum a r n = 21/64 := by
  sorry

end geometric_sequence_sum_l3553_355376


namespace carolyn_unicorns_l3553_355330

/-- Calculates the number of unicorns Carolyn wants to embroider --/
def number_of_unicorns (stitches_per_minute : ℕ) (flower_stitches : ℕ) (unicorn_stitches : ℕ) 
  (godzilla_stitches : ℕ) (total_minutes : ℕ) (number_of_flowers : ℕ) : ℕ :=
  let total_stitches := stitches_per_minute * total_minutes
  let flower_total_stitches := flower_stitches * number_of_flowers
  let remaining_stitches := total_stitches - flower_total_stitches - godzilla_stitches
  remaining_stitches / unicorn_stitches

/-- Theorem stating that Carolyn wants to embroider 3 unicorns --/
theorem carolyn_unicorns : 
  number_of_unicorns 4 60 180 800 1085 50 = 3 := by
  sorry

end carolyn_unicorns_l3553_355330


namespace greatest_two_digit_with_digit_product_8_l3553_355362

/-- A function that returns the product of digits of a two-digit number -/
def digit_product (n : ℕ) : ℕ :=
  (n / 10) * (n % 10)

/-- A predicate that checks if a number is a two-digit number -/
def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

theorem greatest_two_digit_with_digit_product_8 :
  ∀ n : ℕ, is_two_digit n → digit_product n = 8 → n ≤ 81 :=
sorry

end greatest_two_digit_with_digit_product_8_l3553_355362


namespace manicure_cost_l3553_355332

/-- The cost of a manicure before tip, given the total amount paid and tip percentage. -/
theorem manicure_cost (total_paid : ℝ) (tip_percentage : ℝ) (cost : ℝ) : 
  total_paid = 39 → 
  tip_percentage = 0.30 → 
  cost * (1 + tip_percentage) = total_paid → 
  cost = 30 := by
  sorry

end manicure_cost_l3553_355332


namespace correct_email_sequence_l3553_355399

/-- Represents the steps in sending an email -/
inductive EmailStep
  | OpenEmailBox
  | EnterRecipientAddress
  | EnterSubject
  | EnterContent
  | ClickCompose
  | ClickSend

/-- Represents a sequence of email steps -/
def EmailSequence := List EmailStep

/-- The correct sequence of steps for sending an email -/
def correctEmailSequence : EmailSequence :=
  [EmailStep.OpenEmailBox, EmailStep.ClickCompose, EmailStep.EnterRecipientAddress,
   EmailStep.EnterSubject, EmailStep.EnterContent, EmailStep.ClickSend]

/-- Theorem stating that the given sequence is the correct one for sending an email -/
theorem correct_email_sequence :
  correctEmailSequence =
    [EmailStep.OpenEmailBox, EmailStep.ClickCompose, EmailStep.EnterRecipientAddress,
     EmailStep.EnterSubject, EmailStep.EnterContent, EmailStep.ClickSend] :=
by
  sorry

end correct_email_sequence_l3553_355399


namespace triangle_relation_angle_C_measure_l3553_355316

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  angle_sum : A + B + C = π

theorem triangle_relation (t : Triangle)
  (h : Real.sin (2 * t.A + t.B) / Real.sin t.A = 2 + 2 * Real.cos (t.A + t.B)) :
  t.b = 2 * t.a := by sorry

theorem angle_C_measure (t : Triangle)
  (h1 : t.b = 2 * t.a)
  (h2 : t.c = Real.sqrt 7 * t.a) :
  t.C = 2 * π / 3 := by sorry

end triangle_relation_angle_C_measure_l3553_355316


namespace min_value_reciprocal_sum_l3553_355345

theorem min_value_reciprocal_sum (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a + b + c = 3) :
  (1/a + 1/b + 1/c) ≥ 3 ∧ 
  ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 3 ∧ 1/x + 1/y + 1/z = 3 :=
by sorry

end min_value_reciprocal_sum_l3553_355345


namespace train_passing_time_l3553_355393

/-- The time taken for a train to pass a man moving in the same direction -/
theorem train_passing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) :
  train_length = 375 →
  train_speed = 72 * (1000 / 3600) →
  man_speed = 12 * (1000 / 3600) →
  (train_length / (train_speed - man_speed)) = 22.5 := by
  sorry

end train_passing_time_l3553_355393


namespace real_roots_of_equation_l3553_355344

theorem real_roots_of_equation :
  ∀ x : ℝ, x^4 + x^2 - 20 = 0 ↔ x = 2 ∨ x = -2 := by
  sorry

end real_roots_of_equation_l3553_355344


namespace spaghetti_cost_l3553_355334

def hamburger_cost : ℝ := 3
def fries_cost : ℝ := 1.20
def soda_cost : ℝ := 0.5
def num_hamburgers : ℕ := 5
def num_fries : ℕ := 4
def num_sodas : ℕ := 5
def num_friends : ℕ := 5
def individual_payment : ℝ := 5

theorem spaghetti_cost : 
  ∃ (spaghetti_price : ℝ),
    spaghetti_price = 
      num_friends * individual_payment - 
      (num_hamburgers * hamburger_cost + 
       num_fries * fries_cost + 
       num_sodas * soda_cost) ∧
    spaghetti_price = 2.70 :=
sorry

end spaghetti_cost_l3553_355334


namespace tile_difference_l3553_355300

/-- Given an initial figure with blue and red hexagonal tiles, and adding a border of red tiles,
    calculate the difference between the total number of red tiles and blue tiles in the new figure. -/
theorem tile_difference (initial_blue : ℕ) (initial_red : ℕ) (border_red : ℕ) : 
  initial_blue = 17 → initial_red = 8 → border_red = 24 →
  (initial_red + border_red) - initial_blue = 15 := by
sorry

end tile_difference_l3553_355300


namespace equation_solutions_l3553_355319

theorem equation_solutions : 
  {x : ℝ | (x^3 - 3*x^2)/(x^2 - 4*x + 4) + x = -3} = {-2, 3/2} :=
by sorry

end equation_solutions_l3553_355319


namespace polynomial_intersection_l3553_355314

def f (a b x : ℝ) : ℝ := x^2 + a*x + b
def g (c d x : ℝ) : ℝ := x^2 + c*x + d

theorem polynomial_intersection (a b c d : ℝ) :
  (∃ (x : ℝ), f a b x = g c d x ∧ x = 50 ∧ f a b x = -200) →
  (g c d (-a/2) = 0) →
  (f a b (-c/2) = 0) →
  (∃ (m : ℝ), (∀ (x : ℝ), f a b x ≥ m) ∧ (∀ (x : ℝ), g c d x ≥ m) ∧
               (∃ (x₁ x₂ : ℝ), f a b x₁ = m ∧ g c d x₂ = m)) →
  a + c = -200 :=
by sorry

end polynomial_intersection_l3553_355314


namespace cos_to_sin_shift_l3553_355309

open Real

theorem cos_to_sin_shift (x : ℝ) : 
  cos (2*x) = sin (2*(x - π/6)) :=
by sorry

end cos_to_sin_shift_l3553_355309


namespace total_bears_is_98_l3553_355311

/-- The maximum number of teddy bears that can be placed on each shelf. -/
def max_bears_per_shelf : ℕ := 7

/-- The number of filled shelves. -/
def filled_shelves : ℕ := 14

/-- The total number of teddy bears. -/
def total_bears : ℕ := max_bears_per_shelf * filled_shelves

/-- Theorem stating that the total number of teddy bears is 98. -/
theorem total_bears_is_98 : total_bears = 98 := by
  sorry

end total_bears_is_98_l3553_355311


namespace half_work_completed_l3553_355360

/-- Represents the highway construction project -/
structure HighwayProject where
  initialMen : ℕ
  totalLength : ℝ
  initialDays : ℕ
  initialHoursPerDay : ℕ
  actualDays : ℕ
  additionalMen : ℕ
  newHoursPerDay : ℕ

/-- Calculates the fraction of work completed -/
def fractionCompleted (project : HighwayProject) : ℚ :=
  let initialManHours := project.initialMen * project.initialDays * project.initialHoursPerDay
  let actualManHours := project.initialMen * project.actualDays * project.initialHoursPerDay
  actualManHours / initialManHours

/-- Theorem stating that the fraction of work completed is 1/2 -/
theorem half_work_completed (project : HighwayProject) 
  (h1 : project.initialMen = 100)
  (h2 : project.totalLength = 2)
  (h3 : project.initialDays = 50)
  (h4 : project.initialHoursPerDay = 8)
  (h5 : project.actualDays = 25)
  (h6 : project.additionalMen = 60)
  (h7 : project.newHoursPerDay = 10)
  (h8 : (project.initialMen + project.additionalMen) * (project.initialDays - project.actualDays) * project.newHoursPerDay = project.initialMen * project.initialDays * project.initialHoursPerDay / 2) :
  fractionCompleted project = 1/2 := by
  sorry


end half_work_completed_l3553_355360


namespace no_winning_strategy_for_tony_l3553_355367

/-- Represents a counter in the Ring Mafia game -/
inductive Counter
| Mafia
| Town

/-- Represents the state of the Ring Mafia game -/
structure GameState where
  counters : List Counter
  total_counters : Nat
  mafia_counters : Nat
  town_counters : Nat

/-- Represents a strategy for Tony -/
def TonyStrategy := GameState → List Nat

/-- Represents a strategy for Madeline -/
def MadelineStrategy := GameState → Nat

/-- Checks if a game state is valid according to the rules -/
def is_valid_game_state (state : GameState) : Prop :=
  state.total_counters = 2019 ∧
  state.mafia_counters = 673 ∧
  state.town_counters = 1346 ∧
  state.counters.length = state.total_counters

/-- Checks if the game has ended -/
def game_ended (state : GameState) : Prop :=
  state.mafia_counters = 0 ∨ state.town_counters = 0

/-- Theorem: There is no winning strategy for Tony in Ring Mafia -/
theorem no_winning_strategy_for_tony :
  ∀ (initial_state : GameState),
    is_valid_game_state initial_state →
    ∀ (tony_strategy : TonyStrategy),
    ∃ (madeline_strategy : MadelineStrategy),
    ∃ (final_state : GameState),
      game_ended final_state ∧
      final_state.town_counters = 0 :=
sorry

end no_winning_strategy_for_tony_l3553_355367


namespace least_non_lucky_multiple_of_7_l3553_355374

def sumOfDigits (n : ℕ) : ℕ := sorry

def isLucky (n : ℕ) : Prop := n > 0 ∧ n % sumOfDigits n = 0

def isMultipleOf7 (n : ℕ) : Prop := n % 7 = 0

theorem least_non_lucky_multiple_of_7 : 
  (∀ k : ℕ, k > 0 ∧ k < 14 ∧ isMultipleOf7 k → isLucky k) ∧ 
  isMultipleOf7 14 ∧ 
  ¬isLucky 14 := by sorry

end least_non_lucky_multiple_of_7_l3553_355374


namespace equilateral_triangle_tiling_l3553_355364

theorem equilateral_triangle_tiling (large_side : ℝ) (small_side : ℝ) : 
  large_side = 15 →
  small_side = 3 →
  (large_side^2 / small_side^2 : ℝ) = 25 := by
  sorry

end equilateral_triangle_tiling_l3553_355364


namespace arithmetic_sequence_problem_l3553_355340

theorem arithmetic_sequence_problem (x : ℚ) (n : ℕ) : 
  let a₁ := 3 * x - 2
  let a₂ := 7 * x - 15
  let a₃ := 4 * x + 3
  let d := a₂ - a₁
  let aₙ := a₁ + (n - 1) * d
  (a₂ - a₁ = a₃ - a₂) ∧ (aₙ = 4020) → n = 851 := by
sorry

end arithmetic_sequence_problem_l3553_355340


namespace quadratic_domain_range_implies_power_l3553_355396

/-- A quadratic function f(x) = x^2 - 4x + 4 + m with domain and range [2, n] -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 4*x + 4 + m

/-- The theorem stating that if f has domain and range [2, n], then m^n = 8 -/
theorem quadratic_domain_range_implies_power (m n : ℝ) :
  (∀ x, x ∈ Set.Icc 2 n ↔ f m x ∈ Set.Icc 2 n) →
  m^n = 8 := by
  sorry


end quadratic_domain_range_implies_power_l3553_355396


namespace total_distance_walked_l3553_355373

/-- Given a pace of 2 miles per hour maintained for 8 hours, 
    the total distance walked is 16 miles. -/
theorem total_distance_walked 
  (pace : ℝ) 
  (duration : ℝ) 
  (h1 : pace = 2) 
  (h2 : duration = 8) : 
  pace * duration = 16 := by
sorry

end total_distance_walked_l3553_355373


namespace circle_symmetry_l3553_355356

-- Define the symmetry property
def symmetric_point (x y : ℝ) : ℝ × ℝ := (y + 1, x - 1)

-- Define the equation of circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 6*x - 2*y = 0

-- Define the equation of circle C'
def circle_C' (x y : ℝ) : Prop := (x - 2)^2 + (y - 2)^2 = 10

-- Theorem statement
theorem circle_symmetry :
  (∀ x y : ℝ, circle_C x y → circle_C (symmetric_point x y).1 (symmetric_point x y).2) →
  (∀ x y : ℝ, circle_C' x y ↔ circle_C (symmetric_point x y).1 (symmetric_point x y).2) :=
by sorry

end circle_symmetry_l3553_355356


namespace add_9999_seconds_to_1645_l3553_355381

/-- Represents time in 24-hour format -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat
  deriving Repr

/-- Adds seconds to a given time -/
def addSeconds (t : Time) (s : Nat) : Time :=
  sorry

theorem add_9999_seconds_to_1645 :
  let initial_time : Time := ⟨16, 45, 0⟩
  let seconds_to_add : Nat := 9999
  let final_time : Time := addSeconds initial_time seconds_to_add
  final_time = ⟨19, 31, 39⟩ := by sorry

end add_9999_seconds_to_1645_l3553_355381


namespace hexagon_height_is_six_l3553_355361

/-- Represents a rectangle with width and height --/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a hexagon --/
structure Hexagon where
  height : ℝ

/-- Given a 9x16 rectangle that can be cut into two congruent hexagons
    and repositioned to form a different rectangle, 
    prove that the height of each hexagon is 6 --/
theorem hexagon_height_is_six 
  (original : Rectangle)
  (new : Rectangle)
  (hex1 hex2 : Hexagon)
  (h1 : original.width = 16 ∧ original.height = 9)
  (h2 : hex1 = hex2)
  (h3 : original.width * original.height = new.width * new.height)
  (h4 : new.width = new.height)
  (h5 : hex1.height + hex2.height = new.height)
  : hex1.height = 6 := by
  sorry

end hexagon_height_is_six_l3553_355361


namespace factor_expression_l3553_355371

theorem factor_expression (x : ℝ) : 75 * x^11 + 200 * x^22 = 25 * x^11 * (3 + 8 * x^11) := by
  sorry

end factor_expression_l3553_355371


namespace building_stories_l3553_355324

theorem building_stories (apartments_per_floor : ℕ) (people_per_apartment : ℕ) (total_people : ℕ) :
  apartments_per_floor = 4 →
  people_per_apartment = 2 →
  total_people = 200 →
  total_people / (apartments_per_floor * people_per_apartment) = 25 :=
by sorry

end building_stories_l3553_355324


namespace square_sum_value_l3553_355327

theorem square_sum_value (x y : ℝ) (h1 : (x + y)^2 = 16) (h2 : x * y = -9) :
  x^2 + y^2 = 34 := by
  sorry

end square_sum_value_l3553_355327


namespace grapes_in_robs_bowl_l3553_355363

theorem grapes_in_robs_bowl (rob_grapes : ℕ) 
  (allie_grapes : ℕ) (allyn_grapes : ℕ) : 
  (allie_grapes = rob_grapes + 2) → 
  (allyn_grapes = allie_grapes + 4) → 
  (rob_grapes + allie_grapes + allyn_grapes = 83) → 
  rob_grapes = 25 := by
sorry

end grapes_in_robs_bowl_l3553_355363


namespace incorrect_equation_l3553_355365

theorem incorrect_equation (a b : ℤ) : 
  (-a + b = -1) → (a + b = 5) → (4*a + b = 14) → (2*a + b ≠ 7) :=
by
  sorry

end incorrect_equation_l3553_355365


namespace lcm_gcf_problem_l3553_355355

theorem lcm_gcf_problem (n : ℕ) : 
  Nat.lcm n 12 = 54 → Nat.gcd n 12 = 8 → n = 36 := by
  sorry

end lcm_gcf_problem_l3553_355355


namespace gcd_7854_13843_l3553_355331

theorem gcd_7854_13843 : Nat.gcd 7854 13843 = 1 := by
  sorry

end gcd_7854_13843_l3553_355331


namespace divisible_by_eleven_smallest_n_seven_l3553_355390

theorem divisible_by_eleven_smallest_n_seven (x : ℕ) : 
  (∃ k : ℕ, x = 11 * k) ∧ 
  (∀ m : ℕ, m < 7 → ¬(∃ j : ℕ, m * 11 = x)) ∧
  (∃ i : ℕ, 7 * 11 = x) →
  x = 77 := by sorry

end divisible_by_eleven_smallest_n_seven_l3553_355390


namespace A_in_third_quadrant_l3553_355379

/-- A point in a 2D coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the third quadrant -/
def isInThirdQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- The given point A -/
def A : Point :=
  { x := -2, y := -3 }

/-- Theorem stating that point A is in the third quadrant -/
theorem A_in_third_quadrant : isInThirdQuadrant A := by
  sorry

end A_in_third_quadrant_l3553_355379


namespace circle_equation_shortest_chord_line_l3553_355307

-- Define the circle
def circle_center : ℝ × ℝ := (1, -2)
def tangent_line (x y : ℝ) : Prop := x + y - 1 = 0

-- Define point P
def point_P : ℝ × ℝ := (2, -2.5)

-- Function to check if a point is inside the circle
def is_inside_circle (p : ℝ × ℝ) : Prop := sorry

-- Theorem for the standard equation of the circle
theorem circle_equation (x y : ℝ) : 
  is_inside_circle point_P →
  ((x - circle_center.1)^2 + (y - circle_center.2)^2 = 2) ↔ 
  (∃ (p : ℝ × ℝ), p.1 = x ∧ p.2 = y ∧ (is_inside_circle p ∨ tangent_line p.1 p.2)) :=
sorry

-- Theorem for the equation of the line containing the shortest chord
theorem shortest_chord_line (x y : ℝ) :
  is_inside_circle point_P →
  (4*x - 2*y - 13 = 0) ↔ 
  (∃ (p q : ℝ × ℝ), 
    p ≠ q ∧ 
    is_inside_circle p ∧ 
    is_inside_circle q ∧ 
    p.1 = x ∧ p.2 = y ∧
    (q.1 - point_P.1) * (p.2 - point_P.2) = (q.2 - point_P.2) * (p.1 - point_P.1) ∧
    ∀ (r s : ℝ × ℝ), 
      r ≠ s → 
      is_inside_circle r → 
      is_inside_circle s → 
      (r.1 - point_P.1) * (s.2 - point_P.2) = (r.2 - point_P.2) * (s.1 - point_P.1) →
      (p.1 - q.1)^2 + (p.2 - q.2)^2 ≤ (r.1 - s.1)^2 + (r.2 - s.2)^2) :=
sorry

end circle_equation_shortest_chord_line_l3553_355307


namespace quadratic_distinct_roots_l3553_355357

theorem quadratic_distinct_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 6*x + m = 0 ∧ y^2 - 6*y + m = 0) → m < 9 :=
by sorry

end quadratic_distinct_roots_l3553_355357


namespace problem_statement_l3553_355378

def M : Set ℝ := {x | 0 < x ∧ x ≤ 3}
def N : Set ℝ := {x | 0 < x ∧ x ≤ 2}

theorem problem_statement :
  (∀ a, a ∈ M → a ∈ N) ∧ 
  (∃ a, a ∈ M ∧ a ∉ N) ∧
  (∀ a b, (a ∈ M → b ∉ M) ↔ (b ∈ M → a ∉ M)) ∧
  (¬(∃ x : ℝ, x^2 - x - 1 > 0) ↔ (∀ x : ℝ, x^2 - x - 1 ≤ 0)) :=
by sorry

end problem_statement_l3553_355378


namespace mode_is_131_l3553_355304

/- Define the structure of a stem-and-leaf plot entry -/
structure StemLeafEntry :=
  (stem : ℕ)
  (leaves : List ℕ)

/- Define the stem-and-leaf plot -/
def stemLeafPlot : List StemLeafEntry := [
  ⟨9, [5, 5, 6]⟩,
  ⟨10, [4, 8]⟩,
  ⟨11, [2, 2, 2, 6, 6, 7]⟩,
  ⟨12, [0, 0, 3, 7, 7, 7]⟩,
  ⟨13, [1, 1, 1, 1]⟩,
  ⟨14, [5, 9]⟩
]

/- Define a function to calculate the mode -/
def calculateMode (plot : List StemLeafEntry) : ℕ :=
  sorry

/- Theorem stating that the mode of the given stem-and-leaf plot is 131 -/
theorem mode_is_131 : calculateMode stemLeafPlot = 131 :=
  sorry

end mode_is_131_l3553_355304


namespace untouched_shapes_after_game_l3553_355372

-- Define the game state
structure GameState where
  triangles : Nat
  squares : Nat
  pentagons : Nat
  untouchedShapes : Nat
  turn : Nat

-- Define the initial game state
def initialState : GameState :=
  { triangles := 3
  , squares := 4
  , pentagons := 5
  , untouchedShapes := 12
  , turn := 0
  }

-- Define a move function for Petya
def petyaMove (state : GameState) : GameState :=
  { state with
    untouchedShapes := state.untouchedShapes - (if state.turn = 0 then 1 else 0)
    turn := state.turn + 1
  }

-- Define a move function for Vasya
def vasyaMove (state : GameState) : GameState :=
  { state with
    untouchedShapes := state.untouchedShapes - 1
    turn := state.turn + 1
  }

-- Define the final state after 10 turns
def finalState : GameState :=
  (List.range 5).foldl (fun state _ => vasyaMove (petyaMove state)) initialState

-- Theorem statement
theorem untouched_shapes_after_game :
  finalState.untouchedShapes = 6 := by sorry

end untouched_shapes_after_game_l3553_355372


namespace geometric_sequence_ratio_l3553_355385

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  geometric_sequence a q →
  q ≠ 1 →
  (∀ n, a n > 0) →
  (a 3 + a 6 = 2 * a 5) →
  (a 3 + a 4) / (a 4 + a 5) = (-1 + Real.sqrt 5) / 2 :=
by
  sorry

end geometric_sequence_ratio_l3553_355385


namespace gamma_cheaper_at_11_gamma_not_cheaper_at_10_min_shirts_for_gamma_cheaper_l3553_355397

/-- Represents the cost function for a t-shirt company -/
structure TShirtCompany where
  setupFee : ℕ
  costPerShirt : ℕ

/-- Calculates the total cost for a given number of shirts -/
def totalCost (company : TShirtCompany) (shirts : ℕ) : ℕ :=
  company.setupFee + company.costPerShirt * shirts

/-- The Acme T-Shirt Company -/
def acme : TShirtCompany := ⟨40, 10⟩

/-- The Beta T-Shirt Company -/
def beta : TShirtCompany := ⟨0, 15⟩

/-- The Gamma T-Shirt Company -/
def gamma : TShirtCompany := ⟨20, 12⟩

theorem gamma_cheaper_at_11 :
  totalCost gamma 11 < totalCost acme 11 ∧
  totalCost gamma 11 < totalCost beta 11 :=
sorry

theorem gamma_not_cheaper_at_10 :
  ¬(totalCost gamma 10 < totalCost acme 10 ∧
    totalCost gamma 10 < totalCost beta 10) :=
sorry

theorem min_shirts_for_gamma_cheaper : ℕ :=
  11

end gamma_cheaper_at_11_gamma_not_cheaper_at_10_min_shirts_for_gamma_cheaper_l3553_355397


namespace hospital_workers_count_l3553_355353

theorem hospital_workers_count :
  let total_workers : ℕ := 2 + 3  -- Jack, Jill, and 3 others
  let interview_size : ℕ := 2
  let prob_jack_and_jill : ℚ := 1 / 10  -- 0.1 as a rational number
  total_workers = 5 ∧
  interview_size = 2 ∧
  prob_jack_and_jill = 1 / Nat.choose total_workers interview_size :=
by sorry

end hospital_workers_count_l3553_355353
