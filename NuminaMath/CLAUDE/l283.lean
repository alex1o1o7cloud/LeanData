import Mathlib

namespace parabola_vertex_l283_28385

/-- A parabola with vertex (h, k) has the general form y = (x - h)² + k -/
def is_parabola_with_vertex (f : ℝ → ℝ) (h k : ℝ) : Prop :=
  ∀ x, f x = (x - h)^2 + k

/-- The specific parabola we're considering -/
def f (x : ℝ) : ℝ := (x - 4)^2 - 3

/-- Theorem stating that f is a parabola with vertex (4, -3) -/
theorem parabola_vertex : is_parabola_with_vertex f 4 (-3) := by
  sorry

end parabola_vertex_l283_28385


namespace adam_caramel_boxes_l283_28373

/-- The number of boxes of caramel candy Adam bought -/
def caramel_boxes (chocolate_boxes : ℕ) (pieces_per_box : ℕ) (total_candies : ℕ) : ℕ :=
  (total_candies - chocolate_boxes * pieces_per_box) / pieces_per_box

/-- Proof that Adam bought 5 boxes of caramel candy -/
theorem adam_caramel_boxes : 
  caramel_boxes 2 4 28 = 5 := by
  sorry

end adam_caramel_boxes_l283_28373


namespace infinitely_many_terms_greater_than_position_l283_28389

/-- A sequence of natural numbers excluding 1 -/
def NatSequenceExcluding1 := ℕ → {n : ℕ // n ≠ 1}

/-- The proposition that for any sequence of natural numbers excluding 1,
    there are infinitely many terms greater than their positions -/
theorem infinitely_many_terms_greater_than_position
  (seq : NatSequenceExcluding1) :
  ∀ N : ℕ, ∃ n > N, (seq n).val > n := by
  sorry

end infinitely_many_terms_greater_than_position_l283_28389


namespace percentage_calculation_l283_28381

theorem percentage_calculation (N : ℚ) (h : (1/2) * N = 16) : (3/4) * N = 24 := by
  sorry

end percentage_calculation_l283_28381


namespace fair_coin_tosses_l283_28367

theorem fair_coin_tosses (n : ℕ) : 
  (1 / 2 : ℝ) ^ n = (1 / 16 : ℝ) → n = 4 := by
  sorry

end fair_coin_tosses_l283_28367


namespace triangle_right_angle_l283_28386

theorem triangle_right_angle (A B C : ℝ) (h : A = B - C) : B = 90 := by
  sorry

end triangle_right_angle_l283_28386


namespace median_name_length_and_syllables_l283_28327

theorem median_name_length_and_syllables :
  let total_names : ℕ := 23
  let names_4_1 : ℕ := 8  -- 8 names of length 4 and 1 syllable
  let names_5_2 : ℕ := 5  -- 5 names of length 5 and 2 syllables
  let names_3_1 : ℕ := 3  -- 3 names of length 3 and 1 syllable
  let names_6_2 : ℕ := 4  -- 4 names of length 6 and 2 syllables
  let names_7_3 : ℕ := 3  -- 3 names of length 7 and 3 syllables
  
  let median_position : ℕ := (total_names + 1) / 2
  
  let lengths : List ℕ := [3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7]
  let syllables : List ℕ := [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3]
  
  (median_position = 12) ∧
  (lengths.get! (median_position - 1) = 5) ∧
  (syllables.get! (median_position - 1) = 1) :=
by sorry

end median_name_length_and_syllables_l283_28327


namespace problem_statement_l283_28387

theorem problem_statement : 2 * Real.sin (π / 3) + (-1/2)⁻¹ + |2 - Real.sqrt 3| = 0 := by
  sorry

end problem_statement_l283_28387


namespace unique_solution_equation_l283_28366

theorem unique_solution_equation (x : ℝ) :
  x ≥ 0 ∧ (2021 * (x^2020)^(1/202) - 1 = 2020 * x) ↔ x = 1 := by
  sorry

end unique_solution_equation_l283_28366


namespace pizza_sharing_l283_28326

theorem pizza_sharing (total pizza_jovin pizza_anna pizza_olivia : ℚ) : 
  total = 1 →
  pizza_jovin = 1/3 →
  pizza_anna = 1/6 →
  pizza_olivia = 1/4 →
  total - (pizza_jovin + pizza_anna + pizza_olivia) = 1/4 := by
  sorry

end pizza_sharing_l283_28326


namespace max_sphere_in_intersecting_cones_l283_28353

/-- Represents a right circular cone -/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- Represents the configuration of two intersecting cones -/
structure IntersectingCones where
  cone1 : Cone
  cone2 : Cone
  intersectionDistance : ℝ

/-- The maximum squared radius of a sphere that can fit inside two intersecting cones -/
def maxSphereRadiusSquared (ic : IntersectingCones) : ℝ := sorry

/-- The specific configuration of cones described in the problem -/
def problemCones : IntersectingCones :=
  { cone1 := { baseRadius := 3, height := 8 },
    cone2 := { baseRadius := 3, height := 8 },
    intersectionDistance := 3 }

theorem max_sphere_in_intersecting_cones :
  maxSphereRadiusSquared problemCones = 225 / 73 := by sorry

end max_sphere_in_intersecting_cones_l283_28353


namespace water_poured_out_l283_28359

-- Define the initial and final amounts of water
def initial_amount : ℝ := 0.8
def final_amount : ℝ := 0.6

-- Define the amount of water poured out
def poured_out : ℝ := initial_amount - final_amount

-- Theorem to prove
theorem water_poured_out : poured_out = 0.2 := by
  sorry

end water_poured_out_l283_28359


namespace rightmost_three_digits_of_6_to_1993_l283_28360

theorem rightmost_three_digits_of_6_to_1993 :
  6^1993 ≡ 296 [ZMOD 1000] := by
  sorry

end rightmost_three_digits_of_6_to_1993_l283_28360


namespace james_total_toys_l283_28321

/-- The number of toy cars James buys -/
def toy_cars : ℕ := 20

/-- The number of toy soldiers James buys -/
def toy_soldiers : ℕ := 2 * toy_cars

/-- The total number of toys James buys -/
def total_toys : ℕ := toy_cars + toy_soldiers

/-- Theorem stating that the total number of toys James buys is 60 -/
theorem james_total_toys : total_toys = 60 := by
  sorry

end james_total_toys_l283_28321


namespace area_of_right_triangle_l283_28379

-- Define the right triangle ABC
structure RightTriangle where
  AB : ℝ
  BD : ℝ
  isRightTriangle : AB > 0 ∧ BD > 0

-- Define the theorem
theorem area_of_right_triangle (t : RightTriangle) 
  (h1 : t.AB = 13) 
  (h2 : t.BD = 12) : 
  (1 / 2 : ℝ) * t.AB * t.BD = 202.8 := by
  sorry


end area_of_right_triangle_l283_28379


namespace special_polyhedron_property_l283_28357

-- Define the structure of our polyhedron
structure Polyhedron where
  V : ℕ  -- number of vertices
  E : ℕ  -- number of edges
  F : ℕ  -- number of faces
  T : ℕ  -- number of triangular faces
  P : ℕ  -- number of square faces

-- Define the properties of our specific polyhedron
def SpecialPolyhedron (poly : Polyhedron) : Prop :=
  poly.V - poly.E + poly.F = 2 ∧  -- Euler's formula
  poly.F = 40 ∧                   -- Total number of faces
  poly.T + poly.P = poly.F ∧      -- Faces are either triangles or squares
  poly.T = 1 ∧                    -- Number of triangular faces at a vertex
  poly.P = 3 ∧                    -- Number of square faces at a vertex
  poly.E = (3 * poly.T + 4 * poly.P) / 2  -- Edge calculation

-- Theorem statement
theorem special_polyhedron_property (poly : Polyhedron) 
  (h : SpecialPolyhedron poly) : 
  100 * poly.P + 10 * poly.T + poly.V = 351 := by
  sorry

end special_polyhedron_property_l283_28357


namespace sqrt_x_div_sqrt_y_l283_28374

theorem sqrt_x_div_sqrt_y (x y : ℝ) (h : (1/3)^2 + (1/4)^2 = ((37 * x) / (73 * y)) * ((1/5)^2 + (1/6)^2)) :
  Real.sqrt x / Real.sqrt y = 1281 / 94 := by
  sorry

end sqrt_x_div_sqrt_y_l283_28374


namespace probability_prime_or_square_l283_28375

/-- A function that returns true if a number is prime --/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that returns true if a number is a perfect square --/
def isPerfectSquare (n : ℕ) : Prop := sorry

/-- The number of sides on each die --/
def numSides : ℕ := 8

/-- The set of possible outcomes when rolling two dice --/
def outcomes : Finset (ℕ × ℕ) := sorry

/-- The set of favorable outcomes (sum is prime or perfect square) --/
def favorableOutcomes : Finset (ℕ × ℕ) := sorry

/-- Theorem stating the probability of getting a sum that is either prime or a perfect square --/
theorem probability_prime_or_square :
  (Finset.card favorableOutcomes : ℚ) / (Finset.card outcomes : ℚ) = 35 / 64 := by sorry

end probability_prime_or_square_l283_28375


namespace sons_age_l283_28317

theorem sons_age (son_age father_age : ℕ) : 
  father_age = son_age + 20 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 18 := by
sorry

end sons_age_l283_28317


namespace inequality_holds_iff_theta_in_range_l283_28378

theorem inequality_holds_iff_theta_in_range :
  ∀ k : ℤ, ∀ θ : ℝ,
    (2 * k * Real.pi + Real.pi / 12 < θ ∧ θ < 2 * k * Real.pi + 5 * Real.pi / 12) ↔
    (∀ x : ℝ, x ∈ Set.Icc 0 1 →
      x^2 * Real.cos θ - x * (1 - x) + (1 - x)^2 * Real.sin θ > 0) :=
by sorry

end inequality_holds_iff_theta_in_range_l283_28378


namespace constant_term_expansion_l283_28340

theorem constant_term_expansion (b : ℝ) (h : b = -1/2) :
  let c := 6 * b^2
  c = 3/2 := by sorry

end constant_term_expansion_l283_28340


namespace min_value_sin_product_l283_28347

theorem min_value_sin_product (x₁ x₂ x₃ x₄ : ℝ) 
  (h_positive : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0) 
  (h_sum : x₁ + x₂ + x₃ + x₄ = Real.pi) : 
  (2 * Real.sin x₁ ^ 2 + 1 / Real.sin x₁ ^ 2) *
  (2 * Real.sin x₂ ^ 2 + 1 / Real.sin x₂ ^ 2) *
  (2 * Real.sin x₃ ^ 2 + 1 / Real.sin x₃ ^ 2) *
  (2 * Real.sin x₄ ^ 2 + 1 / Real.sin x₄ ^ 2) ≥ 81 :=
by sorry

#check min_value_sin_product

end min_value_sin_product_l283_28347


namespace cubic_roots_sum_of_squares_l283_28323

theorem cubic_roots_sum_of_squares (a b c t : ℝ) : 
  (∀ x, x^3 - 8*x^2 + 14*x - 2 = 0 ↔ (x = a ∨ x = b ∨ x = c)) →
  t = Real.sqrt a + Real.sqrt b + Real.sqrt c →
  t^4 - 16*t^2 - 12*t = -8*Real.sqrt 2 / 3 := by
sorry

end cubic_roots_sum_of_squares_l283_28323


namespace function_max_min_condition_l283_28363

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 3*a*x^2 + 3*(a + 2)*x + 1

-- State the theorem
theorem function_max_min_condition (a : ℝ) : 
  (∃ (max min : ℝ), ∀ x, f a x ≤ max ∧ f a x ≥ min) ↔ (a > 2 ∨ a < -1) := by
  sorry

end function_max_min_condition_l283_28363


namespace james_work_hours_l283_28308

def minimum_wage : ℚ := 8
def meat_pounds : ℕ := 20
def meat_price : ℚ := 5
def fruit_veg_pounds : ℕ := 15
def fruit_veg_price : ℚ := 4
def bread_pounds : ℕ := 60
def bread_price : ℚ := 3/2
def janitor_hours : ℕ := 10
def janitor_wage : ℚ := 10

def total_cost : ℚ := 
  meat_pounds * meat_price + 
  fruit_veg_pounds * fruit_veg_price + 
  bread_pounds * bread_price + 
  janitor_hours * (janitor_wage * 3/2)

theorem james_work_hours : 
  total_cost / minimum_wage = 50 := by sorry

end james_work_hours_l283_28308


namespace quadratic_equation_solution_l283_28364

theorem quadratic_equation_solution : 
  let x₁ : ℝ := (2 + Real.sqrt 2) / 2
  let x₂ : ℝ := (2 - Real.sqrt 2) / 2
  2 * x₁^2 = 4 * x₁ - 1 ∧ 2 * x₂^2 = 4 * x₂ - 1 :=
by sorry

end quadratic_equation_solution_l283_28364


namespace distribution_scheme_count_l283_28331

/-- The number of ways to distribute spots among schools -/
def distribute_spots (total_spots : ℕ) (num_schools : ℕ) (distribution : List ℕ) : ℕ :=
  if total_spots = distribution.sum ∧ num_schools = distribution.length
  then Nat.factorial num_schools
  else 0

theorem distribution_scheme_count :
  distribute_spots 10 4 [1, 2, 3, 4] = 24 := by
  sorry

end distribution_scheme_count_l283_28331


namespace cafeteria_red_apples_l283_28320

/-- The number of red apples ordered by the cafeteria -/
def red_apples : ℕ := sorry

/-- The number of green apples ordered by the cafeteria -/
def green_apples : ℕ := 17

/-- The number of students who took apples -/
def students_took_apples : ℕ := 10

/-- The number of extra apples left -/
def extra_apples : ℕ := 32

/-- The total number of apples ordered by the cafeteria -/
def total_apples : ℕ := red_apples + green_apples

theorem cafeteria_red_apples :
  red_apples = 25 :=
by sorry

end cafeteria_red_apples_l283_28320


namespace shanghai_score_is_75_l283_28355

/-- The score of the Shanghai team in the basketball game -/
def shanghai_score : ℕ := 75

/-- The score of the Beijing team in the basketball game -/
def beijing_score : ℕ := shanghai_score - 10

/-- Yao Ming's score in the basketball game -/
def yao_ming_score : ℕ := 30

theorem shanghai_score_is_75 :
  (shanghai_score - beijing_score = 10) ∧
  (shanghai_score + beijing_score = 5 * yao_ming_score - 10) →
  shanghai_score = 75 := by
sorry

end shanghai_score_is_75_l283_28355


namespace either_shooter_hits_probability_l283_28354

-- Define the probabilities for shooters A and B
def prob_A_hits : ℝ := 0.9
def prob_B_hits : ℝ := 0.8

-- Define the probability that either A or B hits the target
def prob_either_hits : ℝ := 1 - (1 - prob_A_hits) * (1 - prob_B_hits)

-- Theorem statement
theorem either_shooter_hits_probability :
  prob_either_hits = 0.98 := by
  sorry

end either_shooter_hits_probability_l283_28354


namespace linear_equation_solution_l283_28395

theorem linear_equation_solution (x y m : ℝ) : 
  x = 2 ∧ y = -3 ∧ 5 * x + m * y + 2 = 0 → m = 4 := by
  sorry

end linear_equation_solution_l283_28395


namespace furniture_sale_price_l283_28390

theorem furniture_sale_price (wholesale_price : ℝ) 
  (sticker_price : ℝ) (sale_price : ℝ) :
  sticker_price = wholesale_price * 1.4 →
  sale_price = sticker_price * 0.65 →
  sale_price = wholesale_price * 0.91 := by
sorry

end furniture_sale_price_l283_28390


namespace unique_triple_solution_l283_28316

theorem unique_triple_solution (a b c : ℝ) : 
  a > 2 ∧ b > 2 ∧ c > 2 ∧ 
  ((a + 3)^2) / (b + c - 3) + ((b + 5)^2) / (c + a - 5) + ((c + 7)^2) / (a + b - 7) = 45 →
  a = 13 ∧ b = 11 ∧ c = 6 := by
  sorry

end unique_triple_solution_l283_28316


namespace ab_minus_one_lt_a_minus_b_l283_28352

theorem ab_minus_one_lt_a_minus_b (a b : ℝ) (ha : a > 0) (hb : b < 1) :
  a * b - 1 < a - b := by
  sorry

end ab_minus_one_lt_a_minus_b_l283_28352


namespace candy_distribution_l283_28371

theorem candy_distribution (n : Nat) (f : Nat) (h1 : n = 30) (h2 : f = 4) :
  (∃ x : Nat, (n - x) % f = 0 ∧ ∀ y : Nat, y < x → (n - y) % f ≠ 0) →
  (∃ x : Nat, (n - x) % f = 0 ∧ ∀ y : Nat, y < x → (n - y) % f ≠ 0 ∧ x = 2) :=
by sorry

end candy_distribution_l283_28371


namespace village_population_l283_28341

def initial_population : ℝ → ℝ → ℝ → Prop :=
  fun P rate years =>
    P * (1 - rate)^years = 4860

theorem village_population :
  ∃ P : ℝ, initial_population P 0.1 2 ∧ P = 6000 := by
  sorry

end village_population_l283_28341


namespace initial_student_count_l283_28314

theorem initial_student_count (initial_avg : ℝ) (new_avg : ℝ) (dropped_score : ℝ) :
  initial_avg = 61.5 →
  new_avg = 64.0 →
  dropped_score = 24 →
  ∃ n : ℕ, n * initial_avg = (n - 1) * new_avg + dropped_score ∧ n = 16 :=
by sorry

end initial_student_count_l283_28314


namespace ceiling_floor_sum_l283_28303

theorem ceiling_floor_sum : ⌈(7 : ℝ) / 3⌉ + ⌊-(7 : ℝ) / 3⌋ = 0 := by
  sorry

end ceiling_floor_sum_l283_28303


namespace ab_neg_necessary_not_sufficient_for_hyperbola_l283_28380

-- Define the condition for a hyperbola
def is_hyperbola (a b c : ℝ) : Prop :=
  ∃ (x y : ℝ), a * x^2 + b * y^2 = c ∧ c ≠ 0 ∧ a * b < 0

-- State the theorem
theorem ab_neg_necessary_not_sufficient_for_hyperbola :
  (∀ a b c : ℝ, is_hyperbola a b c → a * b < 0) ∧
  (∃ a b c : ℝ, a * b < 0 ∧ ¬(is_hyperbola a b c)) :=
sorry

end ab_neg_necessary_not_sufficient_for_hyperbola_l283_28380


namespace sqrt_six_greater_than_two_l283_28313

theorem sqrt_six_greater_than_two : Real.sqrt 6 > 2 := by
  sorry

end sqrt_six_greater_than_two_l283_28313


namespace correct_calculation_l283_28368

theorem correct_calculation (x : ℚ) : x * 15 = 45 → x * 5 * 10 = 150 := by
  sorry

end correct_calculation_l283_28368


namespace equilateral_triangle_sum_product_l283_28398

-- Define the complex numbers p, q, r
variable (p q r : ℂ)

-- Define the conditions
def is_equilateral_triangle (p q r : ℂ) : Prop :=
  Complex.abs (q - p) = 24 ∧ Complex.abs (r - q) = 24 ∧ Complex.abs (p - r) = 24

-- State the theorem
theorem equilateral_triangle_sum_product (h1 : is_equilateral_triangle p q r) 
  (h2 : Complex.abs (p + q + r) = 48) : 
  Complex.abs (p*q + p*r + q*r) = 768 := by
  sorry

end equilateral_triangle_sum_product_l283_28398


namespace cookies_distribution_l283_28309

theorem cookies_distribution (total_cookies : ℕ) (num_people : ℕ) (cookies_per_person : ℕ) : 
  total_cookies = 35 → 
  num_people = 5 → 
  total_cookies = num_people * cookies_per_person → 
  cookies_per_person = 7 := by
sorry

end cookies_distribution_l283_28309


namespace chess_sets_problem_l283_28350

theorem chess_sets_problem (x : ℕ) (y : ℕ) : 
  (x > 0) →
  (y > 0) →
  (16 * x = y * ((16 * x) / y)) →
  ((16 * x) / y + 2) * (y - 10) = 16 * x →
  ((16 * x) / y + 4) * (y - 16) = 16 * x →
  x = 15 := by
sorry

end chess_sets_problem_l283_28350


namespace subset_condition_intersection_empty_condition_l283_28348

-- Define the sets A and B
def A (m : ℝ) : Set ℝ := {x | (x + 2*m) * (x - m + 4) < 0}
def B : Set ℝ := {x | (1 - x) / (x + 2) > 0}

-- Statement for the first part of the problem
theorem subset_condition (m : ℝ) : 
  B ⊆ A m ↔ m ≥ 5 ∨ m ≤ -1/2 := by sorry

-- Statement for the second part of the problem
theorem intersection_empty_condition (m : ℝ) :
  A m ∩ B = ∅ ↔ 1 ≤ m ∧ m ≤ 2 := by sorry

end subset_condition_intersection_empty_condition_l283_28348


namespace division_remainder_proof_l283_28311

theorem division_remainder_proof (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) : 
  dividend = 690 →
  divisor = 36 →
  quotient = 19 →
  dividend = divisor * quotient + remainder →
  remainder = 6 := by
sorry

end division_remainder_proof_l283_28311


namespace equivalent_statements_l283_28344

theorem equivalent_statements :
  (∀ x : ℝ, x ≥ 0 → x^2 ≤ 0) ↔ (∀ x : ℝ, x^2 ≤ 0 → x ≥ 0) := by
  sorry

end equivalent_statements_l283_28344


namespace inequality_direction_change_l283_28318

theorem inequality_direction_change : ∃ (a b c : ℝ), a < b ∧ c * a > c * b :=
sorry

end inequality_direction_change_l283_28318


namespace ellipse_tangent_line_l283_28343

/-- Given an ellipse x^2/a^2 + y^2/b^2 = 1, the tangent line at point P(x₀, y₀) 
    has the equation x₀x/a^2 + y₀y/b^2 = 1 -/
theorem ellipse_tangent_line (a b x₀ y₀ : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
    (h : x₀^2 / a^2 + y₀^2 / b^2 = 1) :
  ∀ x y, (x₀ * x) / a^2 + (y₀ * y) / b^2 = 1 ↔ 
    (∃ t : ℝ, x = x₀ + t * (-2 * x₀ / a^2) ∧ y = y₀ + t * (-2 * y₀ / b^2)) :=
by sorry

end ellipse_tangent_line_l283_28343


namespace notepad_lasts_four_days_l283_28302

/-- Calculates the number of days a notepad lasts given the specified conditions -/
def notepadDuration (piecesPerNotepad : ℕ) (folds : ℕ) (notesPerDay : ℕ) : ℕ :=
  let sectionsPerPiece := 2^folds
  let totalNotes := piecesPerNotepad * sectionsPerPiece
  totalNotes / notesPerDay

/-- Theorem stating that under the given conditions, a notepad lasts 4 days -/
theorem notepad_lasts_four_days :
  notepadDuration 5 3 10 = 4 := by
  sorry

end notepad_lasts_four_days_l283_28302


namespace wrong_to_right_exists_l283_28306

-- Define a type for single-digit numbers (1-9)
def Digit := {n : ℕ | 1 ≤ n ∧ n ≤ 9}

-- Define a function to convert a 5-digit number to its numerical value
def to_number (a b c d e : Digit) : ℕ :=
  10000 * a + 1000 * b + 100 * c + 10 * d + e

-- State the theorem
theorem wrong_to_right_exists :
  ∃ (W R O N G I H T : Digit),
    (W ≠ R) ∧ (W ≠ O) ∧ (W ≠ N) ∧ (W ≠ G) ∧ (W ≠ I) ∧ (W ≠ H) ∧ (W ≠ T) ∧
    (R ≠ O) ∧ (R ≠ N) ∧ (R ≠ G) ∧ (R ≠ I) ∧ (R ≠ H) ∧ (R ≠ T) ∧
    (O ≠ N) ∧ (O ≠ G) ∧ (O ≠ I) ∧ (O ≠ H) ∧ (O ≠ T) ∧
    (N ≠ G) ∧ (N ≠ I) ∧ (N ≠ H) ∧ (N ≠ T) ∧
    (G ≠ I) ∧ (G ≠ H) ∧ (G ≠ T) ∧
    (I ≠ H) ∧ (I ≠ T) ∧
    (H ≠ T) ∧
    to_number W R O N G + to_number W R O N G = to_number R I G H T :=
by sorry

end wrong_to_right_exists_l283_28306


namespace ellipse_equation_from_parameters_l283_28345

/-- Represents an ellipse with axes aligned to the coordinate system -/
structure Ellipse where
  a : ℝ  -- Semi-major axis
  b : ℝ  -- Semi-minor axis
  e : ℝ  -- Eccentricity
  h : 0 < a ∧ 0 < b ∧ 0 ≤ e ∧ e < 1  -- Constraints on a, b, and e

/-- The equation of an ellipse -/
def ellipse_equation (E : Ellipse) (x y : ℝ) : Prop :=
  x^2 / E.a^2 + y^2 / E.b^2 = 1

theorem ellipse_equation_from_parameters :
  ∀ E : Ellipse,
    E.e = 2/3 →
    E.b = 4 * Real.sqrt 5 →
    (∀ x y : ℝ, ellipse_equation E x y ↔ x^2/144 + y^2/80 = 1) ∨
    (∀ x y : ℝ, ellipse_equation E x y ↔ y^2/144 + x^2/80 = 1) := by
  sorry

end ellipse_equation_from_parameters_l283_28345


namespace remaining_red_cards_l283_28300

/-- Represents a deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (red_cards : ℕ)
  (removed_red_cards : ℕ)

/-- A standard deck with half red cards and 10 red cards removed -/
def standard_deck : Deck :=
  { total_cards := 52,
    red_cards := 52 / 2,
    removed_red_cards := 10 }

/-- Theorem: The number of remaining red cards in the standard deck after removal is 16 -/
theorem remaining_red_cards (d : Deck := standard_deck) :
  d.red_cards - d.removed_red_cards = 16 := by
  sorry

end remaining_red_cards_l283_28300


namespace simplify_expression_l283_28349

theorem simplify_expression (a b : ℝ) :
  (35 * a + 70 * b + 15) + (15 * a + 54 * b + 5) - (20 * a + 85 * b + 10) = 30 * a + 39 * b + 10 := by
  sorry

end simplify_expression_l283_28349


namespace partial_fraction_decomposition_l283_28338

theorem partial_fraction_decomposition :
  ∃ (A B C : ℝ),
    (∀ x : ℝ, x ≠ 2 ∧ x ≠ 4 →
      (5 * x + 2) / ((x - 2) * (x - 4)^2) =
      A / (x - 2) + B / (x - 4) + C / (x - 4)^2) ∧
    A = 3 ∧ B = -3 ∧ C = 11 :=
by sorry

end partial_fraction_decomposition_l283_28338


namespace sphere_tangency_configurations_l283_28346

-- Define the types for our geometric objects
def Plane : Type := ℝ → ℝ → ℝ → Prop
def Sphere : Type := ℝ × ℝ × ℝ × ℝ  -- (center_x, center_y, center_z, radius)

-- Define the concept of tangency between a sphere and a plane
def spherePlaneTangent (s : Sphere) (p : Plane) : Prop := sorry

-- Define the concept of tangency between two spheres
def sphereSphereTangent (s1 s2 : Sphere) : Prop := sorry

-- Main theorem
theorem sphere_tangency_configurations 
  (p1 p2 p3 : Plane) (s : Sphere) : 
  ∃ (n : ℕ), n ≤ 16 ∧ 
  (∃ (configurations : Finset Sphere), 
    (∀ s' ∈ configurations, 
      spherePlaneTangent s' p1 ∧ 
      spherePlaneTangent s' p2 ∧ 
      spherePlaneTangent s' p3 ∧ 
      sphereSphereTangent s' s) ∧
    configurations.card = n) := by sorry

end sphere_tangency_configurations_l283_28346


namespace wine_card_probability_l283_28388

theorem wine_card_probability : 
  let n_card_types : ℕ := 3
  let n_bottles : ℕ := 5
  let total_outcomes : ℕ := n_card_types^n_bottles
  let two_type_outcomes : ℕ := Nat.choose n_card_types 2 * 2^n_bottles
  let one_type_outcomes : ℕ := n_card_types
  let favorable_outcomes : ℕ := total_outcomes - (two_type_outcomes - one_type_outcomes)
  (favorable_outcomes : ℚ) / total_outcomes = 50 / 81 :=
by sorry

end wine_card_probability_l283_28388


namespace xavier_yvonne_not_zelda_probability_l283_28339

/-- The probability that Xavier and Yvonne solve a problem but Zelda does not,
    given their individual success probabilities -/
theorem xavier_yvonne_not_zelda_probability 
  (p_xavier : ℚ) (p_yvonne : ℚ) (p_zelda : ℚ)
  (h_xavier : p_xavier = 1/5)
  (h_yvonne : p_yvonne = 1/2)
  (h_zelda : p_zelda = 5/8)
  (h_independent : True) -- Assumption of independence
  : p_xavier * p_yvonne * (1 - p_zelda) = 3/80 := by
  sorry

end xavier_yvonne_not_zelda_probability_l283_28339


namespace green_peaches_per_basket_l283_28391

/-- Proves the number of green peaches in each basket -/
theorem green_peaches_per_basket 
  (num_baskets : ℕ) 
  (red_per_basket : ℕ) 
  (total_peaches : ℕ) 
  (h1 : num_baskets = 2)
  (h2 : red_per_basket = 4)
  (h3 : total_peaches = 12) :
  (total_peaches - num_baskets * red_per_basket) / num_baskets = 2 := by
sorry

end green_peaches_per_basket_l283_28391


namespace max_intersection_points_l283_28369

theorem max_intersection_points (x_points y_points : ℕ) : x_points = 15 → y_points = 6 → 
  (x_points.choose 2) * (y_points.choose 2) = 1575 := by sorry

end max_intersection_points_l283_28369


namespace expression_evaluation_l283_28312

theorem expression_evaluation :
  let x : ℚ := 1/2
  let y : ℤ := -3
  3 * (x^2 - 2*x^2*y) - 3*x^2 + 2*y - 2*(x^2*y + y) = 6 := by
  sorry

end expression_evaluation_l283_28312


namespace replaced_student_weight_is_96_l283_28342

/-- The weight of the replaced student given the conditions of the problem -/
def replaced_student_weight (initial_students : ℕ) (new_student_weight : ℝ) (average_decrease : ℝ) : ℝ :=
  let total_weight_decrease := initial_students * average_decrease
  let weight_difference := total_weight_decrease + new_student_weight
  weight_difference

/-- Theorem stating that under the given conditions, the replaced student's weight is 96 kg -/
theorem replaced_student_weight_is_96 :
  replaced_student_weight 4 64 8 = 96 := by
  sorry

end replaced_student_weight_is_96_l283_28342


namespace cars_in_north_america_l283_28370

def total_cars : ℕ := 6755
def cars_in_europe : ℕ := 2871

theorem cars_in_north_america : total_cars - cars_in_europe = 3884 := by
  sorry

end cars_in_north_america_l283_28370


namespace parabola_and_line_properties_l283_28330

/-- Represents a parabola of the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- Represents a line of the form y = kx + m -/
structure Line where
  k : ℝ
  m : ℝ

/-- The main theorem stating the properties of the parabola and line -/
theorem parabola_and_line_properties
  (p : Parabola)
  (l : Line)
  (passes_through_A : p.a + p.b + p.c = 0)
  (axis_of_symmetry : ∀ x, p.a * (x - 3)^2 + p.b * (x - 3) + p.c = p.a * x^2 + p.b * x + p.c)
  (line_passes_through_A : l.k + l.m = 0)
  (line_passes_through_B : ∃ x, p.a * x^2 + p.b * x + p.c = l.k * x + l.m ∧ x ≠ 1)
  (triangle_area : |l.m| = 4) :
  ((l.k = -4 ∧ l.m = 4 ∧ p.a = 2 ∧ p.b = -12 ∧ p.c = 10) ∨
   (l.k = 4 ∧ l.m = -4 ∧ p.a = -2 ∧ p.b = 12 ∧ p.c = -10)) :=
by sorry

end parabola_and_line_properties_l283_28330


namespace sequence_property_l283_28332

def sequence_a (n : ℕ+) : ℚ :=
  1 / (2 * n - 1)

theorem sequence_property (n : ℕ+) :
  let a : ℕ+ → ℚ := sequence_a
  (n = 1 → a n = 1) ∧
  (∀ k : ℕ+, a k ≠ 0) ∧
  (∀ k : ℕ+, k ≥ 2 → a k + 2 * a k * a (k - 1) - a (k - 1) = 0) →
  a n = 1 / (2 * n - 1) :=
by
  sorry

#check sequence_property

end sequence_property_l283_28332


namespace profit_share_difference_l283_28396

/-- Given investments and B's profit share, calculate the difference between A's and C's profit shares -/
theorem profit_share_difference (a b c b_profit : ℕ) 
  (h1 : a = 8000) 
  (h2 : b = 10000) 
  (h3 : c = 12000) 
  (h4 : b_profit = 1700) : 
  (c * b_profit / b) - (a * b_profit / b) = 680 := by
  sorry

end profit_share_difference_l283_28396


namespace not_power_of_two_concat_l283_28399

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

def concat_numbers (nums : List ℕ) : ℕ := sorry

theorem not_power_of_two_concat :
  ∀ (perm : List ℕ),
    (∀ n ∈ perm, is_five_digit n) →
    (perm.length = 88889) →
    (∀ n, is_five_digit n → n ∈ perm) →
    ¬ ∃ k : ℕ, concat_numbers perm = 2^k :=
by sorry

end not_power_of_two_concat_l283_28399


namespace f_increasing_implies_a_range_l283_28377

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (|x - a|)

theorem f_increasing_implies_a_range (a : ℝ) :
  (∀ x y, x ∈ Set.Ici 1 → y ∈ Set.Ici 1 → x ≤ y → f a x ≤ f a y) →
  a ∈ Set.Iic 1 :=
by sorry

end f_increasing_implies_a_range_l283_28377


namespace clock_sale_correct_l283_28393

/-- Represents the clock selling scenario --/
structure ClockSale where
  originalCost : ℝ
  collectorPrice : ℝ
  buybackPrice : ℝ
  finalPrice : ℝ

/-- The clock sale scenario satisfying all given conditions --/
def clockScenario : ClockSale :=
  { originalCost := 250,
    collectorPrice := 300,
    buybackPrice := 150,
    finalPrice := 270 }

/-- Theorem stating that the given scenario satisfies all conditions and results in the correct final price --/
theorem clock_sale_correct (c : ClockSale) (h : c = clockScenario) : 
  c.collectorPrice = c.originalCost * 1.2 ∧ 
  c.buybackPrice = c.collectorPrice * 0.5 ∧
  c.originalCost - c.buybackPrice = 100 ∧
  c.finalPrice = c.buybackPrice * 1.8 := by
  sorry

#check clock_sale_correct

end clock_sale_correct_l283_28393


namespace min_a_value_for_quasi_periodic_function_l283_28336

-- Define the a-level quasi-periodic function
def is_a_level_quasi_periodic (f : ℝ → ℝ) (a : ℝ) (D : Set ℝ) : Prop :=
  ∃ T : ℝ, T ≠ 0 ∧ ∀ x ∈ D, a * f x = f (x + T)

-- Define the function f on [1, 2)
def f_on_initial_interval (x : ℝ) : ℝ := 2 * x + 1

-- Main theorem
theorem min_a_value_for_quasi_periodic_function :
  ∀ f : ℝ → ℝ,
  (∀ a : ℝ, is_a_level_quasi_periodic f a (Set.Ici 1)) →
  (∀ x ∈ Set.Icc 1 2, f x = f_on_initial_interval x) →
  (∀ x y, x < y → x ≥ 1 → f x < f y) →
  (∃ a : ℝ, ∀ b : ℝ, is_a_level_quasi_periodic f b (Set.Ici 1) → a ≤ b) →
  (∀ a : ℝ, is_a_level_quasi_periodic f a (Set.Ici 1) → a ≥ 5/3) :=
by sorry

end min_a_value_for_quasi_periodic_function_l283_28336


namespace power_relation_l283_28397

theorem power_relation (a m n : ℝ) (h1 : a^m = 8) (h2 : a^n = 4) : a^(m-2*n) = 1/2 := by
  sorry

end power_relation_l283_28397


namespace tank_emptied_in_three_minutes_l283_28324

/-- Represents the time to empty a water tank given specific conditions. -/
def time_to_empty_tank (initial_fill : ℚ) (fill_rate : ℚ) (empty_rate : ℚ) : ℚ :=
  initial_fill / (empty_rate - fill_rate)

/-- Theorem stating that under given conditions, the tank will be emptied in 3 minutes. -/
theorem tank_emptied_in_three_minutes :
  let initial_fill : ℚ := 1/5
  let fill_rate : ℚ := 1/10
  let empty_rate : ℚ := 1/6
  time_to_empty_tank initial_fill fill_rate empty_rate = 3 := by
  sorry

#eval time_to_empty_tank (1/5) (1/10) (1/6)

end tank_emptied_in_three_minutes_l283_28324


namespace original_savings_l283_28329

/-- Proves that if a person spends 4/5 of their savings on furniture and the remaining 1/5 on a TV that costs $100, their original savings were $500. -/
theorem original_savings (savings : ℝ) (furniture_fraction : ℝ) (tv_cost : ℝ) : 
  furniture_fraction = 4/5 → 
  tv_cost = 100 → 
  (1 - furniture_fraction) * savings = tv_cost → 
  savings = 500 := by
sorry

end original_savings_l283_28329


namespace container_volume_ratio_l283_28304

theorem container_volume_ratio : 
  ∀ (C D : ℝ), C > 0 → D > 0 → (3/4 * C = 2/3 * D) → C / D = 8/9 := by
  sorry

end container_volume_ratio_l283_28304


namespace compound_oxygen_count_l283_28301

/-- Represents the number of atoms of each element in the compound -/
structure Compound where
  hydrogen : ℕ
  bromine : ℕ
  oxygen : ℕ

/-- Calculates the molecular weight of a compound given atomic weights -/
def molecularWeight (c : Compound) (h_weight o_weight br_weight : ℕ) : ℕ :=
  c.hydrogen * h_weight + c.bromine * br_weight + c.oxygen * o_weight

theorem compound_oxygen_count :
  ∀ (c : Compound),
    c.hydrogen = 1 →
    c.bromine = 1 →
    molecularWeight c 1 16 80 = 129 →
    c.oxygen = 3 := by
  sorry

end compound_oxygen_count_l283_28301


namespace circle_C_equation_l283_28328

def symmetric_point (p q : ℝ × ℝ) : Prop :=
  p.1 + q.1 = p.2 + q.2 ∧ p.1 - q.1 = q.2 - p.2

def circle_equation (center : ℝ × ℝ) (radius : ℝ) (x y : ℝ) : Prop :=
  (x - center.1)^2 + (y - center.2)^2 = radius^2

theorem circle_C_equation (center : ℝ × ℝ) :
  symmetric_point center (1, 0) →
  circle_equation center 1 x y ↔ x^2 + (y - 1)^2 = 1 :=
by sorry

end circle_C_equation_l283_28328


namespace remainder_3056_div_32_l283_28333

theorem remainder_3056_div_32 : 3056 % 32 = 16 := by
  sorry

end remainder_3056_div_32_l283_28333


namespace freds_allowance_l283_28372

theorem freds_allowance (allowance : ℝ) : 
  allowance / 2 + 6 = 14 → allowance = 16 := by sorry

end freds_allowance_l283_28372


namespace print_height_preservation_l283_28337

/-- Given a painting and its print with preserved aspect ratio, calculate the height of the print -/
theorem print_height_preservation (original_width original_height print_width : ℝ) 
  (hw : original_width = 15) 
  (hh : original_height = 10) 
  (pw : print_width = 37.5) :
  let print_height := (print_width * original_height) / original_width
  print_height = 25 := by
  sorry

end print_height_preservation_l283_28337


namespace sum_m_n_equals_51_l283_28335

/-- A function that returns the number of positive divisors of a natural number -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- The smallest positive integer with only two positive divisors -/
def m : ℕ := sorry

/-- The largest integer less than 50 with exactly three positive divisors -/
def n : ℕ := sorry

theorem sum_m_n_equals_51 : m + n = 51 := by
  sorry

end sum_m_n_equals_51_l283_28335


namespace solve_matrix_inverse_l283_28334

def matrix_inverse_problem (c d x y : ℚ) : Prop :=
  let A : Matrix (Fin 2) (Fin 2) ℚ := !![4, c; x, 13]
  let B : Matrix (Fin 2) (Fin 2) ℚ := !![13, y; 3, d]
  (A * B = 1) → (x = -3 ∧ y = 17/4 ∧ c + d = -16)

theorem solve_matrix_inverse :
  ∃ c d x y : ℚ, matrix_inverse_problem c d x y :=
sorry

end solve_matrix_inverse_l283_28334


namespace manuscript_productivity_l283_28384

/-- Given a manuscript with 60,000 words, written over 120 hours including 20 hours of breaks,
    the average productivity during actual writing time is 600 words per hour. -/
theorem manuscript_productivity (total_words : ℕ) (total_hours : ℕ) (break_hours : ℕ)
    (h1 : total_words = 60000)
    (h2 : total_hours = 120)
    (h3 : break_hours = 20) :
  (total_words : ℝ) / (total_hours - break_hours : ℝ) = 600 := by
  sorry

end manuscript_productivity_l283_28384


namespace abs_neg_x_eq_2023_l283_28310

theorem abs_neg_x_eq_2023 (x : ℝ) :
  |(-x)| = 2023 → x = 2023 ∨ x = -2023 := by
  sorry

end abs_neg_x_eq_2023_l283_28310


namespace second_quadrant_condition_l283_28394

-- Define the complex number z as a function of m
def z (m : ℝ) : ℂ := Complex.mk (m - 2) (m + 1)

-- Define what it means for a complex number to be in the second quadrant
def in_second_quadrant (w : ℂ) : Prop := w.re < 0 ∧ w.im > 0

-- State the theorem
theorem second_quadrant_condition (m : ℝ) : 
  in_second_quadrant (z m) ↔ -1 < m ∧ m < 2 := by sorry

end second_quadrant_condition_l283_28394


namespace tan_and_cos_relations_l283_28319

theorem tan_and_cos_relations (θ : Real) (h : Real.tan θ = 2) :
  Real.tan (π / 4 - θ) = -1 / 3 ∧ Real.cos (2 * θ) = -3 / 5 := by
  sorry

end tan_and_cos_relations_l283_28319


namespace unique_nine_digit_number_l283_28322

/-- A permutation of digits 1 to 9 -/
def Permutation9 := Fin 9 → Fin 9

/-- Checks if a function is a valid permutation of digits 1 to 9 -/
def is_valid_permutation (p : Permutation9) : Prop :=
  Function.Injective p ∧ Function.Surjective p

/-- Converts a permutation to a natural number -/
def permutation_to_nat (p : Permutation9) : ℕ :=
  (List.range 9).foldl (fun acc i => acc * 10 + (p i).val + 1) 0

/-- The property that a permutation decreases by 8 times after rearrangement -/
def decreases_by_8_times (p : Permutation9) : Prop :=
  ∃ q : Permutation9, is_valid_permutation q ∧ permutation_to_nat p = 8 * permutation_to_nat q

theorem unique_nine_digit_number :
  ∃! p : Permutation9, is_valid_permutation p ∧ decreases_by_8_times p ∧ permutation_to_nat p = 123456789 :=
sorry

end unique_nine_digit_number_l283_28322


namespace ceil_e_plus_pi_l283_28315

theorem ceil_e_plus_pi : ⌈Real.exp 1 + Real.pi⌉ = 6 := by sorry

end ceil_e_plus_pi_l283_28315


namespace fair_attendance_l283_28376

/-- Given the number of people attending a fair over three years, prove the values of x, y, and z. -/
theorem fair_attendance (x y z : ℕ) 
  (h1 : z = 2 * y)
  (h2 : x = z - 200)
  (h3 : y = 600) :
  x = 1000 ∧ y = 600 ∧ z = 1200 := by
  sorry

end fair_attendance_l283_28376


namespace inverse_100_mod_101_l283_28383

theorem inverse_100_mod_101 : ∃ x : ℕ, 0 ≤ x ∧ x ≤ 100 ∧ (100 * x) % 101 = 1 :=
by
  use 100
  sorry

end inverse_100_mod_101_l283_28383


namespace soccer_lineup_count_l283_28392

theorem soccer_lineup_count (n : ℕ) (h : n = 18) : 
  n * (n - 1) * (Nat.choose (n - 2) 9) = 3501120 :=
by sorry

end soccer_lineup_count_l283_28392


namespace local_extremum_and_inequality_l283_28382

-- Define the function f
def f (a b x : ℝ) : ℝ := x^3 + 3*a*x^2 + b*x + a^2

-- State the theorem
theorem local_extremum_and_inequality (a b : ℝ) :
  (∃ δ > 0, ∀ x ∈ Set.Ioo (-1 - δ) (-1 + δ), f a b x ≥ f a b (-1)) ∧
  (f a b (-1) = 0) ∧
  (∀ x ∈ Set.Icc (-2) 1, f a b x ≤ 20) →
  a = 2 ∧ b = 9 ∧ (∀ m : ℝ, (∀ x ∈ Set.Icc (-2) 1, f a b x ≤ m) ↔ m ≥ 20) :=
by sorry

end local_extremum_and_inequality_l283_28382


namespace systematic_sampling_l283_28351

/-- Systematic sampling for a given population and sample size -/
theorem systematic_sampling
  (population : ℕ)
  (sample_size : ℕ)
  (h_pop : population = 1650)
  (h_sample : sample_size = 35)
  : ∃ (removed : ℕ) (segments : ℕ),
    removed = 5 ∧
    segments = 35 ∧
    (population - removed) % segments = 0 ∧
    (population - removed) / segments = sample_size :=
by sorry

end systematic_sampling_l283_28351


namespace distribution_count_correct_l283_28325

/-- The number of ways to distribute 5 indistinguishable objects into 4 distinguishable containers,
    where 2 containers are of type A and 2 are of type B,
    with at least one object in a type A container. -/
def distribution_count : ℕ := 30

/-- The number of cousins -/
def num_cousins : ℕ := 5

/-- The total number of rooms -/
def num_rooms : ℕ := 4

/-- The number of rooms with a garden view -/
def num_garden_view : ℕ := 2

/-- The number of rooms without a garden view -/
def num_no_garden_view : ℕ := 2

/-- Theorem stating that the distribution count is correct -/
theorem distribution_count_correct :
  distribution_count = 30 ∧
  num_cousins = 5 ∧
  num_rooms = 4 ∧
  num_garden_view = 2 ∧
  num_no_garden_view = 2 ∧
  num_garden_view + num_no_garden_view = num_rooms :=
by sorry

end distribution_count_correct_l283_28325


namespace simplify_first_expression_simplify_second_expression_l283_28365

-- First expression
theorem simplify_first_expression (a b : ℝ) :
  2*a + 3*b + 6*a + 9*b - 8*a - 5 = 12*b - 5 := by sorry

-- Second expression
theorem simplify_second_expression (x : ℝ) :
  2*(3*x + 1) - (4 - x - x^2) = x^2 + 7*x - 2 := by sorry

end simplify_first_expression_simplify_second_expression_l283_28365


namespace girls_minus_boys_l283_28358

/-- The number of boys in Grade 7 Class 1 -/
def num_boys (a b : ℤ) : ℤ := 2*a - b

/-- The number of girls in Grade 7 Class 1 -/
def num_girls (a b : ℤ) : ℤ := 3*a + b

/-- The theorem stating the difference between the number of girls and boys -/
theorem girls_minus_boys (a b : ℤ) : 
  num_girls a b - num_boys a b = a + 2*b := by
  sorry

end girls_minus_boys_l283_28358


namespace divisibility_problem_l283_28307

theorem divisibility_problem (N : ℕ) : 
  N = 7 * 13 + 1 → (N / 8 + N % 8 = 15) := by
  sorry

end divisibility_problem_l283_28307


namespace segment_ratios_l283_28361

/-- Given four points P, Q, R, S on a line in that order, with given distances between them,
    prove the ratios of certain segments. -/
theorem segment_ratios (P Q R S : ℝ) (h_order : P < Q ∧ Q < R ∧ R < S)
    (h_PQ : Q - P = 3) (h_QR : R - Q = 7) (h_PS : S - P = 20) :
    (R - P) / (S - Q) = 10 / 17 ∧ (S - P) / (Q - P) = 20 / 3 := by
  sorry

end segment_ratios_l283_28361


namespace f_sum_inequality_l283_28356

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x + 1 else 2^x

theorem f_sum_inequality (x : ℝ) :
  (f x + f (x - 1/2) > 1) ↔ x > -1/4 := by
  sorry

end f_sum_inequality_l283_28356


namespace evaluate_expression_l283_28362

theorem evaluate_expression : 11 + Real.sqrt (-4 + 6 * 4 / 3) = 13 := by
  sorry

end evaluate_expression_l283_28362


namespace xy_sum_product_l283_28305

theorem xy_sum_product (x y : ℝ) (h1 : x * y = 3) (h2 : x + y = 5) :
  x^2 * y + x * y^2 = 15 := by
sorry

end xy_sum_product_l283_28305
