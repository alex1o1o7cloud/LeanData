import Mathlib

namespace cubic_function_properties_l1872_187298

-- Define the function f
def f (a b c x : ℝ) : ℝ := x^3 + 3*a*x^2 + 3*b*x + c

-- Define the derivative of f
def f' (a b x : ℝ) : ℝ := 3*x^2 + 6*a*x + 3*b

theorem cubic_function_properties :
  ∀ a b c : ℝ,
  (∃ x : ℝ, f' a b x = 0 ∧ x = 2) →
  (f' a b 1 = -3) →
  (a = -1 ∧ b = 0) ∧
  (∃ x_max x_min : ℝ, f (-1) 0 c x_max - f (-1) 0 c x_min = 4) :=
by sorry

end cubic_function_properties_l1872_187298


namespace factorial_sum_equality_l1872_187276

theorem factorial_sum_equality : 7 * Nat.factorial 7 + 6 * Nat.factorial 6 + 5 * Nat.factorial 5 + Nat.factorial 5 = 40320 := by
  sorry

end factorial_sum_equality_l1872_187276


namespace calculation_problems_l1872_187222

theorem calculation_problems :
  (((1 : ℚ) / 2 - 5 / 9 + 7 / 12) * (-36 : ℚ) = -19) ∧
  ((-199 - 24 / 25) * (5 : ℚ) = -999 - 4 / 5) := by
  sorry

end calculation_problems_l1872_187222


namespace pot_holds_three_liters_l1872_187243

/-- Represents the volume of a pot in liters -/
def pot_volume (drops_per_minute : ℕ) (ml_per_drop : ℕ) (minutes_to_fill : ℕ) : ℚ :=
  (drops_per_minute * ml_per_drop * minutes_to_fill : ℚ) / 1000

/-- Theorem stating that a pot filled by a leak with given parameters holds 3 liters -/
theorem pot_holds_three_liters :
  pot_volume 3 20 50 = 3 := by
  sorry

end pot_holds_three_liters_l1872_187243


namespace square_sum_given_difference_and_product_l1872_187230

theorem square_sum_given_difference_and_product (x y : ℝ) 
  (h1 : x - y = 10) (h2 : x * y = 9) : x^2 + y^2 = 118 := by
  sorry

end square_sum_given_difference_and_product_l1872_187230


namespace expected_girls_left_10_7_l1872_187286

/-- The expected number of girls standing to the left of all boys in a random arrangement -/
def expected_girls_left (num_boys num_girls : ℕ) : ℚ :=
  num_girls / (num_boys + 1 : ℚ)

/-- Theorem: In a random arrangement of 10 boys and 7 girls, 
    the expected number of girls standing to the left of all boys is 7/11 -/
theorem expected_girls_left_10_7 :
  expected_girls_left 10 7 = 7 / 11 := by sorry

end expected_girls_left_10_7_l1872_187286


namespace band_member_earnings_l1872_187296

theorem band_member_earnings 
  (attendees : ℕ) 
  (revenue_share : ℚ) 
  (ticket_price : ℕ) 
  (band_members : ℕ) 
  (h1 : attendees = 500) 
  (h2 : revenue_share = 7/10) 
  (h3 : ticket_price = 30) 
  (h4 : band_members = 4) :
  (attendees * ticket_price * revenue_share) / band_members = 2625 := by
sorry

end band_member_earnings_l1872_187296


namespace quadrilateral_perimeter_area_inequality_l1872_187274

-- Define a convex quadrilateral
structure ConvexQuadrilateral where
  vertices : Fin 4 → ℝ × ℝ
  is_convex : sorry

-- Define the perimeter of a quadrilateral
def perimeter (q : ConvexQuadrilateral) : ℝ := sorry

-- Define the area of a quadrilateral
def area (q : ConvexQuadrilateral) : ℝ := sorry

-- Define the perimeter of the quadrilateral formed by the centers of inscribed circles
def inscribed_centers_perimeter (q : ConvexQuadrilateral) : ℝ := sorry

-- Statement of the theorem
theorem quadrilateral_perimeter_area_inequality (q : ConvexQuadrilateral) :
  perimeter q * inscribed_centers_perimeter q > 4 * area q := by
  sorry

end quadrilateral_perimeter_area_inequality_l1872_187274


namespace monochromatic_triangle_exists_l1872_187203

-- Define the polyhedron P
structure Polyhedron :=
  (vertices : Finset (Fin 9))
  (edges : Finset (Fin 9 × Fin 9))
  (base : Finset (Fin 7))
  (apex1 : Fin 9)
  (apex2 : Fin 9)

-- Define the coloring of edges
def Coloring (P : Polyhedron) := (Fin 9 × Fin 9) → Bool

-- Define a valid coloring
def ValidColoring (P : Polyhedron) (c : Coloring P) : Prop :=
  ∀ e ∈ P.edges, c e = true ∨ c e = false

-- Define a monochromatic triangle
def MonochromaticTriangle (P : Polyhedron) (c : Coloring P) : Prop :=
  ∃ (v1 v2 v3 : Fin 9), v1 ≠ v2 ∧ v2 ≠ v3 ∧ v1 ≠ v3 ∧
    (v1, v2) ∈ P.edges ∧ (v2, v3) ∈ P.edges ∧ (v1, v3) ∈ P.edges ∧
    c (v1, v2) = c (v2, v3) ∧ c (v2, v3) = c (v1, v3)

-- The main theorem
theorem monochromatic_triangle_exists (P : Polyhedron) (c : Coloring P)
    (h_valid : ValidColoring P c) :
    MonochromaticTriangle P c := by
  sorry

end monochromatic_triangle_exists_l1872_187203


namespace cookies_baked_l1872_187253

/-- Given 5 pans of cookies with 8 cookies per pan, prove that the total number of cookies is 40. -/
theorem cookies_baked (pans : ℕ) (cookies_per_pan : ℕ) (h1 : pans = 5) (h2 : cookies_per_pan = 8) :
  pans * cookies_per_pan = 40 := by
  sorry

end cookies_baked_l1872_187253


namespace parallelogram_diagonals_sides_sum_l1872_187254

/-- A parallelogram with vertices A, B, C, and D. -/
structure Parallelogram :=
  (A B C D : ℝ × ℝ)
  (is_parallelogram : (A.1 - B.1, A.2 - B.2) = (D.1 - C.1, D.2 - C.2) ∧ 
                      (A.1 - D.1, A.2 - D.2) = (B.1 - C.1, B.2 - C.2))

/-- The squared distance between two points in ℝ² -/
def dist_squared (p q : ℝ × ℝ) : ℝ := (p.1 - q.1)^2 + (p.2 - q.2)^2

/-- Theorem: The sum of the squares of the diagonals of a parallelogram 
    is equal to the sum of the squares of its four sides -/
theorem parallelogram_diagonals_sides_sum (P : Parallelogram) : 
  dist_squared P.A P.C + dist_squared P.B P.D = 
  dist_squared P.A P.B + dist_squared P.B P.C + 
  dist_squared P.C P.D + dist_squared P.D P.A :=
sorry

end parallelogram_diagonals_sides_sum_l1872_187254


namespace oplus_example_1_oplus_example_2_l1872_187262

-- Define the ⊕ operation for rational numbers
def oplus (a b : ℚ) : ℚ := (a + 3 * b) / 2

-- Theorem for part (1)
theorem oplus_example_1 : 4 * (oplus 2 5) = 34 := by sorry

-- Define polynomials A and B
def A (x y : ℚ) : ℚ := x^2 + 2*x*y + y^2
def B (x y : ℚ) : ℚ := -2*x*y + y^2

-- Theorem for part (2)
theorem oplus_example_2 (x y : ℚ) : 
  (oplus (A x y) (B x y)) + (oplus (B x y) (A x y)) = 2*x^2 + 4*y^2 := by sorry

end oplus_example_1_oplus_example_2_l1872_187262


namespace triangle_tangent_ratio_l1872_187267

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively,
    prove that if a*cos(B) - b*cos(A) = (3/5)*c, then tan(A) / tan(B) = 4 -/
theorem triangle_tangent_ratio (a b c : ℝ) (A B C : ℝ) 
    (h_triangle : A + B + C = Real.pi)
    (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
    (h_angles : 0 < A ∧ A < Real.pi ∧ 0 < B ∧ B < Real.pi ∧ 0 < C ∧ C < Real.pi)
    (h_law_of_sines : a / Real.sin A = b / Real.sin B)
    (h_given : a * Real.cos B - b * Real.cos A = (3/5) * c) :
  Real.tan A / Real.tan B = 4 := by
  sorry

end triangle_tangent_ratio_l1872_187267


namespace solution_set_cubic_inequality_l1872_187247

theorem solution_set_cubic_inequality :
  {x : ℝ | x + x^3 ≥ 0} = {x : ℝ | x ≥ 0} := by sorry

end solution_set_cubic_inequality_l1872_187247


namespace phd_team_combinations_setup_correct_l1872_187295

def total_engineers : ℕ := 8
def phd_engineers : ℕ := 3
def ms_bs_engineers : ℕ := 5
def team_size : ℕ := 3

-- Function to calculate combinations
def choose (n k : ℕ) : ℕ := Nat.choose n k

-- Theorem statement
theorem phd_team_combinations : 
  choose phd_engineers 1 * choose ms_bs_engineers 2 + 
  choose phd_engineers 2 * choose ms_bs_engineers 1 + 
  choose phd_engineers 3 = 46 := by
  sorry

-- Additional theorem to ensure the setup is correct
theorem setup_correct : 
  total_engineers = phd_engineers + ms_bs_engineers ∧ 
  team_size ≤ total_engineers := by
  sorry

end phd_team_combinations_setup_correct_l1872_187295


namespace cubic_factorization_l1872_187233

theorem cubic_factorization (a : ℝ) : a^3 - 4*a^2 + 4*a = a*(a-2)^2 := by
  sorry

end cubic_factorization_l1872_187233


namespace mall_audit_sampling_is_systematic_l1872_187248

/-- Represents a sampling method --/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified
  | Other

/-- Represents an invoice stub --/
structure InvoiceStub :=
  (number : ℕ)

/-- Represents a book of invoice stubs --/
def InvoiceBook := List InvoiceStub

/-- Represents a sampling process --/
structure SamplingProcess :=
  (book : InvoiceBook)
  (initialSelection : ℕ)
  (interval : ℕ)

/-- Determines if a sampling process is systematic --/
def isSystematicSampling (process : SamplingProcess) : Prop :=
  process.initialSelection ≤ 50 ∧ 
  process.interval = 50 ∧
  (∀ n : ℕ, (process.initialSelection + n * process.interval) ∈ (process.book.map InvoiceStub.number))

/-- The main theorem to prove --/
theorem mall_audit_sampling_is_systematic 
  (book : InvoiceBook)
  (initialStub : ℕ)
  (h1 : initialStub ≤ 50)
  (h2 : initialStub ∈ (book.map InvoiceStub.number))
  : isSystematicSampling ⟨book, initialStub, 50⟩ := by
  sorry

#check mall_audit_sampling_is_systematic

end mall_audit_sampling_is_systematic_l1872_187248


namespace integer_1025_column_l1872_187245

def column_sequence := ["B", "C", "D", "E", "A"]

theorem integer_1025_column :
  let n := 1025 - 1
  let column_index := n % (List.length column_sequence)
  List.get! column_sequence column_index = "E" := by
  sorry

end integer_1025_column_l1872_187245


namespace number_equation_l1872_187238

theorem number_equation (x : ℝ) (h : 5 * x = 2 * x + 10) : 5 * x - 2 * x = 10 := by
  sorry

end number_equation_l1872_187238


namespace last_ball_is_white_l1872_187292

/-- Represents the color of a ball -/
inductive BallColor
  | White
  | Black

/-- Represents the state of the box -/
structure BoxState :=
  (white : Nat)
  (black : Nat)

/-- The process of drawing balls and applying rules -/
def drawProcess (state : BoxState) : BoxState :=
  sorry

/-- The final state of the box after the process ends -/
def finalState (initial : BoxState) : BoxState :=
  sorry

/-- Theorem stating that the last ball is always white -/
theorem last_ball_is_white (initial : BoxState) :
  initial.white = 2011 → initial.black = 2012 →
  (finalState initial).white = 1 ∧ (finalState initial).black = 0 :=
sorry

end last_ball_is_white_l1872_187292


namespace cos_sin_eighteen_degrees_identity_l1872_187221

theorem cos_sin_eighteen_degrees_identity : 
  4 * (Real.cos (18 * π / 180))^2 - 1 = 1 / (4 * (Real.sin (18 * π / 180))^2) := by
  sorry

end cos_sin_eighteen_degrees_identity_l1872_187221


namespace symmetric_point_coordinates_l1872_187249

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Symmetry with respect to the x-axis -/
def symmetricXAxis (p : Point) : Point :=
  { x := p.x, y := -p.y }

theorem symmetric_point_coordinates :
  let B : Point := { x := 4, y := -1 }
  let A : Point := symmetricXAxis B
  A.x = 4 ∧ A.y = 1 := by sorry

end symmetric_point_coordinates_l1872_187249


namespace t_shape_perimeter_l1872_187283

/-- Represents a rectangle with width and height --/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the perimeter of a rectangle --/
def Rectangle.perimeter (r : Rectangle) : ℝ := 2 * (r.width + r.height)

/-- Represents a T-shaped figure formed by two rectangles --/
structure TShape where
  vertical : Rectangle
  horizontal : Rectangle

/-- Calculates the perimeter of a T-shaped figure --/
def TShape.perimeter (t : TShape) : ℝ :=
  t.vertical.perimeter + t.horizontal.perimeter - 4 * t.horizontal.width

theorem t_shape_perimeter :
  let t : TShape := {
    vertical := { width := 2, height := 6 },
    horizontal := { width := 2, height := 4 }
  }
  t.perimeter = 24 := by sorry

end t_shape_perimeter_l1872_187283


namespace hypotenuse_length_l1872_187299

/-- A right triangle with specific medians -/
structure RightTriangle where
  /-- Length of one leg -/
  x : ℝ
  /-- Length of the other leg -/
  y : ℝ
  /-- The triangle is right-angled -/
  right_angle : x ^ 2 + y ^ 2 > 0
  /-- One median has length 3 -/
  median1 : x ^ 2 + (y / 2) ^ 2 = 3 ^ 2
  /-- The other median has length 2√13 -/
  median2 : y ^ 2 + (x / 2) ^ 2 = (2 * Real.sqrt 13) ^ 2

/-- The hypotenuse of the right triangle is 8√1.1 -/
theorem hypotenuse_length (t : RightTriangle) : 
  Real.sqrt (4 * (t.x ^ 2 + t.y ^ 2)) = 8 * Real.sqrt 1.1 := by
  sorry

#check hypotenuse_length

end hypotenuse_length_l1872_187299


namespace prob_four_draws_ge_ten_expected_value_two_draws_l1872_187297

-- Define the bags and their contents
def bagA : Finset (Fin 10) := {0,1,2,3,4,5,6,7,8,9}
def bagB : Finset (Fin 10) := {0,1,2,3,4,5,6,7,8,9}

-- Define the probabilities of drawing each color
def probRedA : ℝ := 0.8
def probWhiteA : ℝ := 0.2
def probYellowB : ℝ := 0.9
def probBlackB : ℝ := 0.1

-- Define the scoring system
def scoreRed : ℤ := 4
def scoreWhite : ℤ := -1
def scoreYellow : ℤ := 6
def scoreBlack : ℤ := -2

-- Define the game rules
def fourDraws : ℕ := 4
def minScore : ℤ := 10

-- Theorem for Question 1
theorem prob_four_draws_ge_ten (p : ℝ) : 
  p = probRedA^4 + 4 * probRedA^3 * probWhiteA → p = 0.8192 := by sorry

-- Theorem for Question 2
theorem expected_value_two_draws (ev : ℝ) :
  ev = scoreRed * probRedA * probYellowB + 
        scoreRed * probRedA * probBlackB + 
        scoreWhite * probWhiteA * probYellowB + 
        scoreWhite * probWhiteA * probBlackB → ev = 8.2 := by sorry

end prob_four_draws_ge_ten_expected_value_two_draws_l1872_187297


namespace ellipse_rolling_conditions_l1872_187282

/-- 
An ellipse with semi-axes a and b rolls without slipping on the curve y = c sin(x/a) 
and completes one revolution in one period of the sine curve. 
This theorem states the conditions that a, b, and c must satisfy.
-/
theorem ellipse_rolling_conditions 
  (a b c : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hc : c ≠ 0) 
  (h_ellipse : ∀ (t : ℝ), ∃ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1) 
  (h_curve : ∀ (x : ℝ), ∃ (y : ℝ), y = c * Real.sin (x / a)) 
  (h_roll : ∀ (t : ℝ), ∃ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1 ∧ y = c * Real.sin (x / a)) 
  (h_period : ∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), c * Real.sin (x / a) = c * Real.sin ((x + T) / a)) :
  b ≥ a ∧ c^2 = b^2 - a^2 ∧ c * b^2 < a^3 :=
sorry

end ellipse_rolling_conditions_l1872_187282


namespace solve_q_l1872_187234

theorem solve_q (p q : ℝ) 
  (h1 : p > 1) 
  (h2 : q > 1) 
  (h3 : 1/p + 1/q = 3/2) 
  (h4 : p*q = 6) : 
  q = (9 + Real.sqrt 57) / 2 := by
sorry

end solve_q_l1872_187234


namespace system_solution_l1872_187275

-- Define the system of equations
def system (x y : ℝ) : Prop :=
  x + y = 3 ∧ x - y = 1

-- Define the solution set
def solution_set : Set (ℝ × ℝ) :=
  {(2, 1)}

-- Theorem statement
theorem system_solution :
  ∀ (x y : ℝ), system x y ↔ (x, y) ∈ solution_set :=
sorry

end system_solution_l1872_187275


namespace perfect_square_fraction_l1872_187200

theorem perfect_square_fraction (a b : ℕ+) (k : ℕ) 
  (h : k = (a.val^2 + b.val^2) / (a.val * b.val + 1)) : 
  ∃ (n : ℕ), k = n^2 := by
  sorry

end perfect_square_fraction_l1872_187200


namespace kishore_savings_percentage_l1872_187213

def rent : ℕ := 5000
def milk : ℕ := 1500
def groceries : ℕ := 4500
def education : ℕ := 2500
def petrol : ℕ := 2000
def miscellaneous : ℕ := 700
def savings : ℕ := 1800

def total_expenses : ℕ := rent + milk + groceries + education + petrol + miscellaneous
def total_salary : ℕ := total_expenses + savings

theorem kishore_savings_percentage :
  (savings : ℚ) / (total_salary : ℚ) * 100 = 10 := by sorry

end kishore_savings_percentage_l1872_187213


namespace card_distribution_exists_iff_odd_l1872_187242

/-- A magic pair is a pair of consecutive numbers or the pair (1, n(n-1)/2) -/
def is_magic_pair (a b : Nat) (n : Nat) : Prop :=
  (a + 1 = b ∨ b + 1 = a) ∨ (a = 1 ∧ b = n * (n - 1) / 2) ∨ (b = 1 ∧ a = n * (n - 1) / 2)

/-- A valid distribution of cards into stacks -/
def valid_distribution (n : Nat) (stacks : Fin n → Finset Nat) : Prop :=
  (∀ i : Fin n, ∀ x ∈ stacks i, x ≤ n * (n - 1) / 2) ∧
  (∀ i j : Fin n, i ≠ j → ∃! (a b : Nat), a ∈ stacks i ∧ b ∈ stacks j ∧ is_magic_pair a b n)

theorem card_distribution_exists_iff_odd (n : Nat) (h : n > 2) :
  (∃ stacks : Fin n → Finset Nat, valid_distribution n stacks) ↔ Odd n :=
sorry

end card_distribution_exists_iff_odd_l1872_187242


namespace total_lost_words_l1872_187202

/-- Represents the number of letters in the language --/
def total_letters : ℕ := 100

/-- Represents the number of forbidden letters --/
def forbidden_letters : ℕ := 6

/-- Calculates the number of lost one-letter words --/
def lost_one_letter_words : ℕ := forbidden_letters

/-- Calculates the number of lost two-letter words with forbidden first letter --/
def lost_two_letter_first : ℕ := forbidden_letters * total_letters

/-- Calculates the number of lost two-letter words with forbidden second letter --/
def lost_two_letter_second : ℕ := total_letters * forbidden_letters

/-- Calculates the number of lost two-letter words with both letters forbidden --/
def lost_two_letter_both : ℕ := forbidden_letters * forbidden_letters

/-- Calculates the total number of lost two-letter words --/
def lost_two_letter_words : ℕ := lost_two_letter_first + lost_two_letter_second - lost_two_letter_both

/-- Theorem stating the total number of lost words --/
theorem total_lost_words :
  lost_one_letter_words + lost_two_letter_words = 1170 := by sorry

end total_lost_words_l1872_187202


namespace tan_negative_five_pi_fourths_l1872_187263

theorem tan_negative_five_pi_fourths : Real.tan (-5 * Real.pi / 4) = -1 := by
  sorry

end tan_negative_five_pi_fourths_l1872_187263


namespace equation_solutions_l1872_187279

theorem equation_solutions : 
  {x : ℝ | x^6 + (2-x)^6 = 272} = {1 + Real.sqrt 3, 1 - Real.sqrt 3} := by
sorry

end equation_solutions_l1872_187279


namespace unique_prime_between_30_and_40_with_remainder_7_mod_9_l1872_187204

theorem unique_prime_between_30_and_40_with_remainder_7_mod_9 :
  ∃! p : ℕ, Prime p ∧ 30 < p ∧ p < 40 ∧ p % 9 = 7 :=
by
  -- The proof goes here
  sorry

end unique_prime_between_30_and_40_with_remainder_7_mod_9_l1872_187204


namespace product_of_repeating_decimal_and_eight_l1872_187216

theorem product_of_repeating_decimal_and_eight :
  let t : ℚ := 456 / 999
  t * 8 = 48 / 13 := by sorry

end product_of_repeating_decimal_and_eight_l1872_187216


namespace total_profit_is_390_4_l1872_187235

/-- Represents the partnership of A, B, and C -/
structure Partnership where
  a_share : Rat
  b_share : Rat
  c_share : Rat
  a_withdrawal_time : Nat
  a_withdrawal_fraction : Rat
  profit_distribution_time : Nat
  b_profit_share : Rat

/-- Calculates the total profit given the partnership conditions -/
def calculate_total_profit (p : Partnership) : Rat :=
  sorry

/-- Theorem stating that the total profit is 390.4 given the specified conditions -/
theorem total_profit_is_390_4 (p : Partnership) 
  (h1 : p.a_share = 1/2)
  (h2 : p.b_share = 1/3)
  (h3 : p.c_share = 1/4)
  (h4 : p.a_withdrawal_time = 2)
  (h5 : p.a_withdrawal_fraction = 1/2)
  (h6 : p.profit_distribution_time = 10)
  (h7 : p.b_profit_share = 144) :
  calculate_total_profit p = 390.4 := by
  sorry

end total_profit_is_390_4_l1872_187235


namespace garden_length_l1872_187268

theorem garden_length (columns : ℕ) (tree_distance : ℝ) (boundary : ℝ) : 
  columns > 0 → 
  tree_distance > 0 → 
  boundary > 0 → 
  (columns - 1) * tree_distance + 2 * boundary = 32 → 
  columns = 12 ∧ tree_distance = 2 ∧ boundary = 5 := by
  sorry

end garden_length_l1872_187268


namespace stanley_distance_difference_l1872_187205

/-- Given Stanley's running and walking distances, prove the difference between them. -/
theorem stanley_distance_difference (run walk : ℝ) 
  (h1 : run = 0.4) 
  (h2 : walk = 0.2) : 
  run - walk = 0.2 := by
sorry

end stanley_distance_difference_l1872_187205


namespace sum_of_reciprocals_bound_l1872_187273

theorem sum_of_reciprocals_bound (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 2) :
  ∃ (z : ℝ), z ≥ 2 ∧ (∃ (x' y' : ℝ), x' > 0 ∧ y' > 0 ∧ x' + y' = 2 ∧ 1 / x' + 1 / y' = z) ∧
  (∀ (a b : ℝ), a > 0 → b > 0 → a + b = 2 → 1 / a + 1 / b ≥ 2) :=
sorry

end sum_of_reciprocals_bound_l1872_187273


namespace cos_90_degrees_eq_zero_l1872_187288

theorem cos_90_degrees_eq_zero : Real.cos (π / 2) = 0 := by
  sorry

end cos_90_degrees_eq_zero_l1872_187288


namespace inequality_solution_set_l1872_187236

theorem inequality_solution_set (m : ℝ) : 
  (∀ x : ℝ, (1/3)*(m*x - 1) > 2 - m ↔ x < -4) → m = -7 := by
  sorry

end inequality_solution_set_l1872_187236


namespace movie_watching_time_l1872_187250

/-- Represents the duration of a part of the movie watching session -/
structure MoviePart where
  watch_time : Nat
  rewind_time : Nat

/-- Calculates the total time for a movie watching session -/
def total_movie_time (parts : List MoviePart) : Nat :=
  parts.foldl (fun acc part => acc + part.watch_time + part.rewind_time) 0

/-- Theorem stating that the total time to watch the movie is 120 minutes -/
theorem movie_watching_time :
  let part1 : MoviePart := { watch_time := 35, rewind_time := 5 }
  let part2 : MoviePart := { watch_time := 45, rewind_time := 15 }
  let part3 : MoviePart := { watch_time := 20, rewind_time := 0 }
  total_movie_time [part1, part2, part3] = 120 := by
  sorry


end movie_watching_time_l1872_187250


namespace repair_labor_hours_l1872_187291

/-- Calculates the number of labor hours given the labor cost per hour, part cost, and total repair cost. -/
def labor_hours (labor_cost_per_hour : ℕ) (part_cost : ℕ) (total_cost : ℕ) : ℕ :=
  (total_cost - part_cost) / labor_cost_per_hour

/-- Proves that given the specified costs, the number of labor hours is 16. -/
theorem repair_labor_hours :
  labor_hours 75 1200 2400 = 16 := by
  sorry

end repair_labor_hours_l1872_187291


namespace regression_line_estimate_l1872_187264

-- Define the regression line
def regression_line (b a x : ℝ) : ℝ := b * x + a

-- State the theorem
theorem regression_line_estimate :
  ∀ (b a : ℝ),
  b = 1.23 →
  regression_line b a 4 = 5 →
  regression_line b a 2 = 2.54 := by
sorry

end regression_line_estimate_l1872_187264


namespace probability_both_presidents_selected_l1872_187240

/-- Represents a math club with its total number of members -/
structure MathClub where
  members : Nat
  presidents : Nat
  mascots : Nat

/-- The list of math clubs in the district -/
def mathClubs : List MathClub := [
  { members := 6, presidents := 2, mascots := 1 },
  { members := 9, presidents := 2, mascots := 1 },
  { members := 10, presidents := 2, mascots := 1 },
  { members := 11, presidents := 2, mascots := 1 }
]

/-- The number of members to be selected from a club -/
def selectCount : Nat := 4

/-- Calculates the probability of selecting both presidents when choosing
    a specific number of members from a given club -/
def probBothPresidentsSelected (club : MathClub) (selectCount : Nat) : Rat :=
  sorry

/-- Calculates the overall probability of selecting both presidents when
    choosing from a randomly selected club -/
def overallProbability (clubs : List MathClub) (selectCount : Nat) : Rat :=
  sorry

/-- The main theorem stating the probability of selecting both presidents -/
theorem probability_both_presidents_selected :
  overallProbability mathClubs selectCount = 7/25 := by sorry

end probability_both_presidents_selected_l1872_187240


namespace adjacent_probability_l1872_187224

/-- The number of people -/
def total_people : ℕ := 9

/-- The number of rows -/
def num_rows : ℕ := 3

/-- The number of chairs in each row -/
def chairs_per_row : ℕ := 3

/-- The probability of two specific people sitting next to each other in the same row -/
def probability_adjacent : ℚ := 2 / 9

theorem adjacent_probability :
  probability_adjacent = (2 : ℚ) / (total_people : ℚ) := by sorry

end adjacent_probability_l1872_187224


namespace greatest_integer_with_gcd_six_gcd_192_18_is_six_less_than_200_exists_no_greater_main_result_l1872_187289

theorem greatest_integer_with_gcd_six (n : ℕ) : n < 200 ∧ Nat.gcd n 18 = 6 → n ≤ 192 :=
by sorry

theorem gcd_192_18_is_six : Nat.gcd 192 18 = 6 :=
by sorry

theorem less_than_200 : 192 < 200 :=
by sorry

theorem exists_no_greater : ¬∃ m : ℕ, 192 < m ∧ m < 200 ∧ Nat.gcd m 18 = 6 :=
by sorry

theorem main_result : 
  (∃ n : ℕ, n < 200 ∧ Nat.gcd n 18 = 6) ∧ 
  (∀ n : ℕ, n < 200 ∧ Nat.gcd n 18 = 6 → n ≤ 192) ∧
  (Nat.gcd 192 18 = 6) ∧
  (192 < 200) :=
by sorry

end greatest_integer_with_gcd_six_gcd_192_18_is_six_less_than_200_exists_no_greater_main_result_l1872_187289


namespace calculation_proof_l1872_187259

theorem calculation_proof :
  (1 / (Real.sqrt 5 + 2) - (Real.sqrt 3 - 1)^0 - Real.sqrt (9 - 4 * Real.sqrt 5) = 2) ∧
  (2 * Real.sqrt 3 * 612 * (3 + 3/2) = 5508 * Real.sqrt 3) := by
  sorry

end calculation_proof_l1872_187259


namespace raisins_amount_l1872_187218

/-- The amount of peanuts used in the trail mix -/
def peanuts : ℝ := 0.16666666666666666

/-- The amount of chocolate chips used in the trail mix -/
def chocolate_chips : ℝ := 0.16666666666666666

/-- The total amount of trail mix -/
def total_mix : ℝ := 0.4166666666666667

/-- The amount of raisins used in the trail mix -/
def raisins : ℝ := total_mix - (peanuts + chocolate_chips)

theorem raisins_amount : raisins = 0.08333333333333337 := by sorry

end raisins_amount_l1872_187218


namespace gingerbread_red_hat_percentage_l1872_187239

/-- Calculates the percentage of gingerbread men with red hats -/
theorem gingerbread_red_hat_percentage
  (red_hats : ℕ)
  (blue_boots : ℕ)
  (both : ℕ)
  (h1 : red_hats = 6)
  (h2 : blue_boots = 9)
  (h3 : both = 3) :
  (red_hats : ℚ) / ((red_hats + blue_boots - both) : ℚ) = 1/2 := by
  sorry

end gingerbread_red_hat_percentage_l1872_187239


namespace phi_value_is_65_degrees_l1872_187290

-- Define the condition that φ is an acute angle
def is_acute_angle (φ : Real) : Prop := 0 < φ ∧ φ < Real.pi / 2

-- State the theorem
theorem phi_value_is_65_degrees :
  ∀ φ : Real,
  is_acute_angle φ →
  Real.sqrt 2 * Real.cos (20 * Real.pi / 180) = Real.sin φ - Real.cos φ →
  φ = 65 * Real.pi / 180 := by
sorry

end phi_value_is_65_degrees_l1872_187290


namespace laptop_price_after_discount_l1872_187266

/-- Calculates the final price of a laptop after a percentage discount --/
def final_price (original_price : ℕ) (discount_percent : ℕ) : ℕ :=
  original_price - (original_price * discount_percent / 100)

/-- Theorem: The final price of a laptop originally costing $800 with a 15% discount is $680 --/
theorem laptop_price_after_discount :
  final_price 800 15 = 680 := by
  sorry

end laptop_price_after_discount_l1872_187266


namespace scooter_price_l1872_187257

theorem scooter_price (upfront_payment : ℝ) (upfront_percentage : ℝ) (total_price : ℝ) : 
  upfront_payment = 240 → 
  upfront_percentage = 0.20 → 
  upfront_payment = upfront_percentage * total_price → 
  total_price = 1200 := by
sorry

end scooter_price_l1872_187257


namespace intercepts_sum_l1872_187237

/-- A line is described by the equation y - 3 = 6(x - 5). -/
def line_equation (x y : ℝ) : Prop := y - 3 = 6 * (x - 5)

/-- The x-intercept of the line. -/
def x_intercept : ℝ := 4.5

/-- The y-intercept of the line. -/
def y_intercept : ℝ := -27

theorem intercepts_sum :
  line_equation x_intercept 0 ∧
  line_equation 0 y_intercept ∧
  x_intercept + y_intercept = -22.5 := by sorry

end intercepts_sum_l1872_187237


namespace minimum_students_l1872_187277

theorem minimum_students (n : ℕ) (h1 : n > 1000) 
  (h2 : n % 10 = 0) (h3 : n % 14 = 0) (h4 : n % 18 = 0) :
  n ≥ 1260 := by
  sorry

end minimum_students_l1872_187277


namespace exponent_manipulation_l1872_187258

theorem exponent_manipulation (x y : ℝ) :
  (x - y)^4 * (y - x)^3 / (y - x)^2 = (x - y)^5 :=
by sorry

end exponent_manipulation_l1872_187258


namespace no_solution_to_equation_l1872_187256

theorem no_solution_to_equation :
  ¬∃ (x : ℝ), (x - 1) / (x + 1) - 4 / (x^2 - 1) = 1 :=
by sorry

end no_solution_to_equation_l1872_187256


namespace product_of_square_roots_l1872_187219

theorem product_of_square_roots (y : ℝ) (h : y ≥ 0) :
  Real.sqrt (48 * y) * Real.sqrt (18 * y) * Real.sqrt (50 * y) = 60 * y * Real.sqrt (3 * y) :=
by sorry

end product_of_square_roots_l1872_187219


namespace negation_of_universal_statement_l1872_187223

theorem negation_of_universal_statement (S : Set ℝ) :
  (¬ ∀ x ∈ S, |x| > 1) ↔ (∃ x ∈ S, |x| ≤ 1) := by sorry

end negation_of_universal_statement_l1872_187223


namespace exponent_rules_l1872_187271

theorem exponent_rules (a b : ℝ) : 
  (a^3 * a^3 = a^6) ∧ 
  ¬((a*b)^3 = a*b^3) ∧ 
  ¬((a^3)^3 = a^6) ∧ 
  ¬(a^8 / a^4 = a^2) := by
  sorry

end exponent_rules_l1872_187271


namespace smallest_number_l1872_187217

theorem smallest_number (a b c d : ℝ) (ha : a = -1) (hb : b = 0) (hc : c = Real.sqrt 2) (hd : d = -1/2) :
  a ≤ b ∧ a ≤ c ∧ a ≤ d := by
  sorry

end smallest_number_l1872_187217


namespace fixed_point_theorem_l1872_187220

-- Define the line equation
def line_equation (k x y : ℝ) : Prop := k * x + y - 2 = 3 * k

-- State the theorem
theorem fixed_point_theorem :
  ∀ k : ℝ, line_equation k 3 2 := by sorry

end fixed_point_theorem_l1872_187220


namespace hyperbola_asymptotes_l1872_187206

/-- Definition of the hyperbola -/
def hyperbola (x y : ℝ) : Prop :=
  (y - 1)^2 / 16 - (x + 2)^2 / 25 = 1

/-- Definition of an asymptote -/
def is_asymptote (m b : ℝ) : Prop :=
  ∀ ε > 0, ∃ M > 0, ∀ x y : ℝ, 
    hyperbola x y → (|x| > M → |y - (m * x + b)| < ε)

/-- Theorem: The asymptotes of the given hyperbola -/
theorem hyperbola_asymptotes :
  (is_asymptote (4/5) (13/5)) ∧ (is_asymptote (-4/5) (13/5)) :=
sorry

end hyperbola_asymptotes_l1872_187206


namespace coconuts_per_crab_calculation_l1872_187260

/-- The number of coconuts Max has -/
def total_coconuts : ℕ := 342

/-- The number of goats Max will have after conversion -/
def total_goats : ℕ := 19

/-- The number of crabs that can be traded for a goat -/
def crabs_per_goat : ℕ := 6

/-- The number of coconuts needed to trade for a crab -/
def coconuts_per_crab : ℕ := 3

theorem coconuts_per_crab_calculation :
  coconuts_per_crab * crabs_per_goat * total_goats = total_coconuts :=
sorry

end coconuts_per_crab_calculation_l1872_187260


namespace inequality_solution_range_l1872_187227

theorem inequality_solution_range (k : ℝ) : 
  (∀ x : ℝ, 2 * k * x^2 + k * x - 3/8 < 0) ↔ k ∈ Set.Ioc (-3) 0 := by
  sorry

end inequality_solution_range_l1872_187227


namespace seating_arrangements_count_l1872_187232

/-- The number of ways to arrange n distinct objects. -/
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

/-- The number of ways six people can sit in a row of seven chairs with the third chair vacant. -/
def seatingArrangements : ℕ := factorial 6

theorem seating_arrangements_count : seatingArrangements = 720 := by
  sorry

end seating_arrangements_count_l1872_187232


namespace smallest_three_digit_multiple_of_17_l1872_187226

theorem smallest_three_digit_multiple_of_17 : ∀ n : ℕ, 
  100 ≤ n ∧ n < 1000 ∧ 17 ∣ n → 102 ≤ n :=
by sorry

end smallest_three_digit_multiple_of_17_l1872_187226


namespace john_hat_days_l1872_187278

def total_cost : ℕ := 700
def hat_cost : ℕ := 50

theorem john_hat_days : (total_cost / hat_cost : ℕ) = 14 := by
  sorry

end john_hat_days_l1872_187278


namespace complex_number_imaginary_part_l1872_187228

theorem complex_number_imaginary_part (a : ℝ) : 
  let z : ℂ := (1 + a * Complex.I) / (1 - Complex.I)
  Complex.im z = 2 → a = 3 := by
  sorry

end complex_number_imaginary_part_l1872_187228


namespace factoring_expression_l1872_187210

theorem factoring_expression (x : ℝ) : 3*x*(x+2) + 2*(x+2) + 5*(x+2) = (x+2)*(3*x+7) := by
  sorry

end factoring_expression_l1872_187210


namespace crayon_ratio_l1872_187207

theorem crayon_ratio (total : ℕ) (blue : ℕ) (red : ℕ) : 
  total = 15 → blue = 3 → red = total - blue → (red : ℚ) / blue = 4 / 1 := by
  sorry

end crayon_ratio_l1872_187207


namespace age_sum_problem_l1872_187231

theorem age_sum_problem (a b : ℕ) (h1 : a > b) (h2 : a * b * b * b = 256) : 
  a + b + b + b = 38 := by
sorry

end age_sum_problem_l1872_187231


namespace meat_market_sales_ratio_l1872_187215

/-- Given the sales data for a Meat Market over four days, prove the ratio of Sunday to Saturday sales --/
theorem meat_market_sales_ratio :
  let thursday_sales : ℕ := 210
  let friday_sales : ℕ := 2 * thursday_sales
  let saturday_sales : ℕ := 130
  let planned_total : ℕ := 500
  let actual_total : ℕ := planned_total + 325
  let sunday_sales : ℕ := actual_total - (thursday_sales + friday_sales + saturday_sales)
  (sunday_sales : ℚ) / saturday_sales = 1 / 2 := by
  sorry

#check meat_market_sales_ratio

end meat_market_sales_ratio_l1872_187215


namespace rogers_crayons_l1872_187201

theorem rogers_crayons (new_crayons used_crayons broken_crayons : ℕ) 
  (h1 : new_crayons = 2)
  (h2 : used_crayons = 4)
  (h3 : broken_crayons = 8) :
  new_crayons + used_crayons + broken_crayons = 14 := by
  sorry

end rogers_crayons_l1872_187201


namespace remainder_theorem_l1872_187244

/-- The remainder when x³ - 3x + 5 is divided by x + 2 is 3 -/
theorem remainder_theorem (x : ℝ) : 
  (x^3 - 3*x + 5) % (x + 2) = 3 := by
sorry

end remainder_theorem_l1872_187244


namespace volcano_lake_depth_l1872_187265

/-- Represents a cone-shaped volcano partially submerged in a lake -/
structure Volcano :=
  (height : ℝ)
  (above_water_ratio : ℝ)

/-- Calculates the depth of the lake at the base of the volcano -/
def lake_depth (v : Volcano) : ℝ :=
  v.height * (1 - (v.above_water_ratio ^ (1/3)))

/-- Theorem stating the depth of the lake for a specific volcano -/
theorem volcano_lake_depth :
  let v := Volcano.mk 6000 (1/6)
  lake_depth v = 390 := by
  sorry

end volcano_lake_depth_l1872_187265


namespace undefined_expression_l1872_187229

theorem undefined_expression (b : ℝ) : 
  ¬ (∃ x : ℝ, x = (b - 1) / (b^2 - 9)) ↔ b = -3 ∨ b = 3 :=
sorry

end undefined_expression_l1872_187229


namespace joshs_initial_money_l1872_187272

theorem joshs_initial_money (hat_cost pencil_cost cookie_cost : ℚ)
  (num_cookies : ℕ) (money_left : ℚ) :
  hat_cost = 10 →
  pencil_cost = 2 →
  cookie_cost = 5/4 →
  num_cookies = 4 →
  money_left = 3 →
  hat_cost + pencil_cost + num_cookies * cookie_cost + money_left = 20 :=
by sorry

end joshs_initial_money_l1872_187272


namespace num_syt_54321_l1872_187252

/-- A partition is a non-increasing sequence of natural numbers. -/
def Partition : Type := List Nat

/-- A Standard Young Tableau is a filling of a partition shape with integers
    such that rows and columns are strictly increasing. -/
def StandardYoungTableau (p : Partition) : Type := sorry

/-- Hook length of a cell in a partition -/
def hookLength (p : Partition) (i j : Nat) : Nat := sorry

/-- Number of Standard Young Tableaux for a given partition -/
def numSYT (p : Partition) : Nat := sorry

/-- The main theorem: number of Standard Young Tableaux for shape (5,4,3,2,1) -/
theorem num_syt_54321 :
  numSYT [5, 4, 3, 2, 1] = 292864 := by sorry

end num_syt_54321_l1872_187252


namespace rational_absolute_value_inequality_l1872_187261

theorem rational_absolute_value_inequality (a : ℚ) (h : a - |a| = 2*a) : a ≤ 0 := by
  sorry

end rational_absolute_value_inequality_l1872_187261


namespace book_club_hardcover_cost_l1872_187281

/-- Proves that the cost of each hardcover book is $30 given the book club fee structure --/
theorem book_club_hardcover_cost :
  let members : ℕ := 6
  let snack_fee : ℕ := 150
  let hardcover_count : ℕ := 6
  let paperback_count : ℕ := 6
  let paperback_cost : ℕ := 12
  let total_collected : ℕ := 2412
  ∃ (hardcover_cost : ℕ),
    members * (snack_fee + hardcover_count * hardcover_cost + paperback_count * paperback_cost) = total_collected ∧
    hardcover_cost = 30 :=
by sorry

end book_club_hardcover_cost_l1872_187281


namespace sin_tan_inequality_l1872_187293

theorem sin_tan_inequality (α : Real) (h1 : 0 < α) (h2 : α < Real.pi / 2) :
  2 * Real.sin α + Real.tan α > 3 * α := by
  sorry

end sin_tan_inequality_l1872_187293


namespace mary_stickers_left_l1872_187284

/-- The number of stickers Mary has left over after distributing them in class -/
def stickers_left_over (total_stickers : ℕ) (num_friends : ℕ) (stickers_per_friend : ℕ) 
  (total_students : ℕ) (stickers_per_other : ℕ) : ℕ :=
  total_stickers - 
  (num_friends * stickers_per_friend + 
   (total_students - 1 - num_friends) * stickers_per_other)

/-- Theorem stating that Mary has 8 stickers left over -/
theorem mary_stickers_left : stickers_left_over 50 5 4 17 2 = 8 := by
  sorry

end mary_stickers_left_l1872_187284


namespace smallest_whole_number_above_sum_l1872_187211

theorem smallest_whole_number_above_sum : ℕ := by
  let sum := 3 + 1/3 + 4 + 1/4 + 5 + 1/6 + 6 + 1/7
  have h1 : sum < 19 := by sorry
  have h2 : sum > 18 := by sorry
  exact 19

end smallest_whole_number_above_sum_l1872_187211


namespace cake_muffin_mix_probability_l1872_187212

theorem cake_muffin_mix_probability :
  ∀ (total buyers cake_buyers muffin_buyers both_buyers : ℕ),
    total = 100 →
    cake_buyers = 50 →
    muffin_buyers = 40 →
    both_buyers = 18 →
    (total - (cake_buyers + muffin_buyers - both_buyers)) / total = 28 / 100 := by
  sorry

end cake_muffin_mix_probability_l1872_187212


namespace pick_school_supply_l1872_187294

/-- The number of pencils in the pencil case -/
def num_pencils : ℕ := 2

/-- The number of erasers in the pencil case -/
def num_erasers : ℕ := 4

/-- The total number of school supplies in the pencil case -/
def total_supplies : ℕ := num_pencils + num_erasers

/-- Theorem stating that the number of ways to pick up a school supply is 6 -/
theorem pick_school_supply : total_supplies = 6 := by
  sorry

end pick_school_supply_l1872_187294


namespace squares_pattern_squares_figure_100_l1872_187246

/-- The number of squares in figure n -/
def num_squares (n : ℕ) : ℕ :=
  3 * n^2 + 3 * n + 1

/-- The sequence of squares follows the given pattern for the first four figures -/
theorem squares_pattern :
  num_squares 0 = 1 ∧
  num_squares 1 = 7 ∧
  num_squares 2 = 19 ∧
  num_squares 3 = 37 := by sorry

/-- The number of squares in figure 100 is 30301 -/
theorem squares_figure_100 :
  num_squares 100 = 30301 := by sorry

end squares_pattern_squares_figure_100_l1872_187246


namespace car_speed_problem_l1872_187270

/-- Proves that for a journey of 225 km, if a car arrives 45 minutes late when traveling at 50 kmph, then its on-time average speed is 60 kmph. -/
theorem car_speed_problem (journey_length : ℝ) (late_speed : ℝ) (delay : ℝ) :
  journey_length = 225 →
  late_speed = 50 →
  delay = 3/4 →
  ∃ (on_time_speed : ℝ),
    (journey_length / on_time_speed) + delay = (journey_length / late_speed) ∧
    on_time_speed = 60 := by
  sorry

end car_speed_problem_l1872_187270


namespace line_equation_of_points_on_parabola_l1872_187280

/-- Given a parabola y² = 4x and two points on it with midpoint (2, 2), 
    the line through these points has equation x - y = 0 -/
theorem line_equation_of_points_on_parabola (A B : ℝ × ℝ) : 
  (A.2^2 = 4 * A.1) →  -- A is on the parabola
  (B.2^2 = 4 * B.1) →  -- B is on the parabola
  ((A.1 + B.1) / 2 = 2 ∧ (A.2 + B.2) / 2 = 2) →  -- midpoint is (2, 2)
  ∃ (k : ℝ), ∀ (x y : ℝ), (x - A.1) = k * (y - A.2) ∧ x - y = 0 :=
sorry

end line_equation_of_points_on_parabola_l1872_187280


namespace sugar_difference_l1872_187208

theorem sugar_difference (brown_sugar white_sugar : ℝ) 
  (h1 : brown_sugar = 0.62)
  (h2 : white_sugar = 0.25) :
  brown_sugar - white_sugar = 0.37 := by
  sorry

end sugar_difference_l1872_187208


namespace meal_cost_is_25_l1872_187214

/-- The cost of Hilary's meal at Delicious Delhi restaurant -/
def meal_cost : ℝ :=
  let samosa_price : ℝ := 2
  let pakora_price : ℝ := 3
  let lassi_price : ℝ := 2
  let samosa_quantity : ℕ := 3
  let pakora_quantity : ℕ := 4
  let tip_percentage : ℝ := 0.25
  let subtotal : ℝ := samosa_price * samosa_quantity + pakora_price * pakora_quantity + lassi_price
  let tip : ℝ := subtotal * tip_percentage
  subtotal + tip

theorem meal_cost_is_25 : meal_cost = 25 := by
  sorry

end meal_cost_is_25_l1872_187214


namespace max_intersections_eight_l1872_187251

/-- Represents a tiled floor with equilateral triangles -/
structure TriangularFloor where
  side_length : ℝ
  side_length_positive : side_length > 0

/-- Represents a needle -/
structure Needle where
  length : ℝ
  length_positive : length > 0

/-- Counts the maximum number of triangles intersected by a needle -/
def max_intersected_triangles (floor : TriangularFloor) (needle : Needle) : ℕ :=
  sorry

/-- Theorem stating the maximum number of intersected triangles -/
theorem max_intersections_eight
  (floor : TriangularFloor)
  (needle : Needle)
  (h_floor : floor.side_length = 1)
  (h_needle : needle.length = 2) :
  max_intersected_triangles floor needle = 8 :=
sorry

end max_intersections_eight_l1872_187251


namespace total_revenue_calculation_l1872_187255

/-- Calculate the total revenue from vegetable sales --/
theorem total_revenue_calculation :
  let morning_potatoes : ℕ := 29
  let morning_onions : ℕ := 15
  let morning_carrots : ℕ := 12
  let afternoon_potatoes : ℕ := 17
  let afternoon_onions : ℕ := 22
  let afternoon_carrots : ℕ := 9
  let potato_weight : ℕ := 7
  let onion_weight : ℕ := 5
  let carrot_weight : ℕ := 4
  let potato_price : ℚ := 1.75
  let onion_price : ℚ := 2.50
  let carrot_price : ℚ := 3.25

  let total_potatoes : ℕ := morning_potatoes + afternoon_potatoes
  let total_onions : ℕ := morning_onions + afternoon_onions
  let total_carrots : ℕ := morning_carrots + afternoon_carrots

  let potato_revenue : ℚ := (total_potatoes * potato_weight : ℚ) * potato_price
  let onion_revenue : ℚ := (total_onions * onion_weight : ℚ) * onion_price
  let carrot_revenue : ℚ := (total_carrots * carrot_weight : ℚ) * carrot_price

  let total_revenue : ℚ := potato_revenue + onion_revenue + carrot_revenue

  total_revenue = 1299.00 := by sorry

end total_revenue_calculation_l1872_187255


namespace lineup_count_is_636_l1872_187209

/-- Represents a basketball team with specified number of players and positions -/
structure BasketballTeam where
  total_players : ℕ
  forwards : ℕ
  guards : ℕ
  versatile_players : ℕ
  lineup_forwards : ℕ
  lineup_guards : ℕ

/-- Calculates the number of different lineups for a given basketball team -/
def count_lineups (team : BasketballTeam) : ℕ :=
  sorry

/-- Theorem stating that the number of different lineups is 636 for the given team configuration -/
theorem lineup_count_is_636 : 
  let team : BasketballTeam := {
    total_players := 12,
    forwards := 6,
    guards := 4,
    versatile_players := 2,
    lineup_forwards := 3,
    lineup_guards := 2
  }
  count_lineups team = 636 := by sorry

end lineup_count_is_636_l1872_187209


namespace widget_earnings_calculation_l1872_187225

/-- Calculates the earnings per widget given the hourly wage, work hours, 
    required widget production, and total weekly earnings. -/
def earnings_per_widget (hourly_wage : ℚ) (work_hours : ℕ) 
  (required_widgets : ℕ) (total_earnings : ℚ) : ℚ :=
  (total_earnings - hourly_wage * work_hours) / required_widgets

theorem widget_earnings_calculation : 
  earnings_per_widget (12.5) 40 500 580 = (16 : ℚ) / 100 := by
  sorry

end widget_earnings_calculation_l1872_187225


namespace range_of_a_l1872_187285

-- Define the propositions p and q
def p (x : ℝ) : Prop := x^2 - 12*x + 20 < 0
def q (x a : ℝ) : Prop := x^2 - 2*x + 1 - a^2 > 0

-- Define the theorem
theorem range_of_a (a : ℝ) :
  (a > 0) →
  (∀ x : ℝ, ¬(q x a) → ¬(p x)) →
  (0 < a ∧ a < 1) :=
by sorry

end range_of_a_l1872_187285


namespace number_equation_solution_l1872_187269

theorem number_equation_solution : ∃ x : ℝ, 3 * x + 4 = 19 ∧ x = 5 := by
  sorry

end number_equation_solution_l1872_187269


namespace symmetry_about_x_2_symmetry_about_2_0_l1872_187241

-- Define a function f: ℝ → ℝ
variable (f : ℝ → ℝ)

-- Theorem for symmetry about x = 2
theorem symmetry_about_x_2 (h : ∀ x, f (1 - x) = f (3 + x)) :
  ∀ x, f (2 - x) = f (2 + x) := by sorry

-- Theorem for symmetry about (2,0)
theorem symmetry_about_2_0 (h : ∀ x, f (1 - x) = -f (3 + x)) :
  ∀ x, f (2 - x) = -f (2 + x) := by sorry

end symmetry_about_x_2_symmetry_about_2_0_l1872_187241


namespace time_after_classes_l1872_187287

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  hLt60 : minutes < 60

/-- Adds a duration in minutes to a given time -/
def addMinutes (t : Time) (m : ℕ) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + m
  { hours := totalMinutes / 60,
    minutes := totalMinutes % 60,
    hLt60 := by sorry }

/-- The starting time of classes -/
def startTime : Time := { hours := 12, minutes := 0, hLt60 := by simp }

/-- The number of completed classes -/
def completedClasses : ℕ := 4

/-- The duration of each class in minutes -/
def classDuration : ℕ := 45

/-- Theorem: After 4 classes of 45 minutes each, starting at 12 pm, the time is 3 pm -/
theorem time_after_classes :
  (addMinutes startTime (completedClasses * classDuration)).hours = 15 := by sorry

end time_after_classes_l1872_187287
