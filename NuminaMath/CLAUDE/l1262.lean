import Mathlib

namespace arithmetic_expression_equals_24_l1262_126215

theorem arithmetic_expression_equals_24 :
  (8 * 9 / 6) + 8 = 24 :=
by sorry

end arithmetic_expression_equals_24_l1262_126215


namespace binomial_coeff_n_n_l1262_126202

theorem binomial_coeff_n_n (n : ℕ) : (n.choose n) = 1 := by
  sorry

end binomial_coeff_n_n_l1262_126202


namespace cubic_equation_roots_l1262_126241

theorem cubic_equation_roots (p q : ℝ) :
  let x₁ := (-p + Real.sqrt (p^2 - 4*q)) / 2
  let x₂ := (-p - Real.sqrt (p^2 - 4*q)) / 2
  let cubic := fun y : ℝ => y^3 - (p^2 - q)*y^2 + (p^2*q - q^2)*y - q^3
  (cubic x₁^2 = 0) ∧ (cubic (x₁*x₂) = 0) ∧ (cubic x₂^2 = 0) :=
sorry

end cubic_equation_roots_l1262_126241


namespace remainder_1897_2048_mod_600_l1262_126279

theorem remainder_1897_2048_mod_600 : (1897 * 2048) % 600 = 256 := by
  sorry

end remainder_1897_2048_mod_600_l1262_126279


namespace sum_of_three_squares_l1262_126231

/-- The value of a triangle -/
def triangle_value : ℝ := sorry

/-- The value of a square -/
def square_value : ℝ := sorry

/-- The sum of three triangles and two squares equals 18 -/
axiom eq1 : 3 * triangle_value + 2 * square_value = 18

/-- The sum of two triangles and three squares equals 22 -/
axiom eq2 : 2 * triangle_value + 3 * square_value = 22

/-- The sum of three squares equals 18 -/
theorem sum_of_three_squares : 3 * square_value = 18 := by sorry

end sum_of_three_squares_l1262_126231


namespace books_together_l1262_126291

-- Define the number of books Tim and Mike have
def tim_books : ℕ := 22
def mike_books : ℕ := 20

-- Define the total number of books
def total_books : ℕ := tim_books + mike_books

-- Theorem to prove
theorem books_together : total_books = 42 := by
  sorry

end books_together_l1262_126291


namespace value_of_expression_l1262_126254

theorem value_of_expression (x : ℝ) (h : x = 2) : 3^x - x^3 = 1 := by
  sorry

end value_of_expression_l1262_126254


namespace sqrt_of_sqrt_16_is_plus_minus_2_l1262_126223

theorem sqrt_of_sqrt_16_is_plus_minus_2 : 
  {x : ℝ | x^2 = Real.sqrt 16} = {2, -2} := by sorry

end sqrt_of_sqrt_16_is_plus_minus_2_l1262_126223


namespace adults_in_group_l1262_126249

theorem adults_in_group (children : ℕ) (meal_cost : ℕ) (total_bill : ℕ) (adults : ℕ) : 
  children = 5 → 
  meal_cost = 3 → 
  total_bill = 21 → 
  adults * meal_cost + children * meal_cost = total_bill → 
  adults = 2 := by
sorry

end adults_in_group_l1262_126249


namespace classroom_window_2023_l1262_126269

/-- Represents a digit as seen through a transparent surface --/
inductive MirroredDigit
| Zero
| Two
| Three

/-- Represents the appearance of a number when viewed through a transparent surface --/
def mirror_number (n : List Nat) : List MirroredDigit :=
  sorry

/-- The property of being viewed from the opposite side of a transparent surface --/
def viewed_from_opposite_side (original : List Nat) (mirrored : List MirroredDigit) : Prop :=
  mirror_number original = mirrored.reverse

theorem classroom_window_2023 :
  viewed_from_opposite_side [2, 0, 2, 3] [MirroredDigit.Three, MirroredDigit.Two, MirroredDigit.Zero, MirroredDigit.Two] :=
by sorry

end classroom_window_2023_l1262_126269


namespace julia_cakes_l1262_126257

theorem julia_cakes (x : ℕ) : 
  (x * 6 - 3 = 21) → x = 4 := by
  sorry

end julia_cakes_l1262_126257


namespace range_of_b_l1262_126220

-- Define the curve C
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (x y : ℝ), p = (x, y) ∧ Real.sqrt (x^2 + y^2) + |x - 4| = 5}

-- Define the point B
def B (b : ℝ) : ℝ × ℝ := (b, 0)

-- Define the symmetry condition
def symmetricPoints (b : ℝ) : Prop :=
  ∃ (p1 p2 p3 p4 p5 p6 : ℝ × ℝ),
    p1 ∈ C ∧ p2 ∈ C ∧ p3 ∈ C ∧ p4 ∈ C ∧ p5 ∈ C ∧ p6 ∈ C ∧
    p1 ≠ p2 ∧ p3 ≠ p4 ∧ p5 ≠ p6 ∧
    (p1.1 + p2.1) / 2 = b ∧ (p3.1 + p4.1) / 2 = b ∧ (p5.1 + p6.1) / 2 = b

-- Theorem statement
theorem range_of_b :
  ∀ b : ℝ, (∀ p ∈ C, Real.sqrt ((p.1)^2 + (p.2)^2) + |p.1 - 4| = 5) →
            symmetricPoints b →
            2 < b ∧ b < 4 :=
sorry

end range_of_b_l1262_126220


namespace partial_fraction_decomposition_l1262_126201

theorem partial_fraction_decomposition :
  ∃ (A B C : ℚ),
    (A = 1/3 ∧ B = 2/3 ∧ C = 1/3) ∧
    (∀ x : ℚ, x ≠ -2 ∧ x^2 + x + 1 ≠ 0 →
      (x + 1)^2 / ((x + 2) * (x^2 + x + 1)) =
      A / (x + 2) + (B * x + C) / (x^2 + x + 1)) :=
by sorry

end partial_fraction_decomposition_l1262_126201


namespace solution_set_inequality1_solution_set_inequality2_l1262_126200

-- First inequality
theorem solution_set_inequality1 (x : ℝ) :
  x ≠ 2 →
  ((x + 1) / (x - 2) ≥ 3) ↔ (2 < x ∧ x ≤ 7/2) :=
sorry

-- Second inequality
theorem solution_set_inequality2 (x a : ℝ) :
  x^2 - a*x - 2*a^2 ≤ 0 ↔
    (a = 0 ∧ x = 0) ∨
    (a > 0 ∧ -a ≤ x ∧ x ≤ 2*a) ∨
    (a < 0 ∧ 2*a ≤ x ∧ x ≤ -a) :=
sorry

end solution_set_inequality1_solution_set_inequality2_l1262_126200


namespace distance_set_exists_l1262_126251

/-- A set of points in the plane satisfying the distance condition -/
def DistanceSet (m : ℕ) (S : Set (ℝ × ℝ)) : Prop :=
  (∀ A ∈ S, ∃! (points : Finset (ℝ × ℝ)), 
    points.card = m ∧ 
    (∀ B ∈ points, B ∈ S ∧ Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 1))

/-- The existence of a finite set satisfying the distance condition for any m ≥ 1 -/
theorem distance_set_exists (m : ℕ) (hm : m ≥ 1) : 
  ∃ S : Set (ℝ × ℝ), S.Finite ∧ DistanceSet m S :=
sorry

end distance_set_exists_l1262_126251


namespace yellow_mugs_count_l1262_126285

/-- Represents the number of mugs of each color in Hannah's collection --/
structure MugCollection where
  red : ℕ
  blue : ℕ
  yellow : ℕ
  other : ℕ

/-- The conditions of Hannah's mug collection --/
def hannahsMugs : MugCollection → Prop
  | m => m.red + m.blue + m.yellow + m.other = 40 ∧
         m.blue = 3 * m.red ∧
         m.red = m.yellow / 2 ∧
         m.other = 4

theorem yellow_mugs_count (m : MugCollection) (h : hannahsMugs m) : m.yellow = 12 := by
  sorry

#check yellow_mugs_count

end yellow_mugs_count_l1262_126285


namespace scientific_notation_correct_l1262_126298

-- Define the original number
def original_number : ℕ := 262883000000

-- Define the scientific notation components
def significand : ℚ := 2.62883
def exponent : ℕ := 11

-- Theorem statement
theorem scientific_notation_correct : 
  (significand * (10 : ℚ) ^ exponent) = original_number := by
  sorry

end scientific_notation_correct_l1262_126298


namespace min_radius_of_circle_l1262_126237

theorem min_radius_of_circle (r a b : ℝ) : 
  ((a - (r + 1))^2 + b^2 = r^2) →  -- Point (a, b) is on the circle
  (b^2 ≥ 4*a) →                    -- Condition b^2 ≥ 4a
  (r ≥ 0) →                        -- Radius is non-negative
  (r ≥ 4) :=                       -- Minimum value of r is 4
by sorry

end min_radius_of_circle_l1262_126237


namespace well_depth_proof_l1262_126243

/-- The depth of the well in feet -/
def depth : ℝ := 918.09

/-- The total time from dropping the stone to hearing it hit the bottom, in seconds -/
def total_time : ℝ := 8.5

/-- The speed of sound in feet per second -/
def sound_speed : ℝ := 1100

/-- The function describing the distance fallen by the stone after t seconds -/
def stone_fall (t : ℝ) : ℝ := 16 * t^2

theorem well_depth_proof :
  ∃ (t_fall : ℝ), 
    t_fall > 0 ∧
    stone_fall t_fall = depth ∧
    t_fall + depth / sound_speed = total_time :=
sorry

end well_depth_proof_l1262_126243


namespace trapezoid_area_l1262_126278

/-- A trapezoid with given dimensions -/
structure Trapezoid where
  AD : ℝ  -- Length of longer base
  BC : ℝ  -- Length of shorter base
  AC : ℝ  -- Length of one diagonal
  BD : ℝ  -- Length of other diagonal

/-- The area of a trapezoid with the given dimensions is 80 -/
theorem trapezoid_area (T : Trapezoid)
    (h1 : T.AD = 24)
    (h2 : T.BC = 8)
    (h3 : T.AC = 13)
    (h4 : T.BD = 5 * Real.sqrt 17) :
    (T.AD + T.BC) * Real.sqrt (T.AC ^ 2 - ((T.AD - T.BC) / 2 + T.BC) ^ 2) / 2 = 80 := by
  sorry

end trapezoid_area_l1262_126278


namespace square_perimeter_l1262_126250

theorem square_perimeter (rectangleA_perimeter : ℝ) (squareB_area_ratio : ℝ) :
  rectangleA_perimeter = 30 →
  squareB_area_ratio = 1/3 →
  ∃ (rectangleA_length rectangleA_width : ℝ),
    rectangleA_length > 0 ∧
    rectangleA_width > 0 ∧
    2 * (rectangleA_length + rectangleA_width) = rectangleA_perimeter ∧
    ∃ (squareB_side : ℝ),
      squareB_side > 0 ∧
      squareB_side^2 = squareB_area_ratio * (rectangleA_length * rectangleA_width) ∧
      4 * squareB_side = 12 * Real.sqrt 2 :=
by sorry

end square_perimeter_l1262_126250


namespace price_reduction_equation_correct_l1262_126282

/-- Represents the price reduction scenario -/
structure PriceReduction where
  initial_price : ℝ
  final_price : ℝ
  num_reductions : ℕ
  
/-- The price reduction equation is correct for the given scenario -/
theorem price_reduction_equation_correct (pr : PriceReduction) 
  (h1 : pr.initial_price = 560)
  (h2 : pr.final_price = 315)
  (h3 : pr.num_reductions = 2) :
  ∃ x : ℝ, pr.initial_price * (1 - x)^pr.num_reductions = pr.final_price :=
sorry

end price_reduction_equation_correct_l1262_126282


namespace expression_equality_l1262_126219

theorem expression_equality : 99^4 - 4 * 99^3 + 6 * 99^2 - 4 * 99 + 1 = 92199816 := by
  sorry

end expression_equality_l1262_126219


namespace cakes_served_yesterday_l1262_126218

/-- The number of cakes served during lunch today -/
def lunch_cakes : ℕ := 5

/-- The number of cakes served during dinner today -/
def dinner_cakes : ℕ := 6

/-- The total number of cakes served over two days -/
def total_cakes : ℕ := 14

/-- The number of cakes served yesterday -/
def yesterday_cakes : ℕ := total_cakes - (lunch_cakes + dinner_cakes)

theorem cakes_served_yesterday : yesterday_cakes = 3 := by
  sorry

end cakes_served_yesterday_l1262_126218


namespace eulers_formula_simply_connected_l1262_126217

/-- A simply connected polyhedron -/
structure SimplyConnectedPolyhedron where
  faces : ℕ
  edges : ℕ
  vertices : ℕ
  is_simply_connected : Bool

/-- Euler's formula for simply connected polyhedra -/
theorem eulers_formula_simply_connected (p : SimplyConnectedPolyhedron) 
  (h : p.is_simply_connected = true) : 
  p.faces - p.edges + p.vertices = 2 := by
  sorry

end eulers_formula_simply_connected_l1262_126217


namespace longest_interval_l1262_126256

-- Define the conversion factors
def minutes_per_hour : ℕ := 60
def hours_per_day : ℕ := 24

-- Define the time intervals
def interval_a : ℕ := 1500  -- in minutes
def interval_b : ℕ := 10    -- in hours
def interval_c : ℕ := 1     -- in days

-- Theorem to prove
theorem longest_interval :
  (interval_a : ℝ) > (interval_b * minutes_per_hour : ℝ) ∧
  (interval_a : ℝ) > (interval_c * hours_per_day * minutes_per_hour : ℝ) :=
by sorry

end longest_interval_l1262_126256


namespace christopher_karen_difference_l1262_126253

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 0.25

/-- The number of quarters Karen has -/
def karen_quarters : ℕ := 32

/-- The number of quarters Christopher has -/
def christopher_quarters : ℕ := 64

/-- The difference in money between Christopher and Karen -/
def money_difference : ℚ := (christopher_quarters - karen_quarters) * quarter_value

theorem christopher_karen_difference : money_difference = 8 := by
  sorry

end christopher_karen_difference_l1262_126253


namespace flow_across_cut_equals_flow_from_single_vertex_l1262_126208

-- Define a network
variable (N : Type*) [Fintype N]

-- Define the flow function
variable (f : Set N → Set N → ℝ)

-- Define the set of all vertices
variable (V : Set N)

-- Theorem statement
theorem flow_across_cut_equals_flow_from_single_vertex
  (S : Set N) (s : N) (h_s_in_S : s ∈ S) (h_S_subset_V : S ⊆ V) :
  f S (V \ S) = f {s} V :=
sorry

end flow_across_cut_equals_flow_from_single_vertex_l1262_126208


namespace concrete_wall_width_l1262_126281

theorem concrete_wall_width
  (r : ℝ)  -- radius of the pool
  (w : ℝ)  -- width of the concrete wall
  (h1 : r = 20)  -- radius of the pool is 20 ft
  (h2 : π * ((r + w)^2 - r^2) = (11/25) * (π * r^2))  -- area of wall is 11/25 of pool area
  : w = 4 := by
  sorry

end concrete_wall_width_l1262_126281


namespace range_of_m_l1262_126280

-- Define the conditions
def p (x : ℝ) : Prop := |1 - (x - 1) / 3| ≤ 2
def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

-- State the theorem
theorem range_of_m (m : ℝ) :
  (∀ x, p x → q x m) ∧  -- q is necessary for p
  (∃ x, p x ∧ ¬(q x m)) ∧  -- q is not sufficient for p
  (m > 0) →
  m ≥ 9 :=
by sorry

end range_of_m_l1262_126280


namespace set_propositions_equivalence_l1262_126293

theorem set_propositions_equivalence (A B : Set α) :
  (((A ∪ B ≠ B) → (A ∩ B ≠ A)) ∧
   ((A ∩ B ≠ A) → (A ∪ B ≠ B)) ∧
   ((A ∪ B = B) → (A ∩ B = A)) ∧
   ((A ∩ B = A) → (A ∪ B = B))) := by
  sorry

end set_propositions_equivalence_l1262_126293


namespace cloth_length_problem_l1262_126230

theorem cloth_length_problem (initial_length : ℝ) 
  (h1 : initial_length > 32)
  (h2 : initial_length > 20) :
  (initial_length - 32) * 3 = initial_length - 20 →
  initial_length = 38 := by
  sorry

end cloth_length_problem_l1262_126230


namespace basketball_handshakes_l1262_126287

/-- Number of players in each team -/
def team_size : ℕ := 6

/-- Number of teams -/
def num_teams : ℕ := 2

/-- Number of referees -/
def num_referees : ℕ := 3

/-- Total number of players -/
def total_players : ℕ := team_size * num_teams

/-- Function to calculate the number of handshakes between two teams -/
def inter_team_handshakes : ℕ := team_size * team_size

/-- Function to calculate the number of handshakes within a team -/
def intra_team_handshakes : ℕ := team_size.choose 2

/-- Function to calculate the number of handshakes between players and referees -/
def player_referee_handshakes : ℕ := total_players * num_referees

/-- The total number of handshakes in the basketball game -/
def total_handshakes : ℕ := 
  inter_team_handshakes + 
  (intra_team_handshakes * num_teams) + 
  player_referee_handshakes

theorem basketball_handshakes : total_handshakes = 102 := by
  sorry

end basketball_handshakes_l1262_126287


namespace circumscribing_circle_diameter_l1262_126206

theorem circumscribing_circle_diameter (n : ℕ) (r : ℝ) :
  n = 8 ∧ r = 2 →
  let R := (2 * r) / (2 * Real.sin (π / n))
  2 * (R + r) = 2 * (4 / Real.sqrt (2 - Real.sqrt 2) + 2) := by sorry

end circumscribing_circle_diameter_l1262_126206


namespace sum_of_squares_149_l1262_126294

theorem sum_of_squares_149 : ∃ (a b c : ℕ), 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a + b + c = 21 ∧
  a^2 + b^2 + c^2 = 149 := by
  sorry

end sum_of_squares_149_l1262_126294


namespace patty_avoids_chores_for_ten_weeks_l1262_126213

/-- Represents the problem of Patty paying her siblings with cookies to do her chores. -/
structure CookieChoresProblem where
  money_available : ℕ  -- Amount of money Patty has in dollars
  pack_cost : ℕ  -- Cost of one pack of cookies in dollars
  cookies_per_pack : ℕ  -- Number of cookies in each pack
  cookies_per_chore : ℕ  -- Number of cookies given for each chore
  chores_per_week : ℕ  -- Number of chores each kid has per week

/-- Calculates the number of weeks Patty can avoid doing chores. -/
def weeks_without_chores (problem : CookieChoresProblem) : ℕ :=
  let packs_bought := problem.money_available / problem.pack_cost
  let total_cookies := packs_bought * problem.cookies_per_pack
  let cookies_per_week := problem.cookies_per_chore * problem.chores_per_week
  total_cookies / cookies_per_week

/-- Theorem stating that given the problem conditions, Patty can avoid doing chores for 10 weeks. -/
theorem patty_avoids_chores_for_ten_weeks (problem : CookieChoresProblem) 
  (h1 : problem.money_available = 15)
  (h2 : problem.pack_cost = 3)
  (h3 : problem.cookies_per_pack = 24)
  (h4 : problem.cookies_per_chore = 3)
  (h5 : problem.chores_per_week = 4) :
  weeks_without_chores problem = 10 := by
  sorry

#eval weeks_without_chores { 
  money_available := 15, 
  pack_cost := 3, 
  cookies_per_pack := 24, 
  cookies_per_chore := 3, 
  chores_per_week := 4 
}

end patty_avoids_chores_for_ten_weeks_l1262_126213


namespace choose_four_from_nine_l1262_126267

theorem choose_four_from_nine : Nat.choose 9 4 = 126 := by
  sorry

end choose_four_from_nine_l1262_126267


namespace sqrt_16_equals_4_l1262_126266

theorem sqrt_16_equals_4 : Real.sqrt 16 = 4 := by
  sorry

end sqrt_16_equals_4_l1262_126266


namespace lecture_distribution_l1262_126204

def total_lecture_time : ℕ := 480
def max_disc_capacity : ℕ := 70

theorem lecture_distribution :
  ∃ (num_discs : ℕ) (minutes_per_disc : ℕ),
    num_discs > 0 ∧
    minutes_per_disc > 0 ∧
    minutes_per_disc ≤ max_disc_capacity ∧
    num_discs * minutes_per_disc = total_lecture_time ∧
    (∀ n : ℕ, n > 0 → n * max_disc_capacity < total_lecture_time → n < num_discs) ∧
    minutes_per_disc = 68 := by
  sorry

end lecture_distribution_l1262_126204


namespace geometric_sequence_general_term_l1262_126299

/-- Given a geometric sequence {a_n} with common ratio q = 4 and S_3 = 21,
    prove that the general term formula is a_n = 4^(n-1) -/
theorem geometric_sequence_general_term 
  (a : ℕ → ℝ) -- The sequence
  (q : ℝ) -- Common ratio
  (S₃ : ℝ) -- Sum of first 3 terms
  (h1 : ∀ n, a (n + 1) = q * a n) -- Definition of geometric sequence
  (h2 : q = 4) -- Given common ratio
  (h3 : S₃ = 21) -- Given sum of first 3 terms
  (h4 : S₃ = a 1 + a 2 + a 3) -- Definition of S₃
  : ∀ n : ℕ, a n = 4^(n - 1) := by
  sorry

end geometric_sequence_general_term_l1262_126299


namespace red_ball_probability_l1262_126234

theorem red_ball_probability (x : ℕ) : 
  (8 : ℝ) / (x + 8 : ℝ) = 0.2 → x = 32 := by
sorry

end red_ball_probability_l1262_126234


namespace last_three_average_l1262_126297

theorem last_three_average (list : List ℝ) : 
  list.length = 6 →
  list.sum / list.length = 60 →
  (list.take 3).sum / 3 = 55 →
  (list.drop 3).sum / 3 = 65 := by
sorry

end last_three_average_l1262_126297


namespace second_month_sale_is_6927_l1262_126214

/-- Calculates the sale amount for the second month given the sales of other months and the average sale --/
def calculate_second_month_sale (first_month : ℕ) (third_month : ℕ) (fourth_month : ℕ) (fifth_month : ℕ) (sixth_month : ℕ) (average_sale : ℕ) : ℕ :=
  6 * average_sale - (first_month + third_month + fourth_month + fifth_month + sixth_month)

/-- Theorem stating that the sale in the second month is 6927 given the problem conditions --/
theorem second_month_sale_is_6927 :
  calculate_second_month_sale 6435 6855 7230 6562 6191 6700 = 6927 := by
  sorry

end second_month_sale_is_6927_l1262_126214


namespace simplify_radical_expression_l1262_126289

theorem simplify_radical_expression :
  (Real.sqrt 6 + 4 * Real.sqrt 3 + 3 * Real.sqrt 2) / 
  ((Real.sqrt 6 + Real.sqrt 3) * (Real.sqrt 3 + Real.sqrt 2)) = 
  Real.sqrt 6 - Real.sqrt 2 := by sorry

end simplify_radical_expression_l1262_126289


namespace arithmetic_sequence_property_l1262_126244

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_sum : a 1 + 3 * a 8 + a 15 = 120) :
  3 * a 9 - a 11 = 48 := by
  sorry

end arithmetic_sequence_property_l1262_126244


namespace continued_fraction_sum_l1262_126212

theorem continued_fraction_sum (w x y : ℕ+) :
  (97 : ℚ) / 19 = w + 1 / (x + 1 / y) →
  (w : ℕ) + x + y = 16 := by
  sorry

end continued_fraction_sum_l1262_126212


namespace scavenger_hunt_items_l1262_126274

theorem scavenger_hunt_items (tanya samantha lewis james : ℕ) : 
  tanya = 4 ∧ 
  samantha = 4 * tanya ∧ 
  lewis = samantha + 4 ∧ 
  james = 2 * lewis →
  lewis = 20 := by
sorry

end scavenger_hunt_items_l1262_126274


namespace inequality_system_solution_range_l1262_126216

theorem inequality_system_solution_range (x m : ℝ) : 
  ((x + 1) / 2 < x / 3 + 1 ∧ x > 3 * m) → m < 1 := by
  sorry

end inequality_system_solution_range_l1262_126216


namespace solution_set_part1_range_of_a_part2_l1262_126203

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := 2 * |x + 2| - |a * x|

-- Part 1
theorem solution_set_part1 : 
  {x : ℝ | f 2 x > 2} = {x : ℝ | x > -1/2} := by sorry

-- Part 2
theorem range_of_a_part2 :
  (∀ x ∈ Set.Ioo (-1) 1, f a x > x + 1) ↔ a ∈ Set.Ioo (-2) 2 := by sorry

end solution_set_part1_range_of_a_part2_l1262_126203


namespace sum_of_coordinates_P_l1262_126233

/-- Given three points P, Q, and R in a plane such that PR/PQ = RQ/PQ = 1/2,
    Q = (2, 5), and R = (0, -10), prove that the sum of coordinates of P is -27. -/
theorem sum_of_coordinates_P (P Q R : ℝ × ℝ) : 
  (dist P R / dist P Q = 1/2) →
  (dist R Q / dist P Q = 1/2) →
  Q = (2, 5) →
  R = (0, -10) →
  P.1 + P.2 = -27 := by
  sorry

#check sum_of_coordinates_P

end sum_of_coordinates_P_l1262_126233


namespace rotated_region_volume_is_19pi_l1262_126238

/-- The volume of a solid formed by rotating a region about the y-axis. The region consists of:
    1. A vertical strip of 7 unit squares high and 1 unit wide along the y-axis.
    2. A horizontal strip of 3 unit squares wide and 2 units high along the x-axis, 
       starting from the top of the vertical strip. -/
def rotated_region_volume : ℝ := sorry

/-- The theorem states that the volume of the rotated region is equal to 19π cubic units. -/
theorem rotated_region_volume_is_19pi : rotated_region_volume = 19 * Real.pi := by sorry

end rotated_region_volume_is_19pi_l1262_126238


namespace lost_card_value_l1262_126283

theorem lost_card_value (n : ℕ) (h1 : n > 0) (h2 : (n * (n + 1)) / 2 - 101 ≤ n) : 
  (n * (n + 1)) / 2 - 101 = 4 :=
by sorry

end lost_card_value_l1262_126283


namespace bowling_ball_weight_l1262_126263

theorem bowling_ball_weight :
  ∀ (bowling_ball_weight canoe_weight : ℝ),
    (10 * bowling_ball_weight = 5 * canoe_weight) →
    (3 * canoe_weight = 120) →
    bowling_ball_weight = 20 := by
  sorry

end bowling_ball_weight_l1262_126263


namespace initial_amount_proof_l1262_126222

/-- Proves that given specific conditions, the initial amount of money is 30000 --/
theorem initial_amount_proof (rate : ℝ) (time : ℝ) (difference : ℝ) : 
  rate = 0.20 →
  time = 2 →
  difference = 723.0000000000146 →
  (fun P : ℝ => P * ((1 + rate / 2) ^ (2 * time) - (1 + rate) ^ time)) difference = 30000 :=
by sorry

end initial_amount_proof_l1262_126222


namespace least_integer_absolute_value_l1262_126224

theorem least_integer_absolute_value (x : ℤ) :
  (∀ y : ℤ, y < x → ∃ z : ℤ, z ≥ y ∧ z < x ∧ |3 * z^2 - 2 * z + 5| > 29) →
  |3 * x^2 - 2 * x + 5| ≤ 29 →
  x = -2 :=
sorry

end least_integer_absolute_value_l1262_126224


namespace initial_machines_count_l1262_126284

/-- The number of pens produced by a group of machines in a given time -/
structure PenProduction where
  machines : ℕ
  pens : ℕ
  minutes : ℕ

/-- The rate of pen production per minute for a given number of machines -/
def production_rate (p : PenProduction) : ℚ :=
  p.pens / (p.machines * p.minutes)

theorem initial_machines_count (total_rate : ℕ) (sample : PenProduction) :
  sample.machines * total_rate = sample.pens * production_rate sample →
  total_rate = 240 →
  sample = { machines := 5, pens := 750, minutes := 5 } →
  ∃ n : ℕ, n * (production_rate sample) = total_rate ∧ n = 8 := by
sorry

end initial_machines_count_l1262_126284


namespace journey_proportions_l1262_126210

theorem journey_proportions 
  (total_distance : ℝ) 
  (rail_proportion bus_proportion : ℝ) 
  (h1 : rail_proportion > 0)
  (h2 : bus_proportion > 0)
  (h3 : rail_proportion + bus_proportion < 1) :
  ∃ (foot_proportion : ℝ),
    foot_proportion > 0 ∧ 
    rail_proportion + bus_proportion + foot_proportion = 1 := by
  sorry

end journey_proportions_l1262_126210


namespace limit_sequence_equals_one_over_e_l1262_126286

theorem limit_sequence_equals_one_over_e :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N,
    |((10 * n - 3) / (10 * n - 1)) ^ (5 * n) - (1 / Real.exp 1)| < ε :=
sorry

end limit_sequence_equals_one_over_e_l1262_126286


namespace range_of_a_l1262_126247

theorem range_of_a (a x : ℝ) : 
  (∀ x, (a - 4 < x ∧ x < a + 4) → (x - 2) * (x - 3) > 0) →
  (a ≤ -2 ∨ a ≥ 7) := by
  sorry

end range_of_a_l1262_126247


namespace binomial_sum_formula_l1262_126227

def binomial_sum (n : ℕ) : ℕ :=
  Finset.sum (Finset.range n) (fun k => (k + 1) * (k + 2) * Nat.choose n (k + 1))

theorem binomial_sum_formula (n : ℕ) (h : n ≥ 4) :
  binomial_sum n = n * (n + 3) * 2^(n - 2) :=
by sorry

end binomial_sum_formula_l1262_126227


namespace surfers_count_l1262_126239

/-- The number of surfers on Santa Monica beach -/
def santa_monica_surfers : ℕ := 20

/-- The number of surfers on Malibu beach -/
def malibu_surfers : ℕ := 2 * santa_monica_surfers

/-- The total number of surfers on both beaches -/
def total_surfers : ℕ := malibu_surfers + santa_monica_surfers

theorem surfers_count : total_surfers = 60 := by
  sorry

end surfers_count_l1262_126239


namespace quilt_patch_cost_is_450_l1262_126258

/-- Calculates the total cost of patches for a quilt with given dimensions and patch pricing. -/
def quilt_patch_cost (quilt_length : ℕ) (quilt_width : ℕ) (patch_area : ℕ) 
                     (initial_patch_cost : ℕ) (initial_patch_count : ℕ) : ℕ :=
  let total_area := quilt_length * quilt_width
  let total_patches := total_area / patch_area
  let initial_cost := initial_patch_count * initial_patch_cost
  let remaining_patches := total_patches - initial_patch_count
  let remaining_cost := remaining_patches * (initial_patch_cost / 2)
  initial_cost + remaining_cost

/-- The total cost of patches for a 16-foot by 20-foot quilt with specified patch pricing is $450. -/
theorem quilt_patch_cost_is_450 : 
  quilt_patch_cost 16 20 4 10 10 = 450 := by
  sorry

end quilt_patch_cost_is_450_l1262_126258


namespace room_length_l1262_126226

/-- The length of a room given its width, total paving cost, and paving rate per square meter. -/
theorem room_length (width : ℝ) (total_cost : ℝ) (rate_per_sqm : ℝ) 
  (h_width : width = 3.75)
  (h_total_cost : total_cost = 12375)
  (h_rate : rate_per_sqm = 600) : 
  total_cost / rate_per_sqm / width = 5.5 := by
sorry

#eval (12375 / 600 / 3.75 : Float)

end room_length_l1262_126226


namespace average_equation_solution_l1262_126259

theorem average_equation_solution (x : ℝ) : 
  (1/3 : ℝ) * ((x + 8) + (5*x + 3) + (3*x + 4)) = 4*x + 1 → x = 4 := by
sorry

end average_equation_solution_l1262_126259


namespace not_perp_planes_implies_no_perp_line_l1262_126248

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between planes
variable (perp_planes : Plane → Plane → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perp_line_plane : Line → Plane → Prop)

-- Define the "line within plane" relation
variable (line_in_plane : Line → Plane → Prop)

-- Theorem statement
theorem not_perp_planes_implies_no_perp_line (α β : Plane) :
  ¬(∀ (α β : Plane), ¬(perp_planes α β) → ∀ (l : Line), line_in_plane l α → ¬(perp_line_plane l β)) :=
sorry

end not_perp_planes_implies_no_perp_line_l1262_126248


namespace quadratic_solution_property_l1262_126273

theorem quadratic_solution_property (k : ℝ) : 
  (∃ a b : ℝ, 6 * a^2 + 5 * a + k = 0 ∧ 
              6 * b^2 + 5 * b + k = 0 ∧ 
              a ≠ b ∧
              |a - b| = 3 * (a^2 + b^2)) ↔ 
  (k = 1 ∨ k = -17900 / 864) :=
sorry

end quadratic_solution_property_l1262_126273


namespace reciprocal_of_negative_five_l1262_126211

theorem reciprocal_of_negative_five :
  ∃ x : ℚ, x * (-5) = 1 ∧ x = -1/5 := by
  sorry

end reciprocal_of_negative_five_l1262_126211


namespace bridge_length_l1262_126205

/-- The length of a bridge given train parameters and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 130 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  (train_speed_kmh * 1000 / 3600) * crossing_time - train_length = 245 := by
  sorry

end bridge_length_l1262_126205


namespace book_survey_difference_l1262_126242

/-- Represents the survey results of students reading books A and B -/
structure BookSurvey where
  total : ℕ
  only_a : ℕ
  only_b : ℕ
  both : ℕ
  h_total : total = only_a + only_b + both
  h_a_both : both = (only_a + both) / 5
  h_b_both : both = (only_b + both) / 4

/-- The difference between students who read only book A and only book B is 75 -/
theorem book_survey_difference (s : BookSurvey) (h_total : s.total = 600) :
  s.only_a - s.only_b = 75 := by
  sorry

end book_survey_difference_l1262_126242


namespace hcf_of_4_and_18_l1262_126268

theorem hcf_of_4_and_18 :
  let a : ℕ := 4
  let b : ℕ := 18
  let lcm_ab : ℕ := 36
  Nat.lcm a b = lcm_ab →
  Nat.gcd a b = 2 := by
sorry

end hcf_of_4_and_18_l1262_126268


namespace triangle_inequality_cosine_law_l1262_126275

theorem triangle_inequality_cosine_law (x y z α β γ : ℝ) 
  (h_positive : x > 0 ∧ y > 0 ∧ z > 0)
  (h_angle_range : 0 ≤ α ∧ α < π ∧ 0 ≤ β ∧ β < π ∧ 0 ≤ γ ∧ γ < π)
  (h_angle_sum : α + β > γ ∧ β + γ > α ∧ γ + α > β) :
  Real.sqrt (x^2 + y^2 - 2*x*y*(Real.cos α)) + Real.sqrt (y^2 + z^2 - 2*y*z*(Real.cos β)) 
  ≥ Real.sqrt (z^2 + x^2 - 2*z*x*(Real.cos γ)) := by
sorry

end triangle_inequality_cosine_law_l1262_126275


namespace cubic_polynomial_third_root_l1262_126232

theorem cubic_polynomial_third_root 
  (a b : ℚ) 
  (h1 : a * 1^3 + (a + 3*b) * 1^2 + (b - 4*a) * 1 + (6 - a) = 0)
  (h2 : a * (-3)^3 + (a + 3*b) * (-3)^2 + (b - 4*a) * (-3) + (6 - a) = 0) :
  ∃ (x : ℚ), x = 7/13 ∧ a * x^3 + (a + 3*b) * x^2 + (b - 4*a) * x + (6 - a) = 0 := by
sorry

end cubic_polynomial_third_root_l1262_126232


namespace trouser_original_price_l1262_126264

theorem trouser_original_price (sale_price : ℝ) (discount_percent : ℝ) : 
  sale_price = 55 → discount_percent = 45 → 
  ∃ (original_price : ℝ), original_price = 100 ∧ sale_price = original_price * (1 - discount_percent / 100) :=
by
  sorry

end trouser_original_price_l1262_126264


namespace ab_geq_4_and_a_plus_b_geq_4_relationship_l1262_126240

theorem ab_geq_4_and_a_plus_b_geq_4_relationship (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ a b, a > 0 → b > 0 → a * b ≥ 4 → a + b ≥ 4) ∧
  (∃ a b, a > 0 ∧ b > 0 ∧ a + b ≥ 4 ∧ a * b < 4) := by
  sorry

end ab_geq_4_and_a_plus_b_geq_4_relationship_l1262_126240


namespace quadratic_inequality_min_value_l1262_126221

/-- Given a quadratic inequality with an empty solution set and a condition on its coefficients,
    prove that a certain expression has a minimum value of 4. -/
theorem quadratic_inequality_min_value (a b c : ℝ) :
  (∀ x, (1/a) * x^2 + b*x + c ≥ 0) →
  a * b > 1 →
  ∀ T, T = 1/(2*(a*b - 1)) + (a*(b + 2*c))/(a*b - 1) →
  T ≥ 4 :=
sorry

end quadratic_inequality_min_value_l1262_126221


namespace newspaper_cost_8_weeks_l1262_126288

/-- The cost of newspapers over a period of weeks -/
def newspaper_cost (weekday_price : ℚ) (sunday_price : ℚ) (num_weeks : ℕ) : ℚ :=
  (3 * weekday_price + sunday_price) * num_weeks

/-- Proof that the total cost of newspapers for 8 weeks is $28.00 -/
theorem newspaper_cost_8_weeks :
  newspaper_cost 0.5 2 8 = 28 := by
  sorry

end newspaper_cost_8_weeks_l1262_126288


namespace cody_lost_tickets_l1262_126272

theorem cody_lost_tickets (initial : Real) (spent : Real) (left : Real) : 
  initial = 49.0 → spent = 25.0 → left = 18 → initial - spent - left = 6.0 := by
  sorry

end cody_lost_tickets_l1262_126272


namespace ellipse_range_l1262_126228

theorem ellipse_range (x y : ℝ) (h : x^2/4 + y^2 = 1) :
  ∃ (z : ℝ), z = 2*x + y ∧ -Real.sqrt 17 ≤ z ∧ z ≤ Real.sqrt 17 :=
by sorry

end ellipse_range_l1262_126228


namespace max_value_expression_max_value_achievable_l1262_126292

theorem max_value_expression (x : ℝ) (hx : x > 0) :
  (x^2 + 3 - Real.sqrt (x^4 + 9)) / x ≤ 3 - Real.sqrt 6 :=
by sorry

theorem max_value_achievable :
  ∃ x : ℝ, x > 0 ∧ (x^2 + 3 - Real.sqrt (x^4 + 9)) / x = 3 - Real.sqrt 6 :=
by sorry

end max_value_expression_max_value_achievable_l1262_126292


namespace f_increasing_on_interval_l1262_126295

/-- The function f(x) = ax^2 - 2x + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2 * x + 2

/-- The function F(x) = |f(x)| -/
def F (a : ℝ) (x : ℝ) : ℝ := |f a x|

/-- The theorem statement -/
theorem f_increasing_on_interval (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 1 2 → x₂ ∈ Set.Icc 1 2 → x₁ ≠ x₂ →
    (F a x₁ - F a x₂) / (x₁ - x₂) > 0) →
  a ∈ Set.Iic 0 ∪ Set.Ici 1 :=
sorry

end f_increasing_on_interval_l1262_126295


namespace team_a_wins_l1262_126236

/-- Represents the outcome of a match for a team -/
inductive MatchResult
  | Win
  | Draw
  | Loss

/-- Calculates points for a given match result -/
def pointsForResult (result : MatchResult) : Nat :=
  match result with
  | MatchResult.Win => 3
  | MatchResult.Draw => 1
  | MatchResult.Loss => 0

/-- Represents the results of a series of matches for a team -/
structure TeamResults where
  wins : Nat
  draws : Nat
  losses : Nat

/-- Calculates total points for a team's results -/
def totalPoints (results : TeamResults) : Nat :=
  results.wins * (pointsForResult MatchResult.Win) +
  results.draws * (pointsForResult MatchResult.Draw) +
  results.losses * (pointsForResult MatchResult.Loss)

theorem team_a_wins (total_matches : Nat) (team_a_points : Nat)
    (h1 : total_matches = 10)
    (h2 : team_a_points = 22)
    (h3 : ∀ (r : TeamResults), 
      r.wins + r.draws = total_matches → 
      r.losses = 0 → 
      totalPoints r = team_a_points → 
      r.wins = 6) :
  ∃ (r : TeamResults), r.wins + r.draws = total_matches ∧ 
                       r.losses = 0 ∧ 
                       totalPoints r = team_a_points ∧ 
                       r.wins = 6 := by
  sorry

#check team_a_wins

end team_a_wins_l1262_126236


namespace plane_hit_probability_l1262_126265

theorem plane_hit_probability (p_A p_B : ℝ) (h_A : p_A = 0.3) (h_B : p_B = 0.5) :
  1 - (1 - p_A) * (1 - p_B) = 0.65 := by
  sorry

end plane_hit_probability_l1262_126265


namespace complement_intersection_theorem_l1262_126255

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def M : Set Nat := {2, 3}
def N : Set Nat := {1, 4}

theorem complement_intersection_theorem :
  (U \ M) ∩ (U \ N) = {5, 6} := by sorry

end complement_intersection_theorem_l1262_126255


namespace brown_eyes_ratio_l1262_126271

/-- Represents the number of people with different eye colors in a theater. -/
structure TheaterEyeColors where
  total : ℕ
  blue : ℕ
  black : ℕ
  green : ℕ
  brown : ℕ

/-- Theorem stating the ratio of people with brown eyes to total people in the theater. -/
theorem brown_eyes_ratio (t : TheaterEyeColors) :
  t.total = 100 ∧ 
  t.blue = 19 ∧ 
  t.black = t.total / 4 ∧ 
  t.green = 6 ∧ 
  t.brown = t.total - (t.blue + t.black + t.green) →
  2 * t.brown = t.total := by
  sorry

#check brown_eyes_ratio

end brown_eyes_ratio_l1262_126271


namespace multiply_and_add_l1262_126246

theorem multiply_and_add : (23 * 37) + 16 = 867 := by
  sorry

end multiply_and_add_l1262_126246


namespace rohan_salary_l1262_126296

def monthly_salary (food_percent : ℚ) (rent_percent : ℚ) (entertainment_percent : ℚ) (conveyance_percent : ℚ) (savings : ℕ) : ℕ :=
  sorry

theorem rohan_salary :
  let food_percent : ℚ := 40 / 100
  let rent_percent : ℚ := 20 / 100
  let entertainment_percent : ℚ := 10 / 100
  let conveyance_percent : ℚ := 10 / 100
  let savings : ℕ := 1000
  monthly_salary food_percent rent_percent entertainment_percent conveyance_percent savings = 5000 := by
  sorry

end rohan_salary_l1262_126296


namespace problems_per_page_problems_per_page_is_four_l1262_126277

theorem problems_per_page : ℕ → Prop :=
  fun p =>
    let math_pages : ℕ := 4
    let reading_pages : ℕ := 6
    let total_pages : ℕ := math_pages + reading_pages
    let total_problems : ℕ := 40
    total_pages * p = total_problems → p = 4

-- The proof is omitted
theorem problems_per_page_is_four : problems_per_page 4 := by sorry

end problems_per_page_problems_per_page_is_four_l1262_126277


namespace binary_sum_is_eleven_l1262_126252

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The first binary number 101₂ -/
def binary1 : List Bool := [true, false, true]

/-- The second binary number 110₂ -/
def binary2 : List Bool := [false, true, true]

/-- The sum of binary1 and binary2 in decimal form -/
def sum_decimal : ℕ := binary_to_decimal binary1 + binary_to_decimal binary2

theorem binary_sum_is_eleven : sum_decimal = 11 := by
  sorry

end binary_sum_is_eleven_l1262_126252


namespace min_chords_for_circle_l1262_126229

/-- Given a circle with chords subtending a central angle of 120°, 
    prove that the minimum number of such chords to complete the circle is 3. -/
theorem min_chords_for_circle (n : ℕ) : n > 0 → (120 * n = 360 * m) → n ≥ 3 :=
sorry

end min_chords_for_circle_l1262_126229


namespace spam_price_theorem_l1262_126260

-- Define the constants from the problem
def peanut_butter_price : ℝ := 5
def bread_price : ℝ := 2
def spam_cans : ℕ := 12
def peanut_butter_jars : ℕ := 3
def bread_loaves : ℕ := 4
def total_paid : ℝ := 59

-- Define the theorem
theorem spam_price_theorem :
  ∃ (spam_price : ℝ),
    spam_price * spam_cans +
    peanut_butter_price * peanut_butter_jars +
    bread_price * bread_loaves = total_paid ∧
    spam_price = 3 :=
by sorry

end spam_price_theorem_l1262_126260


namespace value_of_expression_l1262_126207

theorem value_of_expression (a b : ℝ) (h : a - b = 1) : 3*a - 3*b - 4 = -1 := by
  sorry

end value_of_expression_l1262_126207


namespace no_valid_polygon_pairs_l1262_126290

theorem no_valid_polygon_pairs : ¬∃ (y l : ℕ), 
  (∃ (k : ℕ), y = 30 * k) ∧ 
  (l > 1) ∧
  (∃ (n : ℕ), y = 180 - 360 / n) ∧
  (∃ (m : ℕ), l * y = 180 - 360 / m) := by
  sorry

end no_valid_polygon_pairs_l1262_126290


namespace number_puzzle_l1262_126245

theorem number_puzzle (x : ℝ) : 3 * (2 * x + 9) = 51 → x = 4 := by
  sorry

end number_puzzle_l1262_126245


namespace board_zero_condition_l1262_126225

/-- Represents a board with positive integers -/
def Board (m n : ℕ) := Fin m → Fin n → ℕ+

/-- Checks if two positions are adjacent on the board -/
def adjacent (m n : ℕ) (x1 y1 x2 y2 : ℕ) : Prop :=
  (x1 = x2 ∧ (y1 = y2 + 1 ∨ y2 = y1 + 1)) ∨
  (y1 = y2 ∧ (x1 = x2 + 1 ∨ x2 = x1 + 1))

/-- Represents a move on the board -/
structure Move (m n : ℕ) where
  x1 : Fin m
  y1 : Fin n
  x2 : Fin m
  y2 : Fin n
  k : ℤ
  adj : adjacent m n x1.val y1.val x2.val y2.val

/-- Applies a move to the board -/
def applyMove (b : Board m n) (move : Move m n) : Board m n :=
  sorry

/-- Checks if a position is on a black square in chessboard coloring -/
def isBlack (x y : ℕ) : Bool :=
  (x + y) % 2 = 0

/-- Calculates the sum of numbers on black squares -/
def sumBlack (b : Board m n) : ℕ :=
  sorry

/-- Calculates the sum of numbers on white squares -/
def sumWhite (b : Board m n) : ℕ :=
  sorry

/-- Represents a sequence of moves -/
def MoveSequence (m n : ℕ) := List (Move m n)

/-- Applies a sequence of moves to the board -/
def applyMoveSequence (b : Board m n) (moves : MoveSequence m n) : Board m n :=
  sorry

/-- Checks if all numbers on the board are zero -/
def allZero (b : Board m n) : Prop :=
  ∀ x y, (b x y : ℕ) = 0

theorem board_zero_condition (m n : ℕ) :
  ∀ (b : Board m n),
    (∃ (moves : MoveSequence m n), allZero (applyMoveSequence b moves)) ↔
    (sumBlack b = sumWhite b) :=
  sorry

end board_zero_condition_l1262_126225


namespace shift_arrangements_count_l1262_126235

def total_volunteers : ℕ := 14
def shifts_per_day : ℕ := 3
def people_per_shift : ℕ := 4

def shift_arrangements : ℕ := (total_volunteers.choose people_per_shift) * 
                               ((total_volunteers - people_per_shift).choose people_per_shift) * 
                               ((total_volunteers - 2 * people_per_shift).choose people_per_shift)

theorem shift_arrangements_count : shift_arrangements = 3153150 := by
  sorry

end shift_arrangements_count_l1262_126235


namespace average_of_special_squares_l1262_126262

/-- Represents a 4x4 grid filled with numbers 1, 3, 5, and 7 -/
def Grid := Fin 4 → Fin 4 → Fin 4

/-- Checks if a row contains different numbers -/
def row_valid (g : Grid) (i : Fin 4) : Prop :=
  ∀ j k : Fin 4, j ≠ k → g i j ≠ g i k

/-- Checks if a column contains different numbers -/
def col_valid (g : Grid) (j : Fin 4) : Prop :=
  ∀ i k : Fin 4, i ≠ k → g i j ≠ g k j

/-- Checks if a 2x2 board contains different numbers -/
def board_valid (g : Grid) (i j : Fin 2) : Prop :=
  ∀ x y z w : Fin 2, (x, y) ≠ (z, w) → g (i + x) (j + y) ≠ g (i + z) (j + w)

/-- Checks if the entire grid is valid -/
def grid_valid (g : Grid) : Prop :=
  (∀ i : Fin 4, row_valid g i) ∧
  (∀ j : Fin 4, col_valid g j) ∧
  (∀ i j : Fin 2, board_valid g i j)

/-- The set of valid numbers in the grid -/
def valid_numbers : Finset (Fin 4) :=
  {0, 1, 2, 3}

/-- Maps Fin 4 to the actual numbers used in the grid -/
def to_actual_number (n : Fin 4) : ℕ :=
  2 * n + 1

/-- Theorem: The average of numbers in squares A, B, C, D is 4 -/
theorem average_of_special_squares (g : Grid) (hg : grid_valid g) :
  (to_actual_number (g 0 0) + to_actual_number (g 0 3) +
   to_actual_number (g 3 0) + to_actual_number (g 3 3)) / 4 = 4 := by
  sorry

end average_of_special_squares_l1262_126262


namespace area_of_triple_square_l1262_126209

/-- Given a square (square I) with diagonal length a + b√2, 
    prove that the area of a square (square II) that is three times 
    the area of square I is 3a^2 + 6ab√2 + 6b^2 -/
theorem area_of_triple_square (a b : ℝ) : 
  let diagonal_I := a + b * Real.sqrt 2
  let area_II := 3 * (diagonal_I^2 / 2)
  area_II = 3 * a^2 + 6 * a * b * Real.sqrt 2 + 6 * b^2 := by
  sorry

end area_of_triple_square_l1262_126209


namespace min_sides_convex_polygon_l1262_126270

/-- A convex polygon is a closed planar figure with straight sides. -/
structure ConvexPolygon where
  sides : ℕ
  is_convex : Bool

/-- Theorem: The minimum number of sides for a convex polygon is 3. -/
theorem min_sides_convex_polygon :
  ∀ p : ConvexPolygon, p.is_convex → p.sides ≥ 3 :=
by sorry

end min_sides_convex_polygon_l1262_126270


namespace investment_rate_problem_l1262_126276

/-- Proves that given the conditions of the investment problem, the rate of the first investment is 10% -/
theorem investment_rate_problem (total_investment : ℝ) (second_investment : ℝ) (second_rate : ℝ) (income_difference : ℝ) :
  total_investment = 2000 →
  second_investment = 750 →
  second_rate = 0.08 →
  income_difference = 65 →
  let first_investment := total_investment - second_investment
  let first_rate := (income_difference + second_investment * second_rate) / first_investment
  first_rate = 0.1 := by
  sorry

#check investment_rate_problem

end investment_rate_problem_l1262_126276


namespace common_divisor_nineteen_l1262_126261

theorem common_divisor_nineteen (a : ℤ) : Int.gcd (35 * a + 57) (45 * a + 76) = 19 := by
  sorry

end common_divisor_nineteen_l1262_126261
