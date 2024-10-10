import Mathlib

namespace equality_from_inequalities_l3419_341953

theorem equality_from_inequalities (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h_pos : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0 ∧ x₅ > 0)
  (h1 : (x₁^2 - x₃*x₅)*(x₂^2 - x₃*x₅) ≤ 0)
  (h2 : (x₂^2 - x₄*x₁)*(x₃^2 - x₄*x₁) ≤ 0)
  (h3 : (x₃^2 - x₅*x₂)*(x₄^2 - x₅*x₂) ≤ 0)
  (h4 : (x₄^2 - x₁*x₃)*(x₅^2 - x₁*x₃) ≤ 0)
  (h5 : (x₅^2 - x₂*x₄)*(x₁^2 - x₂*x₄) ≤ 0) :
  x₁ = x₂ ∧ x₂ = x₃ ∧ x₃ = x₄ ∧ x₄ = x₅ :=
by sorry

end equality_from_inequalities_l3419_341953


namespace probability_king_then_heart_l3419_341958

/- Define a standard deck of cards -/
def StandardDeck : ℕ := 52

/- Define the number of Kings in a standard deck -/
def NumberOfKings : ℕ := 4

/- Define the number of hearts in a standard deck -/
def NumberOfHearts : ℕ := 13

/- Theorem statement -/
theorem probability_king_then_heart (deck : ℕ) (kings : ℕ) (hearts : ℕ) 
  (h1 : deck = StandardDeck) 
  (h2 : kings = NumberOfKings) 
  (h3 : hearts = NumberOfHearts) : 
  (kings : ℚ) / deck * hearts / (deck - 1) = 1 / 52 := by
  sorry


end probability_king_then_heart_l3419_341958


namespace x_value_l3419_341983

-- Define the ⊕ operation
def oplus (a b : ℝ) : ℝ := 3 * a - b

-- State the theorem
theorem x_value : ∃ x : ℝ, oplus x (oplus 2 3) = 1 ∧ x = 4/3 := by
  sorry

end x_value_l3419_341983


namespace ellipse_and_line_theorem_l3419_341971

/-- An ellipse with given properties -/
structure Ellipse where
  center : ℝ × ℝ
  foci_on_x_axis : Bool
  eccentricity : ℝ
  passes_through : ℝ × ℝ

/-- A line passing through a given point -/
structure Line where
  point : ℝ × ℝ
  slope : ℝ

/-- The dot product of two 2D vectors -/
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem ellipse_and_line_theorem (C : Ellipse) (l : Line) : 
  C.center = (0, 0) ∧ 
  C.foci_on_x_axis = true ∧ 
  C.eccentricity = 1/2 ∧ 
  C.passes_through = (1, 3/2) ∧
  l.point = (2, 1) →
  (∃ (A B : ℝ × ℝ), 
    -- C has equation x^2/4 + y^2/3 = 1
    (A.1^2/4 + A.2^2/3 = 1 ∧ B.1^2/4 + B.2^2/3 = 1) ∧
    -- A and B are on line l
    (A.2 - l.point.2 = l.slope * (A.1 - l.point.1) ∧ 
     B.2 - l.point.2 = l.slope * (B.1 - l.point.1)) ∧
    -- A and B are distinct
    A ≠ B ∧
    -- PA · PB = PM^2
    dot_product (A.1 - l.point.1, A.2 - l.point.2) (B.1 - l.point.1, B.2 - l.point.2) = 
    dot_product (1 - 2, 3/2 - 1) (1 - 2, 3/2 - 1) ∧
    -- l has equation y = (1/2)x
    l.slope = 1/2) :=
sorry

end ellipse_and_line_theorem_l3419_341971


namespace ratio_q_p_l3419_341907

/-- The number of cards in the box -/
def total_cards : ℕ := 60

/-- The number of distinct numbers on the cards -/
def distinct_numbers : ℕ := 15

/-- The number of cards for each number -/
def cards_per_number : ℕ := 4

/-- The number of cards drawn -/
def cards_drawn : ℕ := 4

/-- The probability of drawing four cards with the same number -/
def p' : ℚ := (distinct_numbers * 1) / Nat.choose total_cards cards_drawn

/-- The probability of drawing three cards with one number and one card with a different number -/
def q' : ℚ := (distinct_numbers * (distinct_numbers - 1) * Nat.choose cards_per_number 3 * Nat.choose cards_per_number 1) / Nat.choose total_cards cards_drawn

/-- The main theorem stating the ratio of q' to p' -/
theorem ratio_q_p : q' / p' = 224 := by sorry

end ratio_q_p_l3419_341907


namespace exists_n_not_perfect_square_l3419_341995

theorem exists_n_not_perfect_square : ∃ n : ℕ, n > 1 ∧ ¬ ∃ m : ℕ, 2^(2^n - 1) - 7 = m^2 := by
  sorry

end exists_n_not_perfect_square_l3419_341995


namespace min_value_sum_of_fractions_l3419_341956

theorem min_value_sum_of_fractions (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + y + z = 3) : 
  (4 / x) + (9 / y) + (16 / z) ≥ 27 := by
  sorry

end min_value_sum_of_fractions_l3419_341956


namespace cubic_foot_to_cubic_inches_l3419_341947

theorem cubic_foot_to_cubic_inches :
  ∀ (foot inch : ℝ), 
    foot > 0 →
    inch > 0 →
    foot = 12 * inch →
    foot^3 = 1728 * inch^3 :=
by sorry

end cubic_foot_to_cubic_inches_l3419_341947


namespace points_on_line_equidistant_from_axes_l3419_341905

theorem points_on_line_equidistant_from_axes :
  ∃ (x y : ℝ), 4 * x - 3 * y = 24 ∧ |x| = |y| ∧
  ((x > 0 ∧ y > 0) ∨ (x > 0 ∧ y < 0)) :=
by sorry

end points_on_line_equidistant_from_axes_l3419_341905


namespace complex_equation_solution_l3419_341916

theorem complex_equation_solution (a b : ℝ) : 
  (1 - Complex.I) * (a + 2 * Complex.I) = b * Complex.I → b = 4 := by
  sorry

end complex_equation_solution_l3419_341916


namespace final_number_lower_bound_l3419_341948

/-- Represents a sequence of operations on the blackboard -/
def BlackboardOperation := List (Nat × Nat)

/-- The result of applying a sequence of operations to the initial numbers -/
def applyOperations (n : Nat) (ops : BlackboardOperation) : Nat :=
  sorry

/-- Theorem: The final number after any sequence of operations is at least 4/9 * n^3 -/
theorem final_number_lower_bound (n : Nat) (ops : BlackboardOperation) :
  applyOperations n ops ≥ (4 * n^3) / 9 := by
  sorry

end final_number_lower_bound_l3419_341948


namespace sin_300_degrees_l3419_341920

theorem sin_300_degrees : Real.sin (300 * π / 180) = -Real.sqrt 3 / 2 := by
  -- Define the properties of sine function
  have sin_periodic : ∀ θ k, Real.sin θ = Real.sin (θ + k * 2 * π) := by sorry
  have sin_symmetry : ∀ θ, Real.sin θ = Real.sin (-θ) := by sorry
  have sin_odd : ∀ θ, Real.sin (-θ) = -Real.sin θ := by sorry
  have sin_60_degrees : Real.sin (60 * π / 180) = Real.sqrt 3 / 2 := by sorry

  -- Proof steps would go here, but we're skipping them as per instructions
  sorry

end sin_300_degrees_l3419_341920


namespace ellipse_interfocal_distance_l3419_341900

/-- An ellipse with given latus rectum and focus-to-vertex distance has a specific interfocal distance -/
theorem ellipse_interfocal_distance 
  (latus_rectum : ℝ) 
  (focus_to_vertex : ℝ) 
  (h1 : latus_rectum = 5.4)
  (h2 : focus_to_vertex = 1.5) :
  ∃ (a b c : ℝ),
    a^2 = b^2 + c^2 ∧
    a - c = focus_to_vertex ∧
    b = latus_rectum / 2 ∧
    2 * c = 12 := by
  sorry

end ellipse_interfocal_distance_l3419_341900


namespace digit_difference_in_base_d_l3419_341928

/-- Given two digits X and Y in base d > 8, prove that X - Y = -1 in base d
    when XY + XX = 234 in base d. -/
theorem digit_difference_in_base_d (d : ℕ) (X Y : ℕ) : d > 8 →
  X < d → Y < d →
  (X * d + Y) + (X * d + X) = 2 * d * d + 3 * d + 4 →
  X - Y = d - 1 := by
  sorry

end digit_difference_in_base_d_l3419_341928


namespace domino_trick_l3419_341986

theorem domino_trick (x y : ℕ) (hx : x ≤ 6) (hy : y ≤ 6) :
  10 * x + y + 30 = 62 → x = 3 ∧ y = 2 := by
  sorry

end domino_trick_l3419_341986


namespace more_girls_than_boys_l3419_341911

/-- Given a school with a boy-to-girl ratio of 5:13 and 50 boys, prove that there are 80 more girls than boys. -/
theorem more_girls_than_boys (num_boys : ℕ) (ratio_boys : ℕ) (ratio_girls : ℕ) : 
  num_boys = 50 →
  ratio_boys = 5 →
  ratio_girls = 13 →
  (ratio_girls * num_boys / ratio_boys) - num_boys = 80 :=
by sorry

end more_girls_than_boys_l3419_341911


namespace integer_triple_sum_product_l3419_341921

theorem integer_triple_sum_product : 
  ∀ a b c : ℕ+, 
    (a + b + c = 25 ∧ a * b * c = 360) ↔ 
    ((a = 4 ∧ b = 6 ∧ c = 15) ∨ 
     (a = 3 ∧ b = 10 ∧ c = 12) ∨
     (a = 4 ∧ b = 15 ∧ c = 6) ∨ 
     (a = 6 ∧ b = 4 ∧ c = 15) ∨
     (a = 6 ∧ b = 15 ∧ c = 4) ∨
     (a = 15 ∧ b = 4 ∧ c = 6) ∨
     (a = 15 ∧ b = 6 ∧ c = 4) ∨
     (a = 3 ∧ b = 12 ∧ c = 10) ∨
     (a = 10 ∧ b = 3 ∧ c = 12) ∨
     (a = 10 ∧ b = 12 ∧ c = 3) ∨
     (a = 12 ∧ b = 3 ∧ c = 10) ∨
     (a = 12 ∧ b = 10 ∧ c = 3)) :=
by sorry

end integer_triple_sum_product_l3419_341921


namespace complex_equation_solution_l3419_341902

theorem complex_equation_solution (a b : ℝ) : 
  (a : ℂ) + (b : ℂ) * Complex.I = (1 - Complex.I) / (2 * Complex.I) → 
  a = -1/2 ∧ b = -1/2 := by
  sorry

end complex_equation_solution_l3419_341902


namespace kaleb_final_amount_l3419_341918

def kaleb_lawn_business (spring_earnings summer_earnings supply_costs : ℕ) : ℕ :=
  spring_earnings + summer_earnings - supply_costs

theorem kaleb_final_amount :
  kaleb_lawn_business 4 50 4 = 50 := by
  sorry

end kaleb_final_amount_l3419_341918


namespace bus_problem_l3419_341975

/-- The number of people who boarded at the second stop of a bus journey --/
def second_stop_boarders (
  rows : ℕ) (seats_per_row : ℕ) 
  (initial_passengers : ℕ) 
  (first_stop_on : ℕ) (first_stop_off : ℕ)
  (second_stop_off : ℕ) 
  (final_empty_seats : ℕ) : ℕ := by
  sorry

/-- The number of people who boarded at the second stop is 17 --/
theorem bus_problem : 
  second_stop_boarders 23 4 16 15 3 10 57 = 17 := by
  sorry

end bus_problem_l3419_341975


namespace decimal_place_150_l3419_341963

/-- The decimal representation of 5/6 -/
def decimal_rep_5_6 : ℚ := 5/6

/-- The length of the repeating cycle in the decimal representation of 5/6 -/
def cycle_length : ℕ := 6

/-- The nth digit in the decimal representation of 5/6 -/
def nth_digit (n : ℕ) : ℕ := sorry

theorem decimal_place_150 :
  nth_digit 150 = 3 :=
sorry

end decimal_place_150_l3419_341963


namespace runners_in_picture_probability_l3419_341943

/-- Represents a runner on a circular track -/
structure Runner where
  name : String
  lap_time : ℕ
  direction : Bool  -- true for counterclockwise, false for clockwise

/-- Represents the photographer's setup -/
structure Photographer where
  start_time : ℕ  -- in seconds
  end_time : ℕ    -- in seconds
  track_coverage : ℚ  -- fraction of track covered in picture

/-- Calculates the probability of both runners being in the picture -/
def probability_both_in_picture (linda : Runner) (luis : Runner) (photographer : Photographer) : ℚ :=
  sorry  -- Proof goes here

/-- The main theorem statement -/
theorem runners_in_picture_probability :
  let linda : Runner := { name := "Linda", lap_time := 120, direction := true }
  let luis : Runner := { name := "Luis", lap_time := 75, direction := false }
  let photographer : Photographer := { start_time := 900, end_time := 960, track_coverage := 1/3 }
  probability_both_in_picture linda luis photographer = 5/6 := by
  sorry  -- Proof goes here

end runners_in_picture_probability_l3419_341943


namespace contest_prize_distribution_l3419_341933

theorem contest_prize_distribution (total_prize : ℕ) (total_winners : ℕ) 
  (first_prize : ℕ) (second_prize : ℕ) (third_prize : ℕ) 
  (h1 : total_prize = 800) (h2 : total_winners = 18) 
  (h3 : first_prize = 200) (h4 : second_prize = 150) (h5 : third_prize = 120) :
  let remaining_prize := total_prize - (first_prize + second_prize + third_prize)
  let remaining_winners := total_winners - 3
  remaining_prize / remaining_winners = 22 := by
sorry

end contest_prize_distribution_l3419_341933


namespace initial_student_count_l3419_341979

theorem initial_student_count (initial_avg : ℝ) (new_avg : ℝ) (dropped_score : ℝ) :
  initial_avg = 62.5 →
  new_avg = 62.0 →
  dropped_score = 70 →
  ∃ n : ℕ, n > 0 ∧ 
    (n : ℝ) * initial_avg = ((n - 1) : ℝ) * new_avg + dropped_score ∧
    n = 16 :=
by sorry

end initial_student_count_l3419_341979


namespace sum_of_circle_areas_in_5_12_13_triangle_l3419_341982

/-- Represents a circle with a given radius -/
structure Circle where
  radius : ℝ
  radius_pos : radius > 0

/-- Represents a right triangle with circles at its vertices -/
structure TriangleWithCircles where
  side1 : ℝ
  side2 : ℝ
  hypotenuse : ℝ
  circle1 : Circle
  circle2 : Circle
  circle3 : Circle
  is_right_triangle : side1^2 + side2^2 = hypotenuse^2
  circles_tangent : 
    circle1.radius + circle2.radius = side1 ∧
    circle2.radius + circle3.radius = side2 ∧
    circle1.radius + circle3.radius = hypotenuse

/-- The sum of the areas of the circles in a 5-12-13 right triangle with mutually tangent circles at its vertices is 81π -/
theorem sum_of_circle_areas_in_5_12_13_triangle (t : TriangleWithCircles) 
  (h1 : t.side1 = 5) (h2 : t.side2 = 12) (h3 : t.hypotenuse = 13) : 
  π * t.circle1.radius^2 + π * t.circle2.radius^2 + π * t.circle3.radius^2 = 81 * π := by
  sorry

end sum_of_circle_areas_in_5_12_13_triangle_l3419_341982


namespace february_average_rainfall_l3419_341984

-- Define the given conditions
def total_rainfall : ℝ := 280
def days_in_february : ℕ := 28
def hours_per_day : ℕ := 24

-- Define the theorem
theorem february_average_rainfall :
  total_rainfall / (days_in_february * hours_per_day : ℝ) = 5 / 12 := by
  sorry

end february_average_rainfall_l3419_341984


namespace thomas_run_conversion_l3419_341915

/-- Converts a number from base 7 to base 10 -/
def base7ToBase10 (d₃ d₂ d₁ d₀ : ℕ) : ℕ :=
  d₃ * 7^3 + d₂ * 7^2 + d₁ * 7^1 + d₀ * 7^0

/-- Thomas's run in base 7 -/
def thomasRunBase7 : ℕ × ℕ × ℕ × ℕ := (4, 2, 1, 3)

theorem thomas_run_conversion :
  let (d₃, d₂, d₁, d₀) := thomasRunBase7
  base7ToBase10 d₃ d₂ d₁ d₀ = 1480 := by
sorry

end thomas_run_conversion_l3419_341915


namespace orlando_weight_gain_l3419_341991

theorem orlando_weight_gain (x : ℝ) : 
  x + (2 * x + 2) + ((1 / 2) * (2 * x + 2) - 3) = 20 → x = 5 := by
  sorry

end orlando_weight_gain_l3419_341991


namespace tan_theta_negative_three_l3419_341931

theorem tan_theta_negative_three (θ : Real) (h : Real.tan θ = -3) :
  (Real.sin θ - 2 * Real.cos θ) / (Real.cos θ + Real.sin θ) = 5/2 := by
  sorry

end tan_theta_negative_three_l3419_341931


namespace monochromatic_triangle_probability_l3419_341960

/-- The number of vertices in the complete graph -/
def n : ℕ := 6

/-- The number of colors used for coloring the edges -/
def num_colors : ℕ := 3

/-- The probability of a specific triangle being non-monochromatic -/
def p_non_monochromatic : ℚ := 24 / 27

/-- The number of triangles in a complete graph with n vertices -/
def num_triangles : ℕ := n.choose 3

/-- The probability of having at least one monochromatic triangle -/
noncomputable def p_at_least_one_monochromatic : ℚ :=
  1 - p_non_monochromatic ^ num_triangles

theorem monochromatic_triangle_probability :
  p_at_least_one_monochromatic = 872 / 1000 := by
  sorry

end monochromatic_triangle_probability_l3419_341960


namespace puzzle_solution_l3419_341999

theorem puzzle_solution (x y z w : ℕ+) 
  (h1 : x^3 = y^2) 
  (h2 : z^4 = w^3) 
  (h3 : z - x = 17) : 
  w - y = 73 := by
  sorry

end puzzle_solution_l3419_341999


namespace rectangle_area_increase_l3419_341974

theorem rectangle_area_increase (p a : ℝ) (h_a : a > 0) :
  let perimeter := 2 * p
  let increase := a
  let area_increase := 
    fun (x y : ℝ) => 
      ((x + increase) * (y + increase)) - (x * y)
  ∀ x y, x + y = p → area_increase x y = a * (p + a) := by
sorry

end rectangle_area_increase_l3419_341974


namespace iesha_school_books_l3419_341952

/-- The number of books Iesha has about school -/
def books_about_school (total_books sports_books : ℕ) : ℕ :=
  total_books - sports_books

/-- Theorem stating that Iesha has 19 books about school -/
theorem iesha_school_books : 
  books_about_school 58 39 = 19 := by
  sorry

end iesha_school_books_l3419_341952


namespace total_tickets_after_sharing_l3419_341934

def tate_initial_tickets : ℕ := 32
def tate_bought_tickets : ℕ := 2

def tate_final_tickets : ℕ := tate_initial_tickets + tate_bought_tickets

def peyton_initial_tickets : ℕ := tate_final_tickets / 2

def peyton_given_away_tickets : ℕ := peyton_initial_tickets / 3

def peyton_final_tickets : ℕ := peyton_initial_tickets - peyton_given_away_tickets

theorem total_tickets_after_sharing :
  tate_final_tickets + peyton_final_tickets = 46 := by
  sorry

end total_tickets_after_sharing_l3419_341934


namespace max_k_value_l3419_341968

theorem max_k_value (m : ℝ) (h1 : 0 < m) (h2 : m < 1/2) :
  (∀ k : ℝ, 1/m + 2/(1-2*m) ≥ k) → (∃ k : ℝ, k = 8 ∧ ∀ k' : ℝ, (∀ m' : ℝ, 0 < m' ∧ m' < 1/2 → 1/m' + 2/(1-2*m') ≥ k') → k' ≤ k) :=
by sorry

end max_k_value_l3419_341968


namespace marly_soup_bags_l3419_341937

/-- The number of bags needed for Marly's soup -/
def bags_needed (milk_quarts chicken_stock_multiplier vegetable_quarts bag_capacity : ℚ) : ℚ :=
  (milk_quarts + chicken_stock_multiplier * milk_quarts + vegetable_quarts) / bag_capacity

/-- Theorem: Marly needs 3 bags for his soup -/
theorem marly_soup_bags :
  bags_needed 2 3 1 3 = 3 := by
sorry

end marly_soup_bags_l3419_341937


namespace seating_theorem_l3419_341967

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def seating_arrangements (n : ℕ) (k : ℕ) : ℕ :=
  factorial n - factorial (n - k + 1) * factorial k

theorem seating_theorem :
  seating_arrangements 8 3 = 36000 :=
sorry

end seating_theorem_l3419_341967


namespace island_not_named_Maya_l3419_341909

-- Define the inhabitants
inductive Inhabitant : Type
| A : Inhabitant
| B : Inhabitant

-- Define the possible states of an inhabitant
inductive State : Type
| TruthTeller : State
| Liar : State

-- Define the name of the island
def IslandName : Type := Bool

-- Define the statements made by A and B
def statement_A (state_A state_B : State) (island_name : IslandName) : Prop :=
  (state_A = State.Liar ∧ state_B = State.Liar) ∧ island_name = true

def statement_B (state_A state_B : State) (island_name : IslandName) : Prop :=
  (state_A = State.Liar ∨ state_B = State.Liar) ∧ island_name = false

-- The main theorem
theorem island_not_named_Maya :
  ∀ (state_A state_B : State) (island_name : IslandName),
    (state_A = State.Liar → ¬statement_A state_A state_B island_name) ∧
    (state_A = State.TruthTeller → statement_A state_A state_B island_name) ∧
    (state_B = State.Liar → ¬statement_B state_A state_B island_name) ∧
    (state_B = State.TruthTeller → statement_B state_A state_B island_name) →
    island_name = false :=
by
  sorry


end island_not_named_Maya_l3419_341909


namespace rectangle_max_m_l3419_341965

/-- Given a rectangle with area S and perimeter p, 
    M = (16 - p) / (p^2 + 2p) is maximized when the rectangle is a square -/
theorem rectangle_max_m (S : ℝ) (p : ℝ) (h_S : S > 0) (h_p : p > 0) :
  let M := (16 - p) / (p^2 + 2*p)
  M ≤ (4 - Real.sqrt S) / (4*S + 2*Real.sqrt S) :=
by sorry

end rectangle_max_m_l3419_341965


namespace expected_original_positions_l3419_341989

/-- The number of balls arranged in a circle -/
def numBalls : ℕ := 7

/-- The number of independent random transpositions -/
def numTranspositions : ℕ := 3

/-- The probability of a ball being in its original position after the transpositions -/
def probOriginalPosition : ℚ := 127 / 343

/-- The expected number of balls in their original positions after the transpositions -/
def expectedOriginalPositions : ℚ := numBalls * probOriginalPosition

theorem expected_original_positions :
  expectedOriginalPositions = 889 / 343 := by sorry

end expected_original_positions_l3419_341989


namespace no_all_ones_quadratic_l3419_341913

/-- A natural number whose decimal representation consists only of ones -/
def all_ones (n : ℕ) : Prop :=
  ∃ k : ℕ, n = (10^k - 1) / 9

/-- The property that a natural number's decimal representation consists only of ones -/
def has_all_ones_representation (n : ℕ) : Prop :=
  all_ones n

/-- A quadratic polynomial with integer coefficients -/
def is_quadratic_polynomial (P : ℕ → ℕ) : Prop :=
  ∃ a b c : ℤ, ∀ x : ℕ, P x = a * x^2 + b * x + c

theorem no_all_ones_quadratic :
  ∀ P : ℕ → ℕ, is_quadratic_polynomial P →
    ∃ n : ℕ, has_all_ones_representation n ∧ ¬(has_all_ones_representation (P n)) :=
sorry

end no_all_ones_quadratic_l3419_341913


namespace initial_cells_theorem_l3419_341910

/-- Calculates the number of cells after one hour given the initial number of cells -/
def cellsAfterOneHour (initialCells : ℕ) : ℕ :=
  2 * (initialCells - 2)

/-- Calculates the number of cells after n hours given the initial number of cells -/
def cellsAfterNHours (initialCells : ℕ) (n : ℕ) : ℕ :=
  match n with
  | 0 => initialCells
  | n + 1 => cellsAfterOneHour (cellsAfterNHours initialCells n)

/-- Theorem stating that 9 initial cells result in 164 cells after 5 hours -/
theorem initial_cells_theorem :
  cellsAfterNHours 9 5 = 164 :=
by sorry

end initial_cells_theorem_l3419_341910


namespace large_cartridge_pages_large_cartridge_pages_proof_l3419_341981

theorem large_cartridge_pages : ℕ → ℕ → ℕ → Prop :=
  fun small_pages medium_pages large_pages =>
    (small_pages = 600) →
    (3 * small_pages = 2 * medium_pages) →
    (3 * medium_pages = 2 * large_pages) →
    (large_pages = 1350)

-- The proof would go here
theorem large_cartridge_pages_proof :
  ∃ (small_pages medium_pages large_pages : ℕ),
    large_cartridge_pages small_pages medium_pages large_pages :=
sorry

end large_cartridge_pages_large_cartridge_pages_proof_l3419_341981


namespace list_number_fraction_l3419_341944

theorem list_number_fraction (list : List ℝ) (n : ℝ) : 
  list.length = 31 ∧ 
  n ∉ list ∧
  n = 5 * ((list.sum) / 30) →
  n / (list.sum + n) = 1 / 7 := by
sorry

end list_number_fraction_l3419_341944


namespace trillion_to_scientific_notation_l3419_341973

/-- Represents the value of one trillion -/
def trillion : ℕ := 1000000000000

/-- Proves that 6.13 trillion is equal to 6.13 × 10^12 -/
theorem trillion_to_scientific_notation : 
  (6.13 : ℝ) * (trillion : ℝ) = 6.13 * (10 : ℝ)^12 := by
  sorry

end trillion_to_scientific_notation_l3419_341973


namespace number_exceeding_twenty_percent_l3419_341994

theorem number_exceeding_twenty_percent : ∃ x : ℝ, x = 0.20 * x + 40 → x = 50 := by
  sorry

end number_exceeding_twenty_percent_l3419_341994


namespace dice_probability_l3419_341939

def num_dice : ℕ := 8
def num_faces : ℕ := 6
def num_pairs : ℕ := 3

def total_outcomes : ℕ := num_faces ^ num_dice

def favorable_outcomes : ℕ :=
  Nat.choose num_faces num_pairs *
  Nat.choose (num_faces - num_pairs) (num_dice - 2 * num_pairs) *
  Nat.factorial num_pairs *
  Nat.factorial (num_dice - 2 * num_pairs) *
  Nat.choose num_dice 2 *
  Nat.choose (num_dice - 2) 2 *
  Nat.choose (num_dice - 4) 2

theorem dice_probability :
  (favorable_outcomes : ℚ) / total_outcomes = 525 / 972 := by
  sorry

end dice_probability_l3419_341939


namespace soccer_match_goals_l3419_341935

/-- Calculates the total number of goals scored in a soccer match -/
def total_goals (kickers_first : ℕ) : ℕ := 
  let kickers_second : ℕ := 2 * kickers_first
  let spiders_first : ℕ := kickers_first / 2
  let spiders_second : ℕ := 2 * kickers_second
  kickers_first + kickers_second + spiders_first + spiders_second

/-- Theorem stating that given the conditions of the soccer match, the total goals scored is 15 -/
theorem soccer_match_goals : total_goals 2 = 15 := by
  sorry

end soccer_match_goals_l3419_341935


namespace tens_digit_of_19_power_2023_l3419_341932

theorem tens_digit_of_19_power_2023 : ∃ n : ℕ, 19^2023 ≡ 10 * n + 5 [ZMOD 100] := by
  sorry

end tens_digit_of_19_power_2023_l3419_341932


namespace business_partnership_problem_l3419_341922

/-- A business partnership problem -/
theorem business_partnership_problem
  (x_capital y_capital z_capital : ℕ)
  (total_profit z_profit : ℕ)
  (x_months y_months z_months : ℕ)
  (h1 : x_capital = 20000)
  (h2 : z_capital = 30000)
  (h3 : total_profit = 50000)
  (h4 : z_profit = 14000)
  (h5 : x_months = 12)
  (h6 : y_months = 12)
  (h7 : z_months = 7)
  (h8 : (z_capital * z_months) / (x_capital * x_months + y_capital * y_months + z_capital * z_months) = z_profit / total_profit) :
  y_capital = 25000 := by
  sorry


end business_partnership_problem_l3419_341922


namespace sampling_survey_appropriate_l3419_341949

/-- Represents a survey method -/
inductive SurveyMethod
| Census
| SamplingSurvey

/-- Represents the characteristics of a population survey -/
structure PopulationSurvey where
  populationSize : Nat
  needsEfficiency : Bool

/-- Determines the appropriate survey method based on population characteristics -/
def appropriateSurveyMethod (survey : PopulationSurvey) : SurveyMethod :=
  if survey.populationSize > 1000000 && survey.needsEfficiency
  then SurveyMethod.SamplingSurvey
  else SurveyMethod.Census

/-- Theorem: For a large population requiring efficient data collection,
    sampling survey is the appropriate method -/
theorem sampling_survey_appropriate
  (survey : PopulationSurvey)
  (h1 : survey.populationSize > 1000000)
  (h2 : survey.needsEfficiency) :
  appropriateSurveyMethod survey = SurveyMethod.SamplingSurvey :=
by
  sorry


end sampling_survey_appropriate_l3419_341949


namespace ferris_wheel_capacity_l3419_341951

/-- The number of small seats on the Ferris wheel -/
def num_small_seats : ℕ := 2

/-- The total number of people that can ride on small seats -/
def total_people_small_seats : ℕ := 28

/-- The number of people each small seat can hold -/
def people_per_small_seat : ℕ := total_people_small_seats / num_small_seats

theorem ferris_wheel_capacity : people_per_small_seat = 14 := by
  sorry

end ferris_wheel_capacity_l3419_341951


namespace inequality_system_solution_l3419_341969

theorem inequality_system_solution (x : ℤ) :
  (-3/2 : ℚ) < x ∧ (x : ℚ) ≤ 2 →
  -2*x + 7 < 10 ∧ (7*x + 1)/5 - 1 ≤ x := by
  sorry

end inequality_system_solution_l3419_341969


namespace place_value_ratio_l3419_341959

/-- Represents a decimal number with its integer and fractional parts -/
structure DecimalNumber where
  integerPart : ℕ
  fractionalPart : ℕ
  fractionalDigits : ℕ

/-- Returns the place value of a digit at a given position in a decimal number -/
def placeValue (n : DecimalNumber) (position : ℤ) : ℚ :=
  10 ^ position

/-- The decimal number 50467.8912 -/
def number : DecimalNumber :=
  { integerPart := 50467
  , fractionalPart := 8912
  , fractionalDigits := 4 }

/-- The position of digit 8 in the number (counting from right, negative for fractional part) -/
def pos8 : ℤ := -1

/-- The position of digit 7 in the number (counting from right) -/
def pos7 : ℤ := 1

theorem place_value_ratio :
  (placeValue number pos8) / (placeValue number pos7) = 1 / 100 := by
  sorry

end place_value_ratio_l3419_341959


namespace no_prime_satisfies_condition_l3419_341964

theorem no_prime_satisfies_condition : ¬ ∃ p : ℕ, Nat.Prime p ∧ (10 : ℝ) * p = p + 5.4 := by
  sorry

end no_prime_satisfies_condition_l3419_341964


namespace comprehensive_survey_appropriate_for_grade9_vision_l3419_341996

/-- Represents the appropriateness of a survey method for a given scenario -/
inductive SurveyAppropriateness
  | Appropriate
  | Inappropriate

/-- Represents different survey methods -/
inductive SurveyMethod
  | Comprehensive
  | Sample

/-- Represents characteristics that can be surveyed -/
inductive Characteristic
  | Vision
  | EquipmentQuality

/-- Represents the size of a group being surveyed -/
inductive GroupSize
  | Large
  | Small

/-- Function to determine if a survey method is appropriate for a given characteristic and group size -/
def is_appropriate (method : SurveyMethod) (char : Characteristic) (size : GroupSize) : SurveyAppropriateness :=
  match method, char, size with
  | SurveyMethod.Comprehensive, Characteristic.Vision, GroupSize.Large => SurveyAppropriateness.Appropriate
  | _, _, _ => SurveyAppropriateness.Inappropriate

theorem comprehensive_survey_appropriate_for_grade9_vision :
  is_appropriate SurveyMethod.Comprehensive Characteristic.Vision GroupSize.Large = SurveyAppropriateness.Appropriate :=
by sorry

end comprehensive_survey_appropriate_for_grade9_vision_l3419_341996


namespace arithmetic_mean_of_sarahs_scores_l3419_341992

def sarahs_scores : List ℝ := [87, 90, 86, 93, 89, 92]

theorem arithmetic_mean_of_sarahs_scores :
  (sarahs_scores.sum / sarahs_scores.length : ℝ) = 89.5 := by
  sorry

end arithmetic_mean_of_sarahs_scores_l3419_341992


namespace largest_harmonious_n_is_correct_l3419_341925

/-- A coloring of a regular polygon's sides and diagonals. -/
def Coloring (n : ℕ) := Fin n → Fin n → Fin 2018

/-- A harmonious coloring has no two-colored triangles. -/
def Harmonious (n : ℕ) (c : Coloring n) : Prop :=
  ∀ i j k : Fin n, (c i j = c i k ∧ c i j ≠ c j k) → c i k = c j k

/-- The largest N for which a harmonious coloring of a regular N-gon exists. -/
def LargestHarmoniousN : ℕ := 2017^2

theorem largest_harmonious_n_is_correct :
  (∃ (c : Coloring LargestHarmoniousN), Harmonious LargestHarmoniousN c) ∧
  (∀ n > LargestHarmoniousN, ¬∃ (c : Coloring n), Harmonious n c) :=
sorry

end largest_harmonious_n_is_correct_l3419_341925


namespace cube_sum_inequality_l3419_341957

theorem cube_sum_inequality (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) :
  2 * (x^3 + y^3 + z^3) ≥ x^2 * y + x^2 * z + y^2 * z + y^2 * x + z^2 * x + z^2 * y :=
by sorry

end cube_sum_inequality_l3419_341957


namespace moe_has_least_money_l3419_341970

-- Define the set of people
inductive Person : Type
  | Bo | Coe | Flo | Jo | Moe | Zoe

-- Define a function to represent the amount of money each person has
variable (money : Person → ℕ)

-- Define the conditions
axiom different_amounts : ∀ (p q : Person), p ≠ q → money p ≠ money q

axiom flo_more_than_jo_bo : money Person.Flo > money Person.Jo ∧ money Person.Flo > money Person.Bo

axiom bo_coe_more_than_moe_less_than_zoe : 
  money Person.Bo > money Person.Moe ∧ 
  money Person.Coe > money Person.Moe ∧
  money Person.Zoe > money Person.Bo ∧
  money Person.Zoe > money Person.Coe

axiom jo_more_than_moe_zoe_less_than_bo : 
  money Person.Jo > money Person.Moe ∧
  money Person.Jo > money Person.Zoe ∧
  money Person.Bo > money Person.Jo

-- Theorem to prove
theorem moe_has_least_money : 
  ∀ (p : Person), p ≠ Person.Moe → money Person.Moe < money p :=
sorry

end moe_has_least_money_l3419_341970


namespace parabola_slope_relation_l3419_341923

/-- Parabola struct representing y^2 = 2px --/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- Point on a 2D plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- Line on a 2D plane --/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Theorem statement --/
theorem parabola_slope_relation 
  (C : Parabola) 
  (A : Point)
  (B : Point)
  (P M N : Point)
  (h_A : A.y^2 = 2 * C.p * A.x)
  (h_A_x : A.x = 1)
  (h_B : B.x = -C.p/2 ∧ B.y = 0)
  (h_AB : (A.x - B.x)^2 + (A.y - B.y)^2 = 8)
  (h_P : P.y^2 = 2 * C.p * P.x ∧ P.y = 2)
  (h_M : M.y^2 = 2 * C.p * M.x)
  (h_N : N.y^2 = 2 * C.p * N.x)
  (k₁ k₂ k₃ : ℝ)
  (h_k₁ : k₁ ≠ 0)
  (h_k₂ : k₂ ≠ 0)
  (h_k₃ : k₃ ≠ 0)
  (h_PM : (M.y - P.y) = k₁ * (M.x - P.x))
  (h_PN : (N.y - P.y) = k₂ * (N.x - P.x))
  (h_MN : (N.y - M.y) = k₃ * (N.x - M.x)) :
  1/k₁ + 1/k₂ - 1/k₃ = 1 := by sorry

end parabola_slope_relation_l3419_341923


namespace altitude_and_equidistant_lines_l3419_341976

/-- Given three points in a plane -/
def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (-2, 4)
def C : ℝ × ℝ := (5, 7)

/-- Line l₁ containing the altitude from B to BC -/
def l₁ (x y : ℝ) : Prop := 7 * x + 3 * y - 10 = 0

/-- Lines l₂ passing through B with equal distances from A and C -/
def l₂₁ (y : ℝ) : Prop := y = 4
def l₂₂ (x y : ℝ) : Prop := 3 * x - 2 * y + 14 = 0

/-- Main theorem -/
theorem altitude_and_equidistant_lines :
  (∀ x y, l₁ x y ↔ (x - B.1) * (C.2 - B.2) = (y - B.2) * (C.1 - B.1)) ∧
  (∀ x y, (l₂₁ y ∨ l₂₂ x y) ↔ 
    ((x = B.1 ∧ y = B.2) ∨ 
     (abs ((y - A.2) - ((y - B.2) / (x - B.1)) * (A.1 - B.1)) = 
      abs ((y - C.2) - ((y - B.2) / (x - B.1)) * (C.1 - B.1))))) :=
sorry

end altitude_and_equidistant_lines_l3419_341976


namespace segment_length_l3419_341926

theorem segment_length : 
  let endpoints := {x : ℝ | |x - (27 : ℝ)^(1/3)| = 5}
  ∃ a b : ℝ, a ∈ endpoints ∧ b ∈ endpoints ∧ |b - a| = 10 :=
by
  sorry

end segment_length_l3419_341926


namespace quadratic_root_sqrt5_minus3_l3419_341930

theorem quadratic_root_sqrt5_minus3 :
  ∃ (a b c : ℚ), a ≠ 0 ∧ 
  (∀ x : ℝ, a * x^2 + b * x + c = 0 ↔ x = Real.sqrt 5 - 3 ∨ x = -Real.sqrt 5 + 1) ∧
  a = 1 ∧ b = 2 ∧ c = -2 :=
sorry

end quadratic_root_sqrt5_minus3_l3419_341930


namespace hamburger_cost_is_four_l3419_341993

/-- The cost of Morgan's lunch items and transaction details -/
structure LunchOrder where
  hamburger_cost : ℝ
  onion_rings_cost : ℝ
  smoothie_cost : ℝ
  total_paid : ℝ
  change_received : ℝ

/-- Theorem stating the cost of the hamburger in Morgan's lunch order -/
theorem hamburger_cost_is_four (order : LunchOrder)
  (h1 : order.onion_rings_cost = 2)
  (h2 : order.smoothie_cost = 3)
  (h3 : order.total_paid = 20)
  (h4 : order.change_received = 11) :
  order.hamburger_cost = 4 := by
  sorry

end hamburger_cost_is_four_l3419_341993


namespace pablos_payment_per_page_l3419_341997

/-- The amount Pablo's mother pays him per page, in dollars. -/
def payment_per_page : ℚ := 1 / 100

/-- The number of pages in each book Pablo reads. -/
def pages_per_book : ℕ := 150

/-- The number of books Pablo read. -/
def books_read : ℕ := 12

/-- The amount Pablo spent on candy, in dollars. -/
def candy_cost : ℕ := 15

/-- The amount Pablo had leftover, in dollars. -/
def leftover : ℕ := 3

theorem pablos_payment_per_page :
  payment_per_page * (pages_per_book * books_read : ℚ) = (candy_cost + leftover : ℚ) := by
  sorry

end pablos_payment_per_page_l3419_341997


namespace sanctuary_animal_pairs_l3419_341903

theorem sanctuary_animal_pairs : 
  let bird_species : ℕ := 29
  let bird_pairs_per_species : ℕ := 7
  let marine_species : ℕ := 15
  let marine_pairs_per_species : ℕ := 9
  let mammal_species : ℕ := 22
  let mammal_pairs_per_species : ℕ := 6
  
  bird_species * bird_pairs_per_species + 
  marine_species * marine_pairs_per_species + 
  mammal_species * mammal_pairs_per_species = 470 :=
by
  sorry

end sanctuary_animal_pairs_l3419_341903


namespace simplify_trigonometric_expression_l3419_341906

theorem simplify_trigonometric_expression (α : Real) 
  (h : π < α ∧ α < 3*π/2) : 
  Real.sqrt ((1 + Real.cos (9*π/2 - α)) / (1 + Real.sin (α - 5*π))) - 
  Real.sqrt ((1 - Real.cos (-3*π/2 - α)) / (1 - Real.sin (α - 9*π))) = 
  -2 * Real.tan α := by
  sorry

end simplify_trigonometric_expression_l3419_341906


namespace edge_coloring_theorem_l3419_341988

/-- A complete graph on n vertices -/
def CompleteGraph (n : ℕ) := Unit

/-- A coloring of edges with k colors -/
def Coloring (G : CompleteGraph 10) (k : ℕ) := Unit

/-- Predicate: Any subset of m vertices contains edges of all k colors -/
def AllColorsInSubset (G : CompleteGraph 10) (c : Coloring G k) (m k : ℕ) : Prop := sorry

theorem edge_coloring_theorem (G : CompleteGraph 10) :
  (∃ c : Coloring G 5, AllColorsInSubset G c 5 5) ∧
  (¬ ∃ c : Coloring G 4, AllColorsInSubset G c 4 4) := by sorry

end edge_coloring_theorem_l3419_341988


namespace isosceles_triangle_most_stable_isosceles_triangle_stable_other_shapes_not_stable_l3419_341942

/-- Represents a geometric shape -/
inductive Shape
  | IsoscelesTriangle
  | Rectangle
  | Square
  | Parallelogram

/-- Stability measure of a shape -/
def stability (s : Shape) : ℕ :=
  match s with
  | Shape.IsoscelesTriangle => 3
  | Shape.Rectangle => 2
  | Shape.Square => 2
  | Shape.Parallelogram => 1

/-- A shape is considered stable if its stability measure is greater than 2 -/
def is_stable (s : Shape) : Prop := stability s > 2

theorem isosceles_triangle_most_stable :
  ∀ s : Shape, s ≠ Shape.IsoscelesTriangle → stability Shape.IsoscelesTriangle > stability s :=
by sorry

theorem isosceles_triangle_stable :
  is_stable Shape.IsoscelesTriangle :=
by sorry

theorem other_shapes_not_stable :
  ¬ is_stable Shape.Rectangle ∧
  ¬ is_stable Shape.Square ∧
  ¬ is_stable Shape.Parallelogram :=
by sorry

end isosceles_triangle_most_stable_isosceles_triangle_stable_other_shapes_not_stable_l3419_341942


namespace shortest_tree_height_l3419_341924

/-- Given three trees with specified height relationships, prove the height of the shortest tree -/
theorem shortest_tree_height (tallest middle shortest : ℝ) : 
  tallest = 150 →
  middle = 2/3 * tallest →
  shortest = 1/2 * middle →
  shortest = 50 := by
sorry

end shortest_tree_height_l3419_341924


namespace rectangle_area_l3419_341990

/-- Given a rectangle with length twice its width and width of 5 inches, prove its area is 50 square inches. -/
theorem rectangle_area (width : ℝ) (length : ℝ) : 
  width = 5 → length = 2 * width → width * length = 50 := by sorry

end rectangle_area_l3419_341990


namespace arithmetic_problem_l3419_341955

theorem arithmetic_problem : 40 + 5 * 12 / (180 / 3) = 41 := by
  sorry

end arithmetic_problem_l3419_341955


namespace computer_price_increase_l3419_341912

theorem computer_price_increase (c : ℝ) (h : 2 * c = 540) : 
  c * (1 + 0.3) = 351 := by
sorry

end computer_price_increase_l3419_341912


namespace sp_length_l3419_341962

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)
  (ab_length : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 8)
  (bc_length : Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) = 9)
  (ca_length : Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 10)

-- Define the point T
def T (triangle : Triangle) : ℝ × ℝ := sorry

-- Define the point S
def S (triangle : Triangle) : ℝ × ℝ := sorry

-- Define the point P
def P (triangle : Triangle) : ℝ × ℝ := sorry

-- Theorem statement
theorem sp_length (triangle : Triangle) : 
  Real.sqrt ((S triangle).1 - (P triangle).1)^2 + ((S triangle).2 - (P triangle).2)^2 = 225/13 := by
  sorry

end sp_length_l3419_341962


namespace range_of_a_l3419_341917

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then 2 * x^3 + x^2 + 1 else Real.exp (a * x)

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc (-2) 3, f a x ≤ 2) ∧
  (∃ x ∈ Set.Icc (-2) 3, f a x = 2) →
  a ≤ (1/3) * Real.log 2 :=
sorry

end range_of_a_l3419_341917


namespace successive_numbers_product_l3419_341927

theorem successive_numbers_product (n : ℤ) : 
  n * (n + 1) = 2652 → n = 51 := by
  sorry

end successive_numbers_product_l3419_341927


namespace weight_of_b_l3419_341919

/-- Given three weights a, b, and c, prove that b equals 31 when:
    1. The average of a, b, and c is 45.
    2. The average of a and b is 40.
    3. The average of b and c is 43. -/
theorem weight_of_b (a b c : ℝ) 
  (h1 : (a + b + c) / 3 = 45)
  (h2 : (a + b) / 2 = 40)
  (h3 : (b + c) / 2 = 43) : 
  b = 31 := by
  sorry


end weight_of_b_l3419_341919


namespace soap_promotion_theorem_l3419_341987

/-- The original price of soap in yuan -/
def original_price : ℝ := 2

/-- The cost of buying n pieces of soap under Promotion 1 -/
def promotion1_cost (n : ℕ) : ℝ :=
  original_price + 0.7 * original_price * (n - 1 : ℝ)

/-- The cost of buying n pieces of soap under Promotion 2 -/
def promotion2_cost (n : ℕ) : ℝ :=
  0.8 * original_price * n

/-- The minimum number of soap pieces for Promotion 1 to be cheaper than Promotion 2 -/
def min_pieces : ℕ := 4

theorem soap_promotion_theorem :
  ∀ n : ℕ, n ≥ min_pieces →
    promotion1_cost n < promotion2_cost n ∧
    ∀ m : ℕ, m < min_pieces → promotion1_cost m ≥ promotion2_cost m :=
by sorry

end soap_promotion_theorem_l3419_341987


namespace increment_and_differential_at_point_l3419_341985

-- Define the function
def f (x : ℝ) : ℝ := x^2

-- Define the point and increment
def x₀ : ℝ := 2
def Δx : ℝ := 0.1

-- Define the increment of the function
def Δy (x : ℝ) (Δx : ℝ) : ℝ := f (x + Δx) - f x

-- Define the differential of the function
def dy (x : ℝ) (Δx : ℝ) : ℝ := 2 * x * Δx

-- Theorem statement
theorem increment_and_differential_at_point :
  (Δy x₀ Δx = 0.41) ∧ (dy x₀ Δx = 0.4) := by
  sorry

end increment_and_differential_at_point_l3419_341985


namespace double_factorial_properties_l3419_341980

def double_factorial : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => (n + 2) * double_factorial n

def units_digit (n : ℕ) : ℕ := n % 10

theorem double_factorial_properties :
  (double_factorial 2011 * double_factorial 2010 = Nat.factorial 2011) ∧
  ¬(double_factorial 2010 = 2 * Nat.factorial 1005) ∧
  ¬(double_factorial 2010 * double_factorial 2010 = Nat.factorial 2011) ∧
  (units_digit (double_factorial 2011) = 5) := by
  sorry

end double_factorial_properties_l3419_341980


namespace divisibility_of_7_power_minus_1_l3419_341978

theorem divisibility_of_7_power_minus_1 : ∃ k : ℤ, 7^51 - 1 = 103 * k := by
  sorry

end divisibility_of_7_power_minus_1_l3419_341978


namespace eggs_from_gertrude_l3419_341998

/-- Represents the number of eggs collected from each chicken -/
structure EggCollection where
  gertrude : ℕ
  blanche : ℕ
  nancy : ℕ
  martha : ℕ

/-- The theorem stating the number of eggs Trevor got from Gertrude -/
theorem eggs_from_gertrude (collection : EggCollection) : 
  collection.blanche = 3 →
  collection.nancy = 2 →
  collection.martha = 2 →
  collection.gertrude + collection.blanche + collection.nancy + collection.martha = 11 →
  collection.gertrude = 4 := by
  sorry

#check eggs_from_gertrude

end eggs_from_gertrude_l3419_341998


namespace second_diagonal_unrestricted_l3419_341940

/-- Represents a convex quadrilateral with specific properties -/
structure ConvexQuadrilateral where
  /-- The area of the quadrilateral in cm² -/
  area : ℝ
  /-- The length of the first diagonal in cm -/
  diagonal1 : ℝ
  /-- The length of the second diagonal in cm -/
  diagonal2 : ℝ
  /-- The sum of two opposite sides in cm -/
  opposite_sides_sum : ℝ
  /-- The area is positive -/
  area_pos : area > 0
  /-- Both diagonals are positive -/
  diag1_pos : diagonal1 > 0
  diag2_pos : diagonal2 > 0
  /-- The sum of opposite sides is non-negative -/
  opp_sides_sum_nonneg : opposite_sides_sum ≥ 0
  /-- The area is 32 cm² -/
  area_is_32 : area = 32
  /-- The sum of one diagonal and two opposite sides is 16 cm -/
  sum_is_16 : diagonal1 + opposite_sides_sum = 16

/-- Theorem stating that the second diagonal can be any positive real number -/
theorem second_diagonal_unrestricted (q : ConvexQuadrilateral) : 
  ∀ x : ℝ, x > 0 → ∃ q' : ConvexQuadrilateral, q'.diagonal2 = x := by
  sorry

end second_diagonal_unrestricted_l3419_341940


namespace equal_division_of_trout_l3419_341950

theorem equal_division_of_trout (total_trout : ℕ) (num_people : ℕ) (trout_per_person : ℕ) :
  total_trout = 18 →
  num_people = 2 →
  trout_per_person = total_trout / num_people →
  trout_per_person = 9 :=
by
  sorry

end equal_division_of_trout_l3419_341950


namespace smallest_positive_sum_l3419_341936

theorem smallest_positive_sum (x y : ℝ) : 
  (Real.sin x + Real.cos y) * (Real.cos x - Real.sin y) = 1 + Real.sin (x - y) * Real.cos (x + y) →
  ∃ (k : ℤ), x + y = 2 * π * (k : ℝ) ∧ 
  (∀ (m : ℤ), x + y = 2 * π * (m : ℝ) → k ≤ m) ∧
  0 < 2 * π * (k : ℝ) :=
sorry

end smallest_positive_sum_l3419_341936


namespace booklet_pages_theorem_l3419_341954

theorem booklet_pages_theorem (n : ℕ) (r : ℕ) : 
  (∃ (n : ℕ) (r : ℕ), 2 * n * (2 * n + 1) / 2 - (4 * r - 1) = 963 ∧ 
   1 ≤ r ∧ r ≤ n) → 
  (n = 22 ∧ r = 7) := by
  sorry

end booklet_pages_theorem_l3419_341954


namespace quadrilateral_area_l3419_341945

/-- A line with slope m passing through point (x₀, y₀) -/
def Line (m x₀ y₀ : ℝ) : ℝ → ℝ := λ x ↦ m * (x - x₀) + y₀

theorem quadrilateral_area : 
  let line1 := Line (-3) 5 5
  let line2 := Line (-1) 10 0
  let B := (0, line1 0)
  let E := (5, 5)
  let C := (10, 0)
  (B.2 * C.1 - (B.2 * E.1 + C.1 * E.2)) / 2 = 125 := by sorry

end quadrilateral_area_l3419_341945


namespace tank_filled_at_10pm_l3419_341929

/-- Represents the rainfall rate at a given hour after 1 pm -/
def rainfall_rate (hour : ℕ) : ℝ :=
  if hour = 0 then 2
  else if hour ≤ 4 then 1
  else 3

/-- Calculates the total rainfall up to a given hour after 1 pm -/
def total_rainfall (hour : ℕ) : ℝ :=
  (Finset.range (hour + 1)).sum rainfall_rate

/-- The height of the fish tank in inches -/
def tank_height : ℝ := 18

/-- Theorem stating that the fish tank will be filled at 10 pm -/
theorem tank_filled_at_10pm :
  ∃ (h : ℕ), h = 9 ∧ total_rainfall h ≥ tank_height ∧ total_rainfall (h - 1) < tank_height :=
sorry

end tank_filled_at_10pm_l3419_341929


namespace hyperbola_canonical_equation_l3419_341938

/-- Represents a hyperbola with foci on the x-axis -/
structure Hyperbola where
  ε : ℝ  -- eccentricity
  f : ℝ  -- focal distance

/-- The canonical equation of a hyperbola -/
def canonical_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 / 5 = 1

/-- Theorem: For a hyperbola with ε = 1.5 and focal distance 6, the canonical equation is x²/4 - y²/5 = 1 -/
theorem hyperbola_canonical_equation (h : Hyperbola) 
    (h_ε : h.ε = 1.5) 
    (h_f : h.f = 6) :
    ∀ x y : ℝ, canonical_equation h x y :=
  sorry

end hyperbola_canonical_equation_l3419_341938


namespace number_problem_l3419_341977

theorem number_problem (N : ℝ) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N - (1/2 : ℝ) * (1/6 : ℝ) * N = 35 →
  (40/100 : ℝ) * N = -280 := by
sorry

end number_problem_l3419_341977


namespace least_integer_absolute_value_l3419_341908

theorem least_integer_absolute_value (x : ℤ) : 
  (∀ y : ℤ, |3*y - 4| ≤ 18 → y ≥ x) → |3*x - 4| ≤ 18 → x = -4 :=
by sorry

end least_integer_absolute_value_l3419_341908


namespace line_tangent_to_ellipse_l3419_341904

/-- The value of j for which the line 4x - 7y + j = 0 is tangent to the ellipse x^2 + 4y^2 = 16 -/
theorem line_tangent_to_ellipse (x y j : ℝ) : 
  (∀ x y, 4*x - 7*y + j = 0 → x^2 + 4*y^2 = 16) ↔ j^2 = 450.5 := by
  sorry

end line_tangent_to_ellipse_l3419_341904


namespace theodore_sturgeon_books_l3419_341901

theorem theodore_sturgeon_books (h p : ℕ) : 
  h + p = 10 →
  30 * h + 20 * p = 250 →
  h = 5 :=
by sorry

end theodore_sturgeon_books_l3419_341901


namespace logarithm_inequality_l3419_341914

theorem logarithm_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  Real.log (1 + Real.sqrt (a * b)) ≤ (Real.log (1 + a) + Real.log (1 + b)) / 2 := by
  sorry

end logarithm_inequality_l3419_341914


namespace tan_beta_value_l3419_341941

theorem tan_beta_value (α β : Real) 
  (h1 : Real.tan α = -2/3) 
  (h2 : Real.tan (α + β) = 1/2) : 
  Real.tan β = 7/4 := by
  sorry

end tan_beta_value_l3419_341941


namespace board_theorem_l3419_341961

/-- Represents a board with gold and silver cells. -/
structure Board :=
  (size : ℕ)
  (is_gold : ℕ → ℕ → Bool)

/-- Counts the number of gold cells in a given rectangle of the board. -/
def count_gold (b : Board) (x y w h : ℕ) : ℕ :=
  (Finset.range w).sum (λ i =>
    (Finset.range h).sum (λ j =>
      if b.is_gold (x + i) (y + j) then 1 else 0))

/-- Checks if the board satisfies the conditions for all 3x3 squares and 2x4/4x2 rectangles. -/
def valid_board (b : Board) (A Z : ℕ) : Prop :=
  (∀ x y, x + 3 ≤ b.size → y + 3 ≤ b.size →
    count_gold b x y 3 3 = A) ∧
  (∀ x y, x + 2 ≤ b.size → y + 4 ≤ b.size →
    count_gold b x y 2 4 = Z) ∧
  (∀ x y, x + 4 ≤ b.size → y + 2 ≤ b.size →
    count_gold b x y 4 2 = Z)

theorem board_theorem :
  ∀ b : Board, b.size = 2016 →
    (∃ A Z, valid_board b A Z) →
    (∃ A Z, valid_board b A Z ∧ ((A = 0 ∧ Z = 0) ∨ (A = 9 ∧ Z = 8))) :=
sorry

end board_theorem_l3419_341961


namespace pirate_treasure_probability_l3419_341972

def num_islands : ℕ := 8
def num_treasure_islands : ℕ := 4
def prob_treasure : ℚ := 1/3
def prob_trap : ℚ := 1/6
def prob_neither : ℚ := 1/2

theorem pirate_treasure_probability :
  (Nat.choose num_islands num_treasure_islands : ℚ) *
  prob_treasure ^ num_treasure_islands *
  prob_neither ^ (num_islands - num_treasure_islands) =
  35/648 := by sorry

end pirate_treasure_probability_l3419_341972


namespace f_has_zero_in_interval_l3419_341966

-- Define the function g
variable (g : ℝ → ℝ)

-- Define the function f in terms of g
def f (x : ℝ) : ℝ := (x^2 - 3*x + 2) * g x + 3*x - 4

-- State the theorem
theorem f_has_zero_in_interval (hg : Continuous g) :
  ∃ c ∈ Set.Ioo 1 2, f g c = 0 := by
  sorry


end f_has_zero_in_interval_l3419_341966


namespace books_count_l3419_341946

/-- The number of books Jason has -/
def jason_books : ℕ := 18

/-- The number of books Mary has -/
def mary_books : ℕ := 42

/-- The total number of books Jason and Mary have together -/
def total_books : ℕ := jason_books + mary_books

theorem books_count : total_books = 60 := by
  sorry

end books_count_l3419_341946
