import Mathlib

namespace p_sufficient_not_necessary_for_q_l4049_404999

theorem p_sufficient_not_necessary_for_q 
  (h1 : p → q) 
  (h2 : ¬(¬p → ¬q)) : 
  (∃ (x : Prop), x → q) ∧ (∃ (y : Prop), q ∧ ¬y) :=
sorry

end p_sufficient_not_necessary_for_q_l4049_404999


namespace restaurant_bill_theorem_l4049_404959

theorem restaurant_bill_theorem (num_people : ℕ) (total_bill : ℚ) (gratuity_rate : ℚ) :
  num_people = 7 →
  total_bill = 840 →
  gratuity_rate = 1/5 →
  (total_bill / (1 + gratuity_rate)) / num_people = 100 := by
  sorry

end restaurant_bill_theorem_l4049_404959


namespace min_distance_complex_circle_l4049_404913

theorem min_distance_complex_circle (z : ℂ) (h : Complex.abs (z + 2 - 2*I) = 1) :
  ∃ (min_val : ℝ), min_val = 3 ∧ ∀ w : ℂ, Complex.abs (w + 2 - 2*I) = 1 → Complex.abs (w - 2 - 2*I) ≥ min_val :=
by
  sorry

end min_distance_complex_circle_l4049_404913


namespace line_parabola_intersection_l4049_404942

/-- A line with equation y = kx + 1 -/
structure Line (k : ℝ) where
  eq : ℝ → ℝ
  h : ∀ x, eq x = k * x + 1

/-- A parabola with equation y² = 4x -/
structure Parabola where
  eq : ℝ → ℝ → Prop
  h : ∀ x y, eq x y ↔ y^2 = 4*x

/-- The number of intersection points between a line and a parabola -/
def intersectionCount (l : Line k) (p : Parabola) : ℕ :=
  sorry

theorem line_parabola_intersection (k : ℝ) (l : Line k) (p : Parabola) :
  intersectionCount l p = 1 → k = 0 ∨ k = 1 :=
sorry

end line_parabola_intersection_l4049_404942


namespace club_leadership_selection_l4049_404945

def total_members : ℕ := 24
def num_boys : ℕ := 14
def num_girls : ℕ := 10

theorem club_leadership_selection :
  (num_boys * (num_boys - 1) + num_girls * (num_girls - 1) : ℕ) = 272 :=
by sorry

end club_leadership_selection_l4049_404945


namespace can_guess_majority_winners_l4049_404910

/-- Represents a tennis tournament with n players -/
structure TennisTournament (n : ℕ) where
  /-- Final scores of each player -/
  scores : Fin n → ℕ
  /-- Total number of matches in the tournament -/
  total_matches : ℕ := n * (n - 1) / 2

/-- Theorem stating that it's possible to guess more than half of the match winners -/
theorem can_guess_majority_winners (n : ℕ) (tournament : TennisTournament n) :
  ∃ (guessed_matches : ℕ), guessed_matches > tournament.total_matches / 2 :=
sorry

end can_guess_majority_winners_l4049_404910


namespace board_9x16_fills_12x12_square_l4049_404967

/-- Represents a rectangular board with integer dimensions -/
structure Board where
  width : ℕ
  length : ℕ

/-- Represents a square hole with integer side length -/
structure Square where
  side : ℕ

/-- Checks if a board can be cut to fill a square hole using the staircase method -/
def canFillSquare (b : Board) (s : Square) : Prop :=
  ∃ (steps : ℕ), 
    steps > 0 ∧
    b.width * (steps + 1) = s.side ∧
    b.length * steps = s.side

theorem board_9x16_fills_12x12_square :
  canFillSquare (Board.mk 9 16) (Square.mk 12) :=
sorry

end board_9x16_fills_12x12_square_l4049_404967


namespace l_shaped_area_specific_l4049_404935

/-- Calculates the area of an 'L'-shaped figure formed by removing a smaller rectangle from a larger rectangle. -/
def l_shaped_area (large_width large_height small_width small_height : ℕ) : ℕ :=
  large_width * large_height - small_width * small_height

theorem l_shaped_area_specific : l_shaped_area 10 7 4 3 = 58 := by
  sorry

end l_shaped_area_specific_l4049_404935


namespace meeting_organization_count_l4049_404932

/-- The number of ways to organize a leadership meeting -/
def organize_meeting (total_schools : ℕ) (members_per_school : ℕ) 
  (host_representatives : ℕ) (other_representatives : ℕ) : ℕ :=
  total_schools * (members_per_school.choose host_representatives) * 
  ((members_per_school.choose other_representatives) ^ (total_schools - 1))

/-- Theorem stating the number of ways to organize the meeting -/
theorem meeting_organization_count :
  organize_meeting 4 6 3 1 = 17280 := by
  sorry

end meeting_organization_count_l4049_404932


namespace inequality_system_solution_l4049_404985

theorem inequality_system_solution (m : ℝ) : 
  (∀ x : ℝ, (x - m > 0 ∧ 2*x + 1 > 3) ↔ x > 1) → m ≤ 1 := by
sorry

end inequality_system_solution_l4049_404985


namespace min_skew_edge_distance_l4049_404929

/-- A regular octahedron with edge length a -/
structure RegularOctahedron (a : ℝ) where
  edge_length : a > 0

/-- A point on an edge of the octahedron -/
structure EdgePoint (O : RegularOctahedron a) where
  -- Additional properties can be added if needed

/-- The distance between two points on skew edges of the octahedron -/
def skew_edge_distance (O : RegularOctahedron a) (p q : EdgePoint O) : ℝ := sorry

/-- The theorem stating the minimal distance between points on skew edges -/
theorem min_skew_edge_distance (a : ℝ) (O : RegularOctahedron a) :
  ∃ (p q : EdgePoint O), 
    skew_edge_distance O p q = a * Real.sqrt 6 / 3 ∧
    ∀ (r s : EdgePoint O), skew_edge_distance O r s ≥ a * Real.sqrt 6 / 3 := by
  sorry

end min_skew_edge_distance_l4049_404929


namespace notebook_increase_l4049_404911

theorem notebook_increase (initial_count mother_bought father_bought : ℕ) :
  initial_count = 33 →
  mother_bought = 7 →
  father_bought = 14 →
  (initial_count + mother_bought + father_bought) - initial_count = 21 := by
  sorry

end notebook_increase_l4049_404911


namespace r_amount_l4049_404955

def total_amount : ℕ := 1210
def num_persons : ℕ := 3

def ratio_p_q : Rat := 5 / 4
def ratio_q_r : Rat := 9 / 10

theorem r_amount (p q r : ℕ) (h1 : p + q + r = total_amount) 
  (h2 : (p : ℚ) / q = ratio_p_q) (h3 : (q : ℚ) / r = ratio_q_r) : r = 400 := by
  sorry

end r_amount_l4049_404955


namespace evaluate_expression_l4049_404936

theorem evaluate_expression : 2010^3 - 2009 * 2010^2 - 2009^2 * 2010 + 2009^3 = 4019 := by
  sorry

end evaluate_expression_l4049_404936


namespace b_finishes_in_two_days_l4049_404981

/-- The number of days A takes to finish the work alone -/
def a_days : ℚ := 4

/-- The number of days B takes to finish the work alone -/
def b_days : ℚ := 8

/-- The number of days A and B work together before A leaves -/
def days_together : ℚ := 2

/-- The fraction of work completed per day when A and B work together -/
def combined_work_rate : ℚ := 1 / a_days + 1 / b_days

/-- The fraction of work completed when A and B work together for 2 days -/
def work_completed_together : ℚ := days_together * combined_work_rate

/-- The fraction of work remaining after A leaves -/
def remaining_work : ℚ := 1 - work_completed_together

/-- The number of days B takes to finish the remaining work alone -/
def days_for_b_to_finish : ℚ := remaining_work / (1 / b_days)

theorem b_finishes_in_two_days : days_for_b_to_finish = 2 := by
  sorry

end b_finishes_in_two_days_l4049_404981


namespace intersection_in_first_quadrant_implies_a_greater_than_two_l4049_404962

/-- Two lines intersect in the first quadrant implies a > 2 -/
theorem intersection_in_first_quadrant_implies_a_greater_than_two 
  (a : ℝ) 
  (l₁ : ℝ → ℝ → Prop) 
  (l₂ : ℝ → ℝ → Prop) 
  (h₁ : ∀ x y, l₁ x y ↔ a * x - y + 2 = 0)
  (h₂ : ∀ x y, l₂ x y ↔ x + y - a = 0)
  (h_intersection : ∃ x y, l₁ x y ∧ l₂ x y ∧ x > 0 ∧ y > 0) : 
  a > 2 := by
sorry


end intersection_in_first_quadrant_implies_a_greater_than_two_l4049_404962


namespace average_growth_rate_inequality_l4049_404931

theorem average_growth_rate_inequality (a p q x : ℝ) 
  (h1 : a > 0) 
  (h2 : p ≥ 0) 
  (h3 : q ≥ 0) 
  (h4 : a * (1 + p) * (1 + q) = a * (1 + x)^2) : 
  x ≤ (p + q) / 2 := by
sorry

end average_growth_rate_inequality_l4049_404931


namespace diameter_angle_property_l4049_404921

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point
def Point := ℝ × ℝ

-- Define a function to check if a point is inside a circle
def isInside (c : Circle) (p : Point) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 < c.radius^2

-- Define a function to check if a point is outside a circle
def isOutside (c : Circle) (p : Point) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 > c.radius^2

-- Define a function to calculate the angle between three points
noncomputable def angle (a b c : Point) : ℝ := sorry

-- Theorem statement
theorem diameter_angle_property (c : Circle) (a b : Point) :
  (∀ (x : ℝ × ℝ), x.1 = a.1 ∧ x.2 = b.2 → isInside c x) →  -- a and b are on opposite sides of the circle
  (∀ (p : Point), isInside c p → angle a p b > Real.pi / 2) ∧
  (∀ (p : Point), isOutside c p → angle a p b < Real.pi / 2) :=
by sorry

end diameter_angle_property_l4049_404921


namespace polynomial_roots_l4049_404917

theorem polynomial_roots : ∃ (a b c d : ℂ),
  (a = (1 + Real.sqrt 5) / 2) ∧
  (b = (1 - Real.sqrt 5) / 2) ∧
  (c = (3 + Real.sqrt 13) / 6) ∧
  (d = (3 - Real.sqrt 13) / 6) ∧
  (∀ x : ℂ, 3 * x^4 - 4 * x^3 - 5 * x^2 - 4 * x + 3 = 0 ↔ (x = a ∨ x = b ∨ x = c ∨ x = d)) :=
by sorry

end polynomial_roots_l4049_404917


namespace christmas_tree_lights_l4049_404920

/-- The number of yellow lights on a Christmas tree -/
def yellow_lights (total red blue : ℕ) : ℕ := total - (red + blue)

/-- Theorem: There are 37 yellow lights on the Christmas tree -/
theorem christmas_tree_lights : yellow_lights 95 26 32 = 37 := by
  sorry

end christmas_tree_lights_l4049_404920


namespace jared_yearly_income_l4049_404998

/-- Calculates the yearly income of a degree holder after one year of employment --/
def yearly_income_after_one_year (diploma_salary : ℝ) : ℝ :=
  let degree_salary := 3 * diploma_salary
  let annual_salary := 12 * degree_salary
  let salary_after_raise := annual_salary * 1.05
  salary_after_raise * 1.05

/-- Theorem stating that Jared's yearly income after one year is $158760 --/
theorem jared_yearly_income :
  yearly_income_after_one_year 4000 = 158760 := by
  sorry

#eval yearly_income_after_one_year 4000

end jared_yearly_income_l4049_404998


namespace polynomial_identity_l4049_404925

theorem polynomial_identity (P : ℝ → ℝ) : 
  (∀ a b c : ℝ, a * b + b * c + c * a = 0 → 
    P (a - b) + P (b - c) + P (c - a) = 2 * P (a + b + c)) → 
  ∃ α β : ℝ, ∀ x : ℝ, P x = α * x^4 + β * x^2 := by
sorry

end polynomial_identity_l4049_404925


namespace min_value_theorem_equality_exists_l4049_404986

theorem min_value_theorem (x : ℝ) (h : x > 0) : x^2 + 12*x + 108/x^4 ≥ 42 := by
  sorry

theorem equality_exists : ∃ x : ℝ, x > 0 ∧ x^2 + 12*x + 108/x^4 = 42 := by
  sorry

end min_value_theorem_equality_exists_l4049_404986


namespace negation_of_proposition_negation_of_greater_than_sin_l4049_404966

theorem negation_of_proposition (p : ℝ → Prop) : 
  (¬ ∀ x : ℝ, p x) ↔ (∃ x : ℝ, ¬ p x) :=
by sorry

theorem negation_of_greater_than_sin :
  (¬ ∀ x : ℝ, x > Real.sin x) ↔ (∃ x : ℝ, x ≤ Real.sin x) :=
by sorry

end negation_of_proposition_negation_of_greater_than_sin_l4049_404966


namespace system_solution_l4049_404927

theorem system_solution : 
  let x : ℚ := 57 / 31
  let y : ℚ := 97 / 31
  (3 * x - 4 * y = -7) ∧ (4 * x + 5 * y = 23) := by
  sorry

end system_solution_l4049_404927


namespace opposite_sides_range_l4049_404987

def line_equation (x y a : ℝ) : ℝ := x - 2*y + a

theorem opposite_sides_range (a : ℝ) : 
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁ = 2 ∧ y₁ = -1 ∧ x₂ = -3 ∧ y₂ = 2 ∧ 
    (line_equation x₁ y₁ a) * (line_equation x₂ y₂ a) < 0) ↔ 
  -4 < a ∧ a < 7 :=
sorry

end opposite_sides_range_l4049_404987


namespace chosen_number_proof_l4049_404965

theorem chosen_number_proof : ∃ x : ℚ, (x / 8 : ℚ) - 100 = 6 ∧ x = 848 := by
  sorry

end chosen_number_proof_l4049_404965


namespace joker_spade_probability_l4049_404996

/-- Custom deck of cards -/
structure CustomDeck :=
  (total_cards : ℕ)
  (ranks : ℕ)
  (suits : ℕ)
  (jokers : ℕ)
  (cards_per_suit : ℕ)

/-- Properties of the custom deck -/
def custom_deck_properties (d : CustomDeck) : Prop :=
  d.total_cards = 60 ∧
  d.ranks = 15 ∧
  d.suits = 4 ∧
  d.jokers = 4 ∧
  d.cards_per_suit = 15

/-- Probability of drawing a Joker first and any spade second -/
def joker_spade_prob (d : CustomDeck) : ℚ :=
  224 / 885

/-- Theorem stating the probability of drawing a Joker first and any spade second -/
theorem joker_spade_probability (d : CustomDeck) 
  (h : custom_deck_properties d) : 
  joker_spade_prob d = 224 / 885 := by
  sorry

end joker_spade_probability_l4049_404996


namespace x_fourth_power_zero_l4049_404916

theorem x_fourth_power_zero (x : ℝ) (h_pos : x > 0) 
  (h_eq : Real.sqrt (1 - x^2) + Real.sqrt (1 + x^2) = 2) : x^4 = 0 := by
  sorry

end x_fourth_power_zero_l4049_404916


namespace min_value_theorem_l4049_404976

theorem min_value_theorem (x : ℝ) (h : x > 0) :
  4 * x + 1 / x^6 ≥ 5 ∧ ∃ y : ℝ, y > 0 ∧ 4 * y + 1 / y^6 = 5 := by
  sorry

end min_value_theorem_l4049_404976


namespace perfect_square_sum_l4049_404977

theorem perfect_square_sum (a b c d : ℤ) (h : a + b + c + d = 0) :
  2 * (a^4 + b^4 + c^4 + d^4) + 8 * a * b * c * d = (a^2 + b^2 + c^2 + d^2)^2 := by
  sorry

end perfect_square_sum_l4049_404977


namespace line_perpendicular_to_plane_l4049_404901

/-- Given a point P₀ and a plane, this theorem states that the line passing through P₀ 
    and perpendicular to the plane has a specific equation. -/
theorem line_perpendicular_to_plane 
  (P₀ : ℝ × ℝ × ℝ) 
  (plane_normal : ℝ × ℝ × ℝ) 
  (plane_constant : ℝ) :
  let (x₀, y₀, z₀) := P₀
  let (a, b, c) := plane_normal
  (P₀ = (3, 4, 2) ∧ 
   plane_normal = (8, -4, 5) ∧ 
   plane_constant = -4) →
  (∀ (x y z : ℝ), 
    ((x - x₀) / a = (y - y₀) / b ∧ (y - y₀) / b = (z - z₀) / c) ↔
    (x, y, z) ∈ {p : ℝ × ℝ × ℝ | ∃ t : ℝ, p = (x₀ + a*t, y₀ + b*t, z₀ + c*t)}) :=
by sorry

end line_perpendicular_to_plane_l4049_404901


namespace orange_probability_is_two_sevenths_l4049_404937

/-- Represents the contents of a fruit basket -/
structure FruitBasket where
  oranges : ℕ
  apples : ℕ
  bananas : ℕ

/-- The initial state of the fruit basket -/
def initialBasket : FruitBasket := sorry

/-- The state of the basket after removing some fruits -/
def updatedBasket : FruitBasket := sorry

/-- The total number of fruits in the initial basket -/
def totalFruits : ℕ := 28

/-- Assertion that the updated basket has 3 oranges and 7 apples -/
axiom updated_basket_state : updatedBasket.oranges = 3 ∧ updatedBasket.apples = 7

/-- Assertion that 5 oranges and 3 apples were removed -/
axiom fruits_removed : initialBasket.oranges = updatedBasket.oranges + 5 ∧
                       initialBasket.apples = updatedBasket.apples + 3

/-- Assertion that the total number of fruits in the initial basket is correct -/
axiom initial_total_correct : initialBasket.oranges + initialBasket.apples + initialBasket.bananas = totalFruits

/-- The probability of choosing an orange from the initial basket -/
def orangeProbability : ℚ := sorry

/-- Theorem stating that the probability of choosing an orange is 2/7 -/
theorem orange_probability_is_two_sevenths : orangeProbability = 2 / 7 := by sorry

end orange_probability_is_two_sevenths_l4049_404937


namespace alberts_to_angelas_marbles_ratio_l4049_404900

/-- Proves that the ratio of Albert's marbles to Angela's marbles is 3:1 -/
theorem alberts_to_angelas_marbles_ratio (allison_marbles : ℕ) (angela_more_than_allison : ℕ) 
  (albert_and_allison_total : ℕ) 
  (h1 : allison_marbles = 28)
  (h2 : angela_more_than_allison = 8)
  (h3 : albert_and_allison_total = 136) : 
  (albert_and_allison_total - allison_marbles) / (allison_marbles + angela_more_than_allison) = 3 := by
  sorry

#check alberts_to_angelas_marbles_ratio

end alberts_to_angelas_marbles_ratio_l4049_404900


namespace rohans_savings_l4049_404905

/-- Rohan's monthly savings calculation -/
theorem rohans_savings (salary : ℝ) (food_percent : ℝ) (rent_percent : ℝ) 
  (entertainment_percent : ℝ) (conveyance_percent : ℝ) : 
  salary = 12500 ∧ 
  food_percent = 40 ∧ 
  rent_percent = 20 ∧ 
  entertainment_percent = 10 ∧ 
  conveyance_percent = 10 → 
  salary * (1 - (food_percent + rent_percent + entertainment_percent + conveyance_percent) / 100) = 2500 :=
by sorry

end rohans_savings_l4049_404905


namespace power_expression_evaluation_l4049_404973

theorem power_expression_evaluation (b : ℕ) (h : b = 2) : b^3 * b^4 = 128 := by
  sorry

end power_expression_evaluation_l4049_404973


namespace cube_surface_area_increase_l4049_404988

/-- Proves that a cube with edge length 7 cm, when cut into 1 cm cubes, results in a 600% increase in surface area. -/
theorem cube_surface_area_increase : 
  let original_edge_length : ℝ := 7
  let original_surface_area : ℝ := 6 * original_edge_length^2
  let new_surface_area : ℝ := 6 * original_edge_length^3
  new_surface_area = 7 * original_surface_area := by
  sorry

#check cube_surface_area_increase

end cube_surface_area_increase_l4049_404988


namespace stair_climbing_time_l4049_404950

theorem stair_climbing_time (a₁ : ℕ) (d : ℕ) (n : ℕ) (h1 : a₁ = 30) (h2 : d = 7) (h3 : n = 8) :
  n * (2 * a₁ + (n - 1) * d) / 2 = 436 := by
  sorry

end stair_climbing_time_l4049_404950


namespace license_plate_count_l4049_404969

def alphabet : ℕ := 26
def vowels : ℕ := 7  -- A, E, I, O, U, W, Y
def consonants : ℕ := alphabet - vowels
def even_digits : ℕ := 5  -- 0, 2, 4, 6, 8

def license_plate_combinations : ℕ := consonants * vowels * consonants * even_digits

theorem license_plate_count : license_plate_combinations = 12565 := by
  sorry

end license_plate_count_l4049_404969


namespace carolyn_practice_time_l4049_404982

/-- Calculates the total practice time in minutes for a month given daily practice times and schedule -/
def monthly_practice_time (piano_time : ℕ) (violin_multiplier : ℕ) (days_per_week : ℕ) (weeks_per_month : ℕ) : ℕ :=
  let violin_time := piano_time * violin_multiplier
  let daily_total := piano_time + violin_time
  let weekly_total := daily_total * days_per_week
  weekly_total * weeks_per_month

/-- Proves that Carolyn's monthly practice time is 1920 minutes -/
theorem carolyn_practice_time :
  monthly_practice_time 20 3 6 4 = 1920 := by
  sorry

end carolyn_practice_time_l4049_404982


namespace quadratic_coefficient_positive_l4049_404906

/-- A quadratic polynomial with integer coefficients -/
structure QuadraticPolynomial where
  a : ℤ
  b : ℤ
  c : ℤ
  a_nonzero : a ≠ 0

/-- Evaluation of a quadratic polynomial at a given point -/
def evaluate (p : QuadraticPolynomial) (x : ℤ) : ℤ :=
  p.a * x * x + p.b * x + p.c

/-- The main theorem -/
theorem quadratic_coefficient_positive
  (p : QuadraticPolynomial)
  (n : ℤ)
  (h : n < evaluate p n ∧ evaluate p n < evaluate p (evaluate p n) ∧
       evaluate p (evaluate p n) < evaluate p (evaluate p (evaluate p n))) :
  0 < p.a :=
sorry

end quadratic_coefficient_positive_l4049_404906


namespace chef_apples_left_l4049_404902

/-- The number of apples the chef has left after making a pie -/
def applesLeft (initialApples usedApples : ℕ) : ℕ :=
  initialApples - usedApples

theorem chef_apples_left : applesLeft 19 15 = 4 := by
  sorry

end chef_apples_left_l4049_404902


namespace perpendicular_line_through_intersection_l4049_404903

/-- The intersection point of two lines -/
def intersection_point (l1 l2 : ℝ → ℝ → Prop) : ℝ × ℝ := sorry

/-- Check if a point satisfies a line equation -/
def satisfies_line (p : ℝ × ℝ) (l : ℝ → ℝ → Prop) : Prop := sorry

/-- Check if two lines are perpendicular -/
def perpendicular (l1 l2 : ℝ → ℝ → Prop) : Prop := sorry

/-- The main theorem -/
theorem perpendicular_line_through_intersection :
  let l1 : ℝ → ℝ → Prop := λ x y => 2*x - y = 0
  let l2 : ℝ → ℝ → Prop := λ x y => x + y - 6 = 0
  let l3 : ℝ → ℝ → Prop := λ x y => 2*x + y - 1 = 0
  let l4 : ℝ → ℝ → Prop := λ x y => x - 2*y + 6 = 0
  let p := intersection_point l1 l2
  satisfies_line p l4 ∧ perpendicular l3 l4 := by sorry

end perpendicular_line_through_intersection_l4049_404903


namespace max_profit_at_price_l4049_404953

/-- Represents the daily sales and profit model of a store --/
structure StoreModel where
  cost_price : ℝ
  max_price_factor : ℝ
  sales_function : ℝ → ℝ
  profit_function : ℝ → ℝ

/-- The store model satisfies the given conditions --/
def satisfies_conditions (model : StoreModel) : Prop :=
  model.cost_price = 100 ∧
  model.max_price_factor = 1.4 ∧
  model.sales_function 130 = 140 ∧
  model.sales_function 140 = 120 ∧
  (∀ x, model.sales_function x = -2 * x + 400) ∧
  (∀ x, model.profit_function x = (x - model.cost_price) * model.sales_function x)

/-- The maximum profit occurs at the given price and value --/
theorem max_profit_at_price (model : StoreModel) 
    (h : satisfies_conditions model) :
    (∀ x, x ≤ model.max_price_factor * model.cost_price → 
      model.profit_function x ≤ model.profit_function 140) ∧
    model.profit_function 140 = 4800 := by
  sorry

#check max_profit_at_price

end max_profit_at_price_l4049_404953


namespace tape_overlap_length_l4049_404926

theorem tape_overlap_length 
  (num_pieces : ℕ) 
  (piece_length : ℝ) 
  (total_overlapped_length : ℝ) 
  (h1 : num_pieces = 4) 
  (h2 : piece_length = 250) 
  (h3 : total_overlapped_length = 925) :
  (num_pieces * piece_length - total_overlapped_length) / (num_pieces - 1) = 25 := by
sorry

end tape_overlap_length_l4049_404926


namespace food_duration_l4049_404979

/-- The number of days the food was initially meant to last -/
def initial_days : ℝ := 22

/-- The initial number of men -/
def initial_men : ℝ := 760

/-- The number of men that join after two days -/
def additional_men : ℝ := 134.11764705882354

/-- The number of days the food lasts after the additional men join -/
def remaining_days : ℝ := 17

/-- The total number of men after the additional men join -/
def total_men : ℝ := initial_men + additional_men

theorem food_duration :
  initial_men * (initial_days - 2) = total_men * remaining_days :=
sorry

end food_duration_l4049_404979


namespace area_of_quadrilateral_PAQR_l4049_404968

-- Define the points
variable (P A Q R B : ℝ × ℝ)

-- Define the distances
def AP : ℝ := 10
def PB : ℝ := 20
def PR : ℝ := 25

-- Define the right triangles
def is_right_triangle (X Y Z : ℝ × ℝ) : Prop :=
  (X.1 - Y.1) * (X.1 - Z.1) + (X.2 - Y.2) * (X.2 - Z.2) = 0

-- State the theorem
theorem area_of_quadrilateral_PAQR :
  is_right_triangle P A Q →
  is_right_triangle P B R →
  (let area_PAQ := (1/2) * ‖A - P‖ * ‖Q - P‖;
   let area_PBR := (1/2) * ‖B - P‖ * ‖R - B‖;
   area_PAQ + area_PBR = 174) :=
by sorry

end area_of_quadrilateral_PAQR_l4049_404968


namespace sphere_cube_intersection_areas_l4049_404915

/-- Given a cube with edge length a and a sphere circumscribed around it, 
    this theorem proves the areas of the sections formed by the intersection 
    of the sphere and the cube's faces. -/
theorem sphere_cube_intersection_areas (a : ℝ) (h : a > 0) :
  let R := a * Real.sqrt 3 / 2
  ∃ (bicorn_area curvilinear_quad_area : ℝ),
    bicorn_area = π * a^2 * (2 - Real.sqrt 3) / 4 ∧
    curvilinear_quad_area = π * a^2 * (Real.sqrt 3 - 1) / 2 :=
by sorry

end sphere_cube_intersection_areas_l4049_404915


namespace concatenation_product_relation_l4049_404946

theorem concatenation_product_relation :
  ∃! (x y : ℕ), 10 ≤ x ∧ x ≤ 99 ∧ 100 ≤ y ∧ y ≤ 999 ∧ 
  1000 * x + y = 11 * x * y ∧ x + y = 110 := by
sorry

end concatenation_product_relation_l4049_404946


namespace rhombus_rectangle_diagonals_bisect_l4049_404940

-- Define a quadrilateral
class Quadrilateral :=
  (diagonals_bisect : Bool)
  (diagonals_perpendicular : Bool)
  (diagonals_equal : Bool)

-- Define a rhombus
def Rhombus : Quadrilateral :=
{ diagonals_bisect := true,
  diagonals_perpendicular := true,
  diagonals_equal := false }

-- Define a rectangle
def Rectangle : Quadrilateral :=
{ diagonals_bisect := true,
  diagonals_perpendicular := false,
  diagonals_equal := true }

-- Theorem: Both rhombuses and rectangles have diagonals that bisect each other
theorem rhombus_rectangle_diagonals_bisect :
  Rhombus.diagonals_bisect ∧ Rectangle.diagonals_bisect :=
sorry

end rhombus_rectangle_diagonals_bisect_l4049_404940


namespace arithmetic_sequence_sum_l4049_404984

theorem arithmetic_sequence_sum (a₁ d n : ℝ) (S_n : ℕ → ℝ) : 
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    y₁ = a₁ * x₁ ∧ 
    y₂ = a₁ * x₂ ∧ 
    (x₁ - 2)^2 + y₁^2 = 1 ∧ 
    (x₂ - 2)^2 + y₂^2 = 1 ∧
    (x₁ + y₁ + d) = -(x₂ + y₂ + d)) →
  (∀ k : ℕ, S_n k = k * a₁ + k * (k - 1) / 2 * d) →
  ∀ k : ℕ, S_n k = 2*k - k^2 := by
sorry

end arithmetic_sequence_sum_l4049_404984


namespace car_repair_cost_l4049_404951

/-- Proves that the repair cost is approximately 13000, given the initial cost,
    selling price, and profit percentage of a car sale. -/
theorem car_repair_cost (initial_cost selling_price : ℕ) (profit_percentage : ℚ) :
  initial_cost = 42000 →
  selling_price = 66900 →
  profit_percentage = 21636363636363637 / 100000000000000 →
  ∃ (repair_cost : ℕ), 
    (repair_cost ≥ 12999 ∧ repair_cost ≤ 13001) ∧
    profit_percentage = (selling_price - (initial_cost + repair_cost)) / (initial_cost + repair_cost) :=
by sorry

end car_repair_cost_l4049_404951


namespace circumcircle_of_triangle_ABP_l4049_404983

-- Define the given circle
def given_circle (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the external point P
def point_P : ℝ × ℝ := (4, 2)

-- Define the property of A and B being points of tangency
def tangent_points (A B : ℝ × ℝ) : Prop :=
  given_circle A.1 A.2 ∧ given_circle B.1 B.2 ∧
  (∀ t : ℝ, t ≠ 0 → ¬(given_circle (A.1 + t * (point_P.1 - A.1)) (A.2 + t * (point_P.2 - A.2)))) ∧
  (∀ t : ℝ, t ≠ 0 → ¬(given_circle (B.1 + t * (point_P.1 - B.1)) (B.2 + t * (point_P.2 - B.2))))

-- Define the equation of the circumcircle
def circumcircle_equation (x y : ℝ) : Prop := (x - 4)^2 + (y - 2)^2 = 16

-- Theorem statement
theorem circumcircle_of_triangle_ABP :
  ∀ A B : ℝ × ℝ, tangent_points A B →
  ∀ x y : ℝ, (x - A.1)^2 + (y - A.2)^2 = (x - B.1)^2 + (y - B.2)^2 →
  (x - point_P.1)^2 + (y - point_P.2)^2 = (x - A.1)^2 + (y - A.2)^2 →
  circumcircle_equation x y :=
sorry

end circumcircle_of_triangle_ABP_l4049_404983


namespace existential_and_true_proposition_l4049_404908

theorem existential_and_true_proposition :
  (∃ a : ℕ, a^2 + a ≤ 0) ∧
  (∃ a : ℕ, a^2 + a ≤ 0) = True :=
by sorry

end existential_and_true_proposition_l4049_404908


namespace point_on_graph_l4049_404991

def f (x : ℝ) : ℝ := |x^3 + 1| + |x^3 - 1|

theorem point_on_graph (a : ℝ) : (a, f (-a)) ∈ {p : ℝ × ℝ | p.2 = f p.1} := by
  sorry

end point_on_graph_l4049_404991


namespace polynomial_composition_problem_l4049_404933

-- Define the polynomial P
def P (x : ℝ) : ℝ := x^2 - 1

-- Define the theorem
theorem polynomial_composition_problem (a : ℝ) (m n : ℕ) 
  (h1 : a > 0)
  (h2 : P (P (P a)) = 99)
  (h3 : a^2 = m + Real.sqrt n)
  (h4 : n > 0)
  (h5 : ∀ (p : ℕ), Nat.Prime p → ¬(p^2 ∣ n)) :
  m + n = 12 := by
sorry

end polynomial_composition_problem_l4049_404933


namespace smallest_nonneg_minus_opposite_largest_neg_l4049_404943

theorem smallest_nonneg_minus_opposite_largest_neg : ∃ a b : ℤ,
  (∀ x : ℤ, x ≥ 0 → a ≤ x) ∧
  (∀ y : ℤ, y < 0 → y ≤ -b) ∧
  (a - b = 1) := by
  sorry

end smallest_nonneg_minus_opposite_largest_neg_l4049_404943


namespace original_price_after_percentage_changes_l4049_404923

theorem original_price_after_percentage_changes (p : ℝ) :
  let initial_price := (10000 : ℝ) / (10000 - p^2)
  let price_after_increase := initial_price * (1 + p / 100)
  let final_price := price_after_increase * (1 - p / 100)
  final_price = 1 :=
by sorry

end original_price_after_percentage_changes_l4049_404923


namespace books_on_shelf_l4049_404939

/-- The number of books remaining on a shelf after some are removed. -/
def booksRemaining (initial : ℝ) (removed : ℝ) : ℝ :=
  initial - removed

theorem books_on_shelf (initial : ℝ) (removed : ℝ) :
  initial ≥ removed →
  booksRemaining initial removed = initial - removed :=
by
  sorry

end books_on_shelf_l4049_404939


namespace towel_shrinkage_l4049_404971

theorem towel_shrinkage (original_length original_breadth : ℝ) 
  (original_length_pos : 0 < original_length) 
  (original_breadth_pos : 0 < original_breadth) : 
  let new_length := 0.7 * original_length
  let new_area := 0.595 * (original_length * original_breadth)
  ∃ new_breadth : ℝ, 
    new_breadth = 0.85 * original_breadth ∧ 
    new_area = new_length * new_breadth :=
by sorry

end towel_shrinkage_l4049_404971


namespace cashew_price_in_mixture_l4049_404914

/-- The price per pound of cashews in a mixture with peanuts -/
def cashew_price (peanut_price : ℚ) (total_weight : ℚ) (total_value : ℚ) (cashew_weight : ℚ) : ℚ :=
  (total_value - (total_weight - cashew_weight) * peanut_price) / cashew_weight

/-- Theorem stating the price of cashews in the given mixture -/
theorem cashew_price_in_mixture :
  cashew_price 2 25 92 11 = 64/11 := by
  sorry

end cashew_price_in_mixture_l4049_404914


namespace even_odd_solution_l4049_404997

theorem even_odd_solution (m n p q : ℤ) 
  (h_m_odd : Odd m)
  (h_n_even : Even n)
  (h_eq1 : p - 1998*q = n)
  (h_eq2 : 1999*p + 3*q = m) :
  Even p ∧ Odd q := by
sorry

end even_odd_solution_l4049_404997


namespace ab_nonpositive_l4049_404924

theorem ab_nonpositive (a b : ℝ) (h : 2011 * a + 2012 * b = 0) : a * b ≤ 0 := by
  sorry

end ab_nonpositive_l4049_404924


namespace set_condition_implies_range_l4049_404954

theorem set_condition_implies_range (a : ℝ) : 
  let A := {x : ℝ | x > 5}
  let B := {x : ℝ | x > a}
  (∀ x, x ∈ A → x ∈ B) ∧ (∃ x, x ∈ B ∧ x ∉ A) → a < 5 := by
  sorry

end set_condition_implies_range_l4049_404954


namespace chicken_count_l4049_404904

theorem chicken_count (total_chickens : ℕ) (hens : ℕ) (roosters : ℕ) (chicks : ℕ) : 
  total_chickens = 15 → 
  hens = 3 → 
  roosters = total_chickens - hens → 
  chicks = roosters - 4 → 
  chicks = 8 := by
sorry

end chicken_count_l4049_404904


namespace last_digit_of_one_over_2_pow_12_l4049_404975

-- Define the function to get the last digit of a rational number's decimal expansion
noncomputable def lastDigitOfDecimalExpansion (q : ℚ) : ℕ :=
  sorry

-- Theorem statement
theorem last_digit_of_one_over_2_pow_12 :
  lastDigitOfDecimalExpansion (1 / 2^12) = 5 := by
  sorry

end last_digit_of_one_over_2_pow_12_l4049_404975


namespace cubic_equation_solution_range_l4049_404949

theorem cubic_equation_solution_range (m : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc 0 2 ∧ x^3 - 3*x + m = 0) → m ∈ Set.Icc (-2) 2 := by
  sorry

end cubic_equation_solution_range_l4049_404949


namespace lowest_common_denominator_l4049_404963

theorem lowest_common_denominator (a b c : Nat) : a = 9 → b = 4 → c = 18 → Nat.lcm a (Nat.lcm b c) = 36 := by
  sorry

end lowest_common_denominator_l4049_404963


namespace no_digit_product_6552_l4049_404941

theorem no_digit_product_6552 : ¬ ∃ (s : List ℕ), (∀ n ∈ s, n ≤ 9) ∧ s.prod = 6552 := by
  sorry

end no_digit_product_6552_l4049_404941


namespace placement_count_l4049_404995

/-- Represents a painting with width and height -/
structure Painting :=
  (width : Nat)
  (height : Nat)

/-- Represents a wall with width and height -/
structure Wall :=
  (width : Nat)
  (height : Nat)

/-- Represents the collection of paintings -/
def paintings : List Painting := [
  ⟨2, 1⟩,
  ⟨1, 1⟩, ⟨1, 1⟩,
  ⟨1, 2⟩, ⟨1, 2⟩,
  ⟨2, 2⟩, ⟨2, 2⟩,
  ⟨4, 3⟩, ⟨4, 3⟩,
  ⟨4, 4⟩, ⟨4, 4⟩
]

/-- The wall on which paintings are to be placed -/
def wall : Wall := ⟨12, 6⟩

/-- Function to calculate the number of ways to place paintings on the wall -/
def numberOfPlacements (w : Wall) (p : List Painting) : Nat :=
  sorry

/-- Theorem stating that the number of placements is 16896 -/
theorem placement_count : numberOfPlacements wall paintings = 16896 := by
  sorry

end placement_count_l4049_404995


namespace least_three_digit_multiple_of_11_l4049_404970

theorem least_three_digit_multiple_of_11 : ∃ n : ℕ, 
  n = 110 ∧ 
  n % 11 = 0 ∧ 
  100 ≤ n ∧ n ≤ 999 ∧ 
  ∀ m : ℕ, (m % 11 = 0 ∧ 100 ≤ m ∧ m ≤ 999) → n ≤ m :=
by sorry

end least_three_digit_multiple_of_11_l4049_404970


namespace sum_of_squares_l4049_404922

theorem sum_of_squares (x y : ℕ+) 
  (h1 : x * y + x + y = 83)
  (h2 : x * x * y + x * y * y = 1056) :
  x * x + y * y = 458 := by
  sorry

end sum_of_squares_l4049_404922


namespace perpendicular_line_equation_l4049_404961

/-- Given a line L1 with equation 3x - 4y + 6 = 0 and a point P(4, -1),
    the line L2 passing through P and perpendicular to L1 has equation 4x + 3y - 13 = 0 -/
theorem perpendicular_line_equation :
  let L1 : ℝ → ℝ → Prop := λ x y ↦ 3 * x - 4 * y + 6 = 0
  let P : ℝ × ℝ := (4, -1)
  let L2 : ℝ → ℝ → Prop := λ x y ↦ 4 * x + 3 * y - 13 = 0
  (∀ x y, L2 x y ↔ (y - P.2 = -(4/3) * (x - P.1))) ∧
  (∀ x₁ y₁ x₂ y₂, L1 x₁ y₁ → L2 x₂ y₂ → 
    ((x₂ - x₁) * (3/4) + (y₂ - y₁) * (-1) = 0)) :=
by sorry

end perpendicular_line_equation_l4049_404961


namespace largest_valid_p_l4049_404952

def is_valid_p (p : ℝ) : Prop :=
  p > 1 ∧ ∀ a b c : ℝ, 
    1/p ≤ a ∧ a ≤ p ∧
    1/p ≤ b ∧ b ≤ p ∧
    1/p ≤ c ∧ c ≤ p →
    9 * (a*b + b*c + c*a) * (a^2 + b^2 + c^2) ≥ (a + b + c)^4

theorem largest_valid_p :
  ∃ p : ℝ, p = Real.sqrt (4 + 3 * Real.sqrt 2) ∧
    is_valid_p p ∧
    ∀ q : ℝ, q > p → ¬is_valid_p q :=
sorry

end largest_valid_p_l4049_404952


namespace distance_from_origin_to_point_l4049_404964

theorem distance_from_origin_to_point :
  let x : ℝ := 8
  let y : ℝ := -15
  (x^2 + y^2).sqrt = 17 := by sorry

end distance_from_origin_to_point_l4049_404964


namespace set_equality_implies_x_values_l4049_404948

-- Define the sets A and B
def A (x : ℝ) : Set ℝ := {1, 4, x}
def B (x : ℝ) : Set ℝ := {1, x^2}

-- State the theorem
theorem set_equality_implies_x_values (x : ℝ) :
  A x ∪ B x = A x → x = 2 ∨ x = -2 ∨ x = 0 := by
  sorry

end set_equality_implies_x_values_l4049_404948


namespace total_price_theorem_l4049_404930

/-- The price of a pear in dollars -/
def pear_price : ℕ := 90

/-- The total cost of an orange and a pear in dollars -/
def orange_pear_total : ℕ := 120

/-- The price of an orange in dollars -/
def orange_price : ℕ := orange_pear_total - pear_price

/-- The price of a banana in dollars -/
def banana_price : ℕ := pear_price - orange_price

/-- The number of bananas to buy -/
def num_bananas : ℕ := 200

/-- The number of oranges to buy -/
def num_oranges : ℕ := 2 * num_bananas

theorem total_price_theorem : 
  banana_price * num_bananas + orange_price * num_oranges = 24000 := by
  sorry

end total_price_theorem_l4049_404930


namespace area_of_quadrilateral_l4049_404934

/-- Given a quadrilateral EFGH with the following properties:
  - m∠F = m∠G = 135°
  - EF = 4
  - FG = 6
  - GH = 8
  Prove that the area of EFGH is 18√2 -/
theorem area_of_quadrilateral (E F G H : ℝ × ℝ) : 
  let angle (A B C : ℝ × ℝ) := Real.arccos ((B.1 - A.1) * (C.1 - B.1) + (B.2 - A.2) * (C.2 - B.2)) / 
    (((B.1 - A.1)^2 + (B.2 - A.2)^2)^(1/2) * ((C.1 - B.1)^2 + (C.2 - B.2)^2)^(1/2))
  angle F E G = 135 * π / 180 ∧
  angle G F H = 135 * π / 180 ∧
  ((E.1 - F.1)^2 + (E.2 - F.2)^2)^(1/2) = 4 ∧
  ((F.1 - G.1)^2 + (F.2 - G.2)^2)^(1/2) = 6 ∧
  ((G.1 - H.1)^2 + (G.2 - H.2)^2)^(1/2) = 8 →
  let area := 
    1/2 * ((E.1 - F.1)^2 + (E.2 - F.2)^2)^(1/2) * ((F.1 - G.1)^2 + (F.2 - G.2)^2)^(1/2) * Real.sin (angle F E G) +
    1/2 * ((F.1 - G.1)^2 + (F.2 - G.2)^2)^(1/2) * ((G.1 - H.1)^2 + (G.2 - H.2)^2)^(1/2) * Real.sin (angle G F H)
  area = 18 * Real.sqrt 2 := by
sorry

end area_of_quadrilateral_l4049_404934


namespace monday_kids_count_l4049_404947

/-- The number of kids Julia played with on Tuesday -/
def tuesday_kids : ℕ := 18

/-- The total number of kids Julia played with on Monday and Tuesday combined -/
def monday_tuesday_total : ℕ := 33

/-- The number of kids Julia played with on Monday -/
def monday_kids : ℕ := monday_tuesday_total - tuesday_kids

theorem monday_kids_count : monday_kids = 15 := by
  sorry

end monday_kids_count_l4049_404947


namespace region_volume_l4049_404928

-- Define the region in three-dimensional space
def region (x y z : ℝ) : Prop :=
  abs x + abs y + abs z ≤ 1 ∧ abs x + abs y + abs (z - 2) ≤ 1

-- Define the volume of a region
noncomputable def volume (R : ℝ → ℝ → ℝ → Prop) : ℝ := sorry

-- Theorem statement
theorem region_volume : volume region = 1/12 := by sorry

end region_volume_l4049_404928


namespace smallest_m_fibonacci_l4049_404990

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

def representable (F : ℕ → ℕ) (x : List ℕ) (n : ℕ) : Prop :=
  ∃ (subset : List ℕ), subset.Sublist x ∧ subset.sum = F n

theorem smallest_m_fibonacci :
  ∃ (m : ℕ) (x : List ℕ),
    (∀ i, i ∈ x → i > 0) ∧
    (∀ n, n ≤ 2018 → representable fibonacci x n) ∧
    (∀ m' < m, ¬∃ x' : List ℕ,
      (∀ i, i ∈ x' → i > 0) ∧
      (∀ n, n ≤ 2018 → representable fibonacci x' n)) ∧
    m = 1009 := by
  sorry

end smallest_m_fibonacci_l4049_404990


namespace wengs_hourly_rate_l4049_404919

/-- Weng's hourly rate given her earnings and work duration --/
theorem wengs_hourly_rate (work_duration : ℚ) (earnings : ℚ) : 
  work_duration = 50 / 60 → earnings = 10 → (earnings / work_duration) = 12 := by
  sorry

end wengs_hourly_rate_l4049_404919


namespace second_tap_empty_time_l4049_404960

-- Define the filling time of the first tap
def fill_time : ℝ := 3

-- Define the simultaneous filling time when both taps are open
def simultaneous_fill_time : ℝ := 4.2857142857142865

-- Define the emptying time of the second tap
def empty_time : ℝ := 10

-- Theorem statement
theorem second_tap_empty_time :
  (1 / fill_time - 1 / empty_time = 1 / simultaneous_fill_time) ∧
  empty_time = 10 := by
  sorry

end second_tap_empty_time_l4049_404960


namespace min_value_problem_l4049_404978

theorem min_value_problem (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 1) (hab : a + b = 1) :
  (((a^2 + 1) / (a * b) - 2) * c + Real.sqrt 2 / (c - 1)) ≥ 4 + 2 * Real.sqrt 2 :=
by sorry

end min_value_problem_l4049_404978


namespace solution_set_inequality_holds_max_m_value_l4049_404958

-- Define the function f
def f (x : ℝ) := |2*x + 1| + |3*x - 2|

-- Theorem for the solution set of f(x) ≤ 5
theorem solution_set :
  {x : ℝ | f x ≤ 5} = {x : ℝ | -4/5 ≤ x ∧ x ≤ 6/5} :=
sorry

-- Theorem for the inequality |x-1| + |x+2| ≥ 3
theorem inequality_holds (x : ℝ) :
  |x - 1| + |x + 2| ≥ 3 :=
sorry

-- Theorem for the maximum value of m
theorem max_m_value :
  ∃ (m : ℝ), m = 2 ∧ 
  (∀ (x : ℝ), m^2 - 3*m + 5 ≤ |x - 1| + |x + 2|) ∧
  (∀ (m' : ℝ), (∀ (x : ℝ), m'^2 - 3*m' + 5 ≤ |x - 1| + |x + 2|) → m' ≤ m) :=
sorry

end solution_set_inequality_holds_max_m_value_l4049_404958


namespace uncovered_area_of_shoebox_l4049_404993

/-- Uncovered area of a shoebox with a square block inside -/
theorem uncovered_area_of_shoebox (shoebox_length shoebox_width block_side : ℕ) 
  (h1 : shoebox_length = 6)
  (h2 : shoebox_width = 4)
  (h3 : block_side = 4) :
  shoebox_length * shoebox_width - block_side * block_side = 8 := by
  sorry

end uncovered_area_of_shoebox_l4049_404993


namespace sum_of_unknowns_l4049_404912

theorem sum_of_unknowns (x₁ x₂ x₃ : ℝ) 
  (h : (1 + 2 + 3 + 4 + x₁ + x₂ + x₃) / 7 = 8) : 
  x₁ + x₂ + x₃ = 46 := by
sorry

end sum_of_unknowns_l4049_404912


namespace cherry_pie_angle_l4049_404980

theorem cherry_pie_angle (total_students : ℕ) (chocolate_pref : ℕ) (apple_pref : ℕ) (blueberry_pref : ℕ) :
  total_students = 45 →
  chocolate_pref = 15 →
  apple_pref = 10 →
  blueberry_pref = 9 →
  let remaining := total_students - (chocolate_pref + apple_pref + blueberry_pref)
  let cherry_pref := remaining / 2
  let lemon_pref := remaining / 3
  let pecan_pref := remaining - cherry_pref - lemon_pref
  (cherry_pref : ℚ) / total_students * 360 = 40 :=
by sorry

end cherry_pie_angle_l4049_404980


namespace max_grid_size_is_five_l4049_404989

/-- A coloring of an n × n grid using two colors. -/
def Coloring (n : ℕ) := Fin n → Fin n → Bool

/-- Predicate to check if a rectangle in the grid has all corners of the same color. -/
def hasMonochromaticRectangle (c : Coloring n) : Prop :=
  ∃ (i j k l : Fin n), i < k ∧ j < l ∧
    c i j = c i l ∧ c i l = c k j ∧ c k j = c k l

/-- The maximum size of a grid that can be colored without monochromatic rectangles. -/
def maxGridSize : ℕ := 5

/-- Theorem stating that 5 is the maximum size of a grid that can be colored
    with two colors such that no rectangle has all four corners the same color. -/
theorem max_grid_size_is_five :
  (∀ n : ℕ, n ≤ maxGridSize →
    ∃ c : Coloring n, ¬hasMonochromaticRectangle c) ∧
  (∀ n : ℕ, n > maxGridSize →
    ∀ c : Coloring n, hasMonochromaticRectangle c) :=
sorry

end max_grid_size_is_five_l4049_404989


namespace gross_profit_calculation_l4049_404938

theorem gross_profit_calculation (sales_price : ℝ) (gross_profit_percentage : ℝ) :
  sales_price = 91 →
  gross_profit_percentage = 1.6 →
  ∃ (cost : ℝ) (gross_profit : ℝ),
    cost > 0 ∧
    gross_profit = gross_profit_percentage * cost ∧
    sales_price = cost + gross_profit ∧
    gross_profit = 56 :=
by sorry

end gross_profit_calculation_l4049_404938


namespace square_root_b_minus_a_l4049_404992

/-- Given that the square roots of a positive number are 2-3a and a+2,
    and the cube root of 5a+3b-1 is 3, prove that the square root of b-a is 2. -/
theorem square_root_b_minus_a (a b : ℝ) : 
  (∃ (k : ℝ), k > 0 ∧ (2 - 3*a)^2 = k ∧ (a + 2)^2 = k) →  -- square roots condition
  (5*a + 3*b - 1)^(1/3) = 3 →                             -- cube root condition
  Real.sqrt (b - a) = 2 := by
sorry

end square_root_b_minus_a_l4049_404992


namespace min_unboxed_balls_l4049_404972

/-- Represents the number of balls that can be stored in a big box -/
def big_box_capacity : ℕ := 25

/-- Represents the number of balls that can be stored in a small box -/
def small_box_capacity : ℕ := 20

/-- Represents the total number of balls to be stored -/
def total_balls : ℕ := 104

/-- 
Given:
- Big boxes can store 25 balls each
- Small boxes can store 20 balls each
- There are 104 balls to be stored

Prove that the minimum number of balls that cannot be completely boxed is 4.
-/
theorem min_unboxed_balls : 
  ∀ (big_boxes small_boxes : ℕ), 
    big_boxes * big_box_capacity + small_boxes * small_box_capacity ≤ total_balls →
    4 ≤ total_balls - (big_boxes * big_box_capacity + small_boxes * small_box_capacity) :=
by sorry

end min_unboxed_balls_l4049_404972


namespace parallel_vectors_x_value_l4049_404956

/-- Two vectors in ℝ² are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2

theorem parallel_vectors_x_value :
  ∀ x : ℝ, 
  let a : ℝ × ℝ := (x, -1)
  let b : ℝ × ℝ := (4, 2)
  parallel a b → x = -2 := by
sorry

end parallel_vectors_x_value_l4049_404956


namespace orange_distribution_difference_l4049_404907

/-- The number of students in the class -/
def num_students : ℕ := 25

/-- The initial total number of oranges -/
def total_oranges : ℕ := 240

/-- The number of bad oranges that were removed -/
def bad_oranges : ℕ := 65

/-- The difference in oranges per student before and after removing bad oranges -/
def orange_difference : ℚ := (total_oranges : ℚ) / num_students - ((total_oranges - bad_oranges) : ℚ) / num_students

theorem orange_distribution_difference :
  orange_difference = 2.6 := by sorry

end orange_distribution_difference_l4049_404907


namespace fourth_root_over_sixth_root_of_seven_l4049_404957

theorem fourth_root_over_sixth_root_of_seven (x : ℝ) :
  (7 : ℝ) ^ (1 / 4) / (7 : ℝ) ^ (1 / 6) = (7 : ℝ) ^ (1 / 12) :=
by sorry

end fourth_root_over_sixth_root_of_seven_l4049_404957


namespace sqrt_nine_minus_one_l4049_404909

theorem sqrt_nine_minus_one : Real.sqrt 9 - 1 = 2 := by
  sorry

end sqrt_nine_minus_one_l4049_404909


namespace trig_identity_quadratic_equation_solution_l4049_404944

-- Problem 1
theorem trig_identity : 
  Real.sin (π / 4) - 3 * Real.tan (π / 6) + Real.sqrt 2 * Real.cos (π / 3) = Real.sqrt 2 - Real.sqrt 3 := by
  sorry

-- Problem 2
theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x ↦ x^2 - 6*x + 8
  ∀ x : ℝ, f x = 0 ↔ x = 4 ∨ x = 2 := by
  sorry

end trig_identity_quadratic_equation_solution_l4049_404944


namespace inequality_range_l4049_404994

theorem inequality_range (a : ℝ) : (∀ x : ℝ, a < |x - 4| + |x + 3|) → a < 7 := by
  sorry

end inequality_range_l4049_404994


namespace sqrt_sum_equation_solutions_l4049_404918

theorem sqrt_sum_equation_solutions :
  ∀ x : ℝ, (Real.sqrt ((3 + 2 * Real.sqrt 2) ^ x) + Real.sqrt ((3 - 2 * Real.sqrt 2) ^ x) = 6) ↔ (x = 2 ∨ x = -2) := by
  sorry

end sqrt_sum_equation_solutions_l4049_404918


namespace exactly_two_cheaper_to_buy_more_l4049_404974

-- Define the cost function
def C (n : ℕ) : ℝ :=
  if n ≤ 30 then 15 * n - 20
  else if n ≤ 55 then 14 * n
  else 13 * n + 10

-- Define a function that checks if it's cheaper to buy n+1 books than n books
def cheaperToBuyMore (n : ℕ) : Prop := C (n + 1) < C n

-- Theorem statement
theorem exactly_two_cheaper_to_buy_more :
  ∃! (s : Finset ℕ), (∀ n ∈ s, cheaperToBuyMore n) ∧ s.card = 2 := by
  sorry

end exactly_two_cheaper_to_buy_more_l4049_404974
