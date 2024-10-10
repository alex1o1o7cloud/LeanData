import Mathlib

namespace triangle_construction_theorem_l3558_355848

-- Define the types for points and triangles
def Point := ℝ × ℝ
def Triangle := Point × Point × Point

-- Define a function to calculate the area of a triangle
def triangleArea (t : Triangle) : ℝ := sorry

-- Define a predicate for parallel lines
def parallel (p1 p2 q1 q2 : Point) : Prop := sorry

-- Define a predicate for congruent triangles
def congruent (t1 t2 : Triangle) : Prop := sorry

theorem triangle_construction_theorem 
  (ABC A₁B₁C₁ : Triangle) 
  (h_equal_area : triangleArea ABC = triangleArea A₁B₁C₁) :
  ∃ (A₂B₂C₂ : Triangle),
    congruent A₂B₂C₂ A₁B₁C₁ ∧ 
    parallel (ABC.1) (A₂B₂C₂.1) (ABC.2.1) (A₂B₂C₂.2.1) ∧
    parallel (ABC.2.1) (A₂B₂C₂.2.1) (ABC.2.2) (A₂B₂C₂.2.2) :=
by sorry

end triangle_construction_theorem_l3558_355848


namespace stratified_sampling_most_appropriate_l3558_355890

-- Define the population
structure Population where
  grades : List String
  students : List String

-- Define the sampling methods
inductive SamplingMethod
  | DrawingLots
  | RandomNumber
  | Stratified
  | Systematic

-- Define the survey requirements
structure SurveyRequirements where
  proportional_sampling : Bool
  multiple_grades : Bool

-- Define a function to determine the most appropriate sampling method
def most_appropriate_method (pop : Population) (req : SurveyRequirements) : SamplingMethod :=
  sorry

-- Theorem stating that stratified sampling is most appropriate
-- for a population with multiple grades and proportional sampling requirement
theorem stratified_sampling_most_appropriate 
  (pop : Population) 
  (req : SurveyRequirements) :
  pop.grades.length > 1 → 
  req.proportional_sampling = true → 
  req.multiple_grades = true → 
  most_appropriate_method pop req = SamplingMethod.Stratified :=
  sorry

end stratified_sampling_most_appropriate_l3558_355890


namespace intersection_and_union_when_m_eq_2_subset_complement_iff_m_geq_3_l3558_355847

-- Define sets A and B
def A (m : ℝ) : Set ℝ := {x : ℝ | x > 2^m}
def B : Set ℝ := {x : ℝ | -4 < x - 4 ∧ x - 4 < 4}

-- Theorem for part (1)
theorem intersection_and_union_when_m_eq_2 :
  (A 2 ∩ B = {x : ℝ | 4 < x ∧ x < 8}) ∧
  (A 2 ∪ B = {x : ℝ | x > 0}) := by sorry

-- Theorem for part (2)
theorem subset_complement_iff_m_geq_3 (m : ℝ) :
  A m ⊆ (Set.univ \ B) ↔ m ≥ 3 := by sorry

end intersection_and_union_when_m_eq_2_subset_complement_iff_m_geq_3_l3558_355847


namespace pure_imaginary_condition_l3558_355873

/-- Given that z = (a - i) / (2 - i) is a pure imaginary number, prove that a = -1/2 --/
theorem pure_imaginary_condition (a : ℝ) : 
  let z : ℂ := (a - Complex.I) / (2 - Complex.I)
  (∃ (b : ℝ), z = Complex.I * b) → a = -1/2 := by
  sorry

end pure_imaginary_condition_l3558_355873


namespace ellipse_intersection_fixed_point_l3558_355809

/-- Definition of the ellipse C -/
def ellipse_C (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

/-- Definition of a point being on the ellipse C -/
def on_ellipse_C (p : ℝ × ℝ) : Prop := ellipse_C p.1 p.2

/-- Definition of the line l -/
def line_l (k m x y : ℝ) : Prop := y = k * x + m

/-- Definition of a point being on the line l -/
def on_line_l (k m : ℝ) (p : ℝ × ℝ) : Prop := line_l k m p.1 p.2

/-- Definition of the right vertex of the ellipse C -/
def right_vertex : ℝ × ℝ := (2, 0)

/-- Definition of the circle with diameter AB passing through a point -/
def circle_AB_passes_through (A B p : ℝ × ℝ) : Prop :=
  (p.1 - A.1) * (p.1 - B.1) + (p.2 - A.2) * (p.2 - B.2) = 0

/-- The main theorem -/
theorem ellipse_intersection_fixed_point :
  ∀ (k m : ℝ) (A B : ℝ × ℝ),
    on_ellipse_C A ∧ on_ellipse_C B ∧
    on_line_l k m A ∧ on_line_l k m B ∧
    A ≠ right_vertex ∧ B ≠ right_vertex ∧
    circle_AB_passes_through A B right_vertex →
    on_line_l k m (1/2, 0) :=
sorry

end ellipse_intersection_fixed_point_l3558_355809


namespace sum_digits_base5_588_l3558_355839

/-- Converts a natural number from base 10 to base 5 -/
def toBase5 (n : ℕ) : List ℕ :=
  sorry

/-- Sums the digits in a list -/
def sumDigits (digits : List ℕ) : ℕ :=
  sorry

theorem sum_digits_base5_588 :
  sumDigits (toBase5 588) = 12 := by
  sorry

end sum_digits_base5_588_l3558_355839


namespace solution_equality_l3558_355870

theorem solution_equality (a : ℝ) : 
  (∃ x, 2 - a - x = 0 ∧ 2*x + 1 = 3) → a = 1 := by
sorry

end solution_equality_l3558_355870


namespace chess_draw_probability_l3558_355803

theorem chess_draw_probability (p_win p_not_lose : ℝ) 
  (h_win : p_win = 0.4) 
  (h_not_lose : p_not_lose = 0.9) : 
  p_not_lose - p_win = 0.5 := by
sorry

end chess_draw_probability_l3558_355803


namespace closest_integer_to_sqrt_6_l3558_355801

theorem closest_integer_to_sqrt_6 : 
  ∃ (n : ℤ), ∀ (m : ℤ), |n - Real.sqrt 6| ≤ |m - Real.sqrt 6| ∧ n = 2 :=
sorry

end closest_integer_to_sqrt_6_l3558_355801


namespace cubic_roots_sum_l3558_355886

theorem cubic_roots_sum (p q r : ℝ) : 
  (p^3 - 2*p^2 - p + 2 = 0) → 
  (q^3 - 2*q^2 - q + 2 = 0) → 
  (r^3 - 2*r^2 - r + 2 = 0) → 
  p*(q - r)^2 + q*(r - p)^2 + r*(p - q)^2 = 16 := by
sorry

end cubic_roots_sum_l3558_355886


namespace quadratic_inequality_solution_set_l3558_355819

def f (a : ℤ) (x : ℝ) : ℝ := a * x^2 - (a + 2) * x + 1

theorem quadratic_inequality_solution_set 
  (a : ℤ) 
  (h1 : ∃! x : ℝ, -2 < x ∧ x < -1 ∧ f a x = 0) :
  {x : ℝ | f a x > 1} = {x : ℝ | -1 < x ∧ x < 0} := by
sorry

end quadratic_inequality_solution_set_l3558_355819


namespace organization_size_l3558_355822

/-- The total number of employees in an organization -/
def total_employees : ℕ := sorry

/-- The number of employees earning below 10k $ -/
def below_10k : ℕ := 250

/-- The number of employees earning between 10k $ and 50k $ -/
def between_10k_50k : ℕ := 500

/-- The percentage of employees earning less than 50k $ -/
def percent_below_50k : ℚ := 75 / 100

theorem organization_size :
  (below_10k + between_10k_50k : ℚ) = percent_below_50k * total_employees ∧
  total_employees = 1000 := by sorry

end organization_size_l3558_355822


namespace solution_set_inequality_l3558_355887

theorem solution_set_inequality (x : ℝ) :
  (x ≠ 2) → (1 / (x - 2) > -2 ↔ x ∈ Set.Iio (3/2) ∪ Set.Ioi 2) :=
by sorry

end solution_set_inequality_l3558_355887


namespace min_dot_product_ep_qp_l3558_355896

/-- The ellipse defined by x^2/36 + y^2/9 = 1 -/
def is_on_ellipse (x y : ℝ) : Prop := x^2/36 + y^2/9 = 1

/-- The fixed point E -/
def E : ℝ × ℝ := (3, 0)

/-- The dot product of two 2D vectors -/
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

/-- The squared distance between two points -/
def distance_squared (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

/-- Theorem: The minimum value of EP · QP is 6 -/
theorem min_dot_product_ep_qp :
  ∃ (min : ℝ),
    (∀ (P Q : ℝ × ℝ),
      is_on_ellipse P.1 P.2 →
      is_on_ellipse Q.1 Q.2 →
      dot_product (P.1 - E.1, P.2 - E.2) (Q.1 - P.1, Q.2 - P.2) = 0 →
      dot_product (P.1 - E.1, P.2 - E.2) (Q.1 - P.1, Q.2 - P.2) ≥ min) ∧
    min = 6 :=
sorry

end min_dot_product_ep_qp_l3558_355896


namespace brady_current_yards_l3558_355811

/-- The passing yards record in a season -/
def record : ℕ := 5999

/-- The number of games left in the season -/
def games_left : ℕ := 6

/-- The average passing yards needed per game to beat the record -/
def average_needed : ℕ := 300

/-- Tom Brady's current passing yards -/
def current_yards : ℕ := 4200

theorem brady_current_yards : 
  current_yards = record + 1 - (games_left * average_needed) :=
sorry

end brady_current_yards_l3558_355811


namespace rectangle_perimeter_l3558_355843

/-- A rectangle with length thrice its breadth and area 675 square meters has a perimeter of 120 meters. -/
theorem rectangle_perimeter (b : ℝ) (h1 : b > 0) : 
  let l := 3 * b
  let area := l * b
  area = 675 →
  2 * (l + b) = 120 := by
sorry

end rectangle_perimeter_l3558_355843


namespace exists_non_polynomial_satisfying_inequality_l3558_355898

-- Define a periodic function with period 2
def periodic_function (k : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, k (x + 2) = k x

-- Define a bounded function
def bounded_function (k : ℝ → ℝ) : Prop :=
  ∃ M : ℝ, ∀ x : ℝ, |k x| ≤ M

-- Define a non-constant function
def non_constant_function (k : ℝ → ℝ) : Prop :=
  ∃ x y : ℝ, k x ≠ k y

-- Main theorem
theorem exists_non_polynomial_satisfying_inequality :
  ∃ f : ℝ → ℝ, 
    (∀ x : ℝ, (x - 1) * f (x + 1) - (x + 1) * f (x - 1) ≥ 4 * x * (x^2 - 1)) ∧
    (∃ k : ℝ → ℝ, 
      (∀ x : ℝ, f x = x^3 + x * k x) ∧
      periodic_function k ∧
      bounded_function k ∧
      non_constant_function k) :=
sorry

end exists_non_polynomial_satisfying_inequality_l3558_355898


namespace probability_arts_and_sciences_is_two_thirds_l3558_355863

/-- Represents a class subject -/
inductive Subject
  | Mathematics
  | Chinese
  | Politics
  | Geography
  | English
  | History
  | PhysicalEducation

/-- Represents the time of day for a class -/
inductive TimeOfDay
  | Morning
  | Afternoon

/-- Defines the class schedule -/
def schedule : TimeOfDay → List Subject
  | TimeOfDay.Morning => [Subject.Mathematics, Subject.Chinese, Subject.Politics, Subject.Geography]
  | TimeOfDay.Afternoon => [Subject.English, Subject.History, Subject.PhysicalEducation]

/-- Determines if a subject is related to arts and sciences -/
def isArtsAndSciences : Subject → Bool
  | Subject.Politics => true
  | Subject.History => true
  | Subject.Geography => true
  | _ => false

/-- The probability of selecting at least one arts and sciences class -/
def probabilityArtsAndSciences : ℚ := 2/3

theorem probability_arts_and_sciences_is_two_thirds :
  probabilityArtsAndSciences = 2/3 := by
  sorry

end probability_arts_and_sciences_is_two_thirds_l3558_355863


namespace hyperbola_eccentricity_l3558_355885

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if the distance from the focus to the asymptote is equal to the length of the real axis,
    then the eccentricity of the hyperbola is √5. -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let c := Real.sqrt (a^2 + b^2)
  let focus_to_asymptote := (b * c) / Real.sqrt (a^2 + b^2)
  focus_to_asymptote = 2 * a →
  c / a = Real.sqrt 5 := by
  sorry

end hyperbola_eccentricity_l3558_355885


namespace base_two_representation_123_l3558_355814

theorem base_two_representation_123 :
  ∃ (a b c d e f g : Nat),
    123 = a * 2^6 + b * 2^5 + c * 2^4 + d * 2^3 + e * 2^2 + f * 2^1 + g * 2^0 ∧
    a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ e = 0 ∧ f = 1 ∧ g = 1 :=
by sorry

end base_two_representation_123_l3558_355814


namespace max_prime_factors_b_l3558_355878

theorem max_prime_factors_b (a b : ℕ+) 
  (h_gcd : (Nat.gcd a b).factors.length = 10)
  (h_lcm : (Nat.lcm a b).factors.length = 25)
  (h_fewer : (b.val.factors.length : ℤ) < a.val.factors.length) :
  b.val.factors.length ≤ 17 :=
sorry

end max_prime_factors_b_l3558_355878


namespace arithmetic_mean_of_first_four_primes_reciprocals_l3558_355850

theorem arithmetic_mean_of_first_four_primes_reciprocals :
  let first_four_primes := [2, 3, 5, 7]
  let reciprocals := first_four_primes.map (λ x => 1 / x)
  let arithmetic_mean := (reciprocals.sum) / 4
  arithmetic_mean = 247 / 840 := by
sorry

end arithmetic_mean_of_first_four_primes_reciprocals_l3558_355850


namespace range_of_a_l3558_355855

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 - x + a ≥ 0) → a ≥ 1/4 := by
  sorry

end range_of_a_l3558_355855


namespace transformed_area_l3558_355844

-- Define the matrix
def A : Matrix (Fin 2) (Fin 2) ℝ := !![3, 2; 4, -5]

-- Define the area of the original region R
def area_R : ℝ := 15

-- Theorem statement
theorem transformed_area :
  let det_A := Matrix.det A
  area_R * |det_A| = 345 := by
sorry

end transformed_area_l3558_355844


namespace triangle_side_ratio_bounds_l3558_355895

theorem triangle_side_ratio_bounds (a b c : ℝ) (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  let t := (a + b + c) / Real.sqrt (a * b + b * c + c * a)
  Real.sqrt 3 ≤ t ∧ t < 2 := by sorry

end triangle_side_ratio_bounds_l3558_355895


namespace marley_louis_orange_ratio_l3558_355802

theorem marley_louis_orange_ratio :
  let louis_oranges : ℕ := 5
  let samantha_apples : ℕ := 7
  let marley_apples : ℕ := 3 * samantha_apples
  let marley_total_fruits : ℕ := 31
  let marley_oranges : ℕ := marley_total_fruits - marley_apples
  (marley_oranges : ℚ) / louis_oranges = 2 := by sorry

end marley_louis_orange_ratio_l3558_355802


namespace total_footprints_pogo_and_grimzi_footprints_l3558_355860

/-- Calculates the total number of footprints left by two creatures on their respective planets -/
theorem total_footprints (pogo_footprints_per_meter : ℕ) 
                         (grimzi_footprints_per_six_meters : ℕ) 
                         (distance : ℕ) : ℕ :=
  let pogo_total := pogo_footprints_per_meter * distance
  let grimzi_total := grimzi_footprints_per_six_meters * (distance / 6)
  pogo_total + grimzi_total

/-- Proves that the combined total number of footprints left by Pogo and Grimzi is 27,000 -/
theorem pogo_and_grimzi_footprints : 
  total_footprints 4 3 6000 = 27000 := by
  sorry

end total_footprints_pogo_and_grimzi_footprints_l3558_355860


namespace sum_of_reciprocal_roots_l3558_355836

theorem sum_of_reciprocal_roots (γ δ : ℝ) : 
  (∃ c d : ℝ, c ≠ 0 ∧ d ≠ 0 ∧ 
   6 * c^2 + 5 * c + 7 = 0 ∧ 
   6 * d^2 + 5 * d + 7 = 0 ∧ 
   γ = 1 / c ∧ 
   δ = 1 / d) → 
  γ + δ = -5 / 7 := by
sorry

end sum_of_reciprocal_roots_l3558_355836


namespace shadow_length_sequence_l3558_355869

/-- Represents the position of a person relative to a street lamp -/
inductive Position
  | Before
  | Under
  | After

/-- Represents the length of a shadow -/
inductive ShadowLength
  | Long
  | Short

/-- A street lamp as a fixed light source -/
structure StreetLamp where
  position : ℝ × ℝ  -- (x, y) coordinates
  height : ℝ

/-- A person walking past a street lamp -/
structure Person where
  height : ℝ

/-- Calculates the shadow length based on the person's position relative to the lamp -/
def shadowLength (lamp : StreetLamp) (person : Person) (pos : Position) : ShadowLength :=
  sorry

/-- Theorem stating how the shadow length changes as a person walks under a street lamp -/
theorem shadow_length_sequence (lamp : StreetLamp) (person : Person) :
  shadowLength lamp person Position.Before = ShadowLength.Long ∧
  shadowLength lamp person Position.Under = ShadowLength.Short ∧
  shadowLength lamp person Position.After = ShadowLength.Long :=
sorry

end shadow_length_sequence_l3558_355869


namespace inscribed_prism_volume_l3558_355862

/-- Regular triangular prism inscribed in a sphere -/
structure InscribedPrism where
  /-- Radius of the sphere -/
  R : ℝ
  /-- Distance from vertex A to point D on the sphere -/
  AD : ℝ
  /-- Assertion that CD is a diameter of the sphere -/
  is_diameter : Bool

/-- Volume of the inscribed prism -/
def prism_volume (p : InscribedPrism) : ℝ :=
  sorry

/-- Theorem stating the volume of the specific inscribed prism -/
theorem inscribed_prism_volume :
  ∀ (p : InscribedPrism),
    p.R = 3 ∧ p.AD = 2 * Real.sqrt 6 ∧ p.is_diameter = true →
    prism_volume p = 6 * Real.sqrt 15 :=
  sorry

end inscribed_prism_volume_l3558_355862


namespace urn_contents_l3558_355829

/-- Represents the contents of an urn with yellow, white, and red balls. -/
structure Urn :=
  (yellow : ℕ)
  (white : ℕ)
  (red : ℕ)

/-- Calculates the probability of drawing balls of given colors from the urn. -/
def probability (u : Urn) (colors : List ℕ) : ℚ :=
  (colors.sum : ℚ) / ((u.yellow + u.white + u.red) : ℚ)

/-- The main theorem about the urn contents. -/
theorem urn_contents : 
  ∀ (u : Urn), 
    u.yellow = 18 →
    probability u [u.white, u.red] = probability u [u.white, u.yellow] - 1/15 →
    probability u [u.red, u.yellow] = probability u [u.white, u.yellow] * 11/10 →
    u.white = 27 ∧ u.red = 16 := by
  sorry

end urn_contents_l3558_355829


namespace exists_m_for_inequality_l3558_355817

def sequence_a : ℕ → ℚ
  | 7 => 16/3
  | n+1 => (3 * sequence_a n + 4) / (7 - sequence_a n)
  | _ => 0  -- Define for n < 7 to make the function total

theorem exists_m_for_inequality :
  ∃ m : ℕ, ∀ n ≥ m, sequence_a n > (sequence_a (n-1) + sequence_a (n+1)) / 2 :=
sorry

end exists_m_for_inequality_l3558_355817


namespace debate_team_arrangements_l3558_355813

-- Define the number of students
def total_students : ℕ := 6

-- Define the number of team members
def team_size : ℕ := 4

-- Define the number of positions where student A can be placed
def positions_for_A : ℕ := 3

-- Define the number of remaining students after A is placed
def remaining_students : ℕ := total_students - 1

-- Define the number of remaining positions after A is placed
def remaining_positions : ℕ := team_size - 1

-- Theorem statement
theorem debate_team_arrangements :
  (positions_for_A * (remaining_students.factorial / (remaining_students - remaining_positions).factorial)) = 180 := by
  sorry

end debate_team_arrangements_l3558_355813


namespace prob_A_wins_four_consecutive_prob_need_fifth_game_prob_C_ultimate_winner_l3558_355871

/-- Represents a player in the badminton game --/
inductive Player : Type
  | A
  | B
  | C

/-- Represents the state of the game --/
structure GameState :=
  (current_players : List Player)
  (bye_player : Player)
  (eliminated_player : Option Player)

/-- The probability of a player winning a single game --/
def win_probability : ℚ := 1/2

/-- The initial game state --/
def initial_state : GameState :=
  { current_players := [Player.A, Player.B],
    bye_player := Player.C,
    eliminated_player := none }

/-- Calculates the probability of a specific game outcome --/
def outcome_probability (num_games : ℕ) : ℚ :=
  (win_probability ^ num_games : ℚ)

/-- Theorem stating the probability of A winning four consecutive games --/
theorem prob_A_wins_four_consecutive :
  outcome_probability 4 = 1/16 := by sorry

/-- Theorem stating the probability of needing a fifth game --/
theorem prob_need_fifth_game :
  1 - 4 * outcome_probability 4 = 3/4 := by sorry

/-- Theorem stating the probability of C being the ultimate winner --/
theorem prob_C_ultimate_winner :
  7/16 = 1 - 2 * (outcome_probability 4 + 7 * outcome_probability 5) := by sorry

end prob_A_wins_four_consecutive_prob_need_fifth_game_prob_C_ultimate_winner_l3558_355871


namespace rectangle_width_equals_three_l3558_355841

theorem rectangle_width_equals_three (square_side : ℝ) (rect_length : ℝ) (rect_width : ℝ)
  (h1 : square_side = 9)
  (h2 : rect_length = 27)
  (h3 : square_side * square_side = rect_length * rect_width) :
  rect_width = 3 :=
by sorry

end rectangle_width_equals_three_l3558_355841


namespace ms_leech_class_boys_l3558_355849

/-- Proves that the number of boys in Ms. Leech's class is 10 -/
theorem ms_leech_class_boys (total_students : ℕ) (total_cups : ℕ) (cups_per_boy : ℕ) :
  total_students = 30 →
  total_cups = 90 →
  cups_per_boy = 5 →
  ∃ (boys : ℕ),
    boys * 3 = total_students ∧
    boys * cups_per_boy = total_cups / 2 ∧
    boys = 10 :=
by sorry

end ms_leech_class_boys_l3558_355849


namespace total_budget_is_40_l3558_355851

/-- The total budget for Samuel and Kevin's cinema outing -/
def total_budget : ℕ :=
  let samuel_ticket := 14
  let samuel_snacks := 6
  let kevin_ticket := 14
  let kevin_drinks := 2
  let kevin_food := 4
  samuel_ticket + samuel_snacks + kevin_ticket + kevin_drinks + kevin_food

/-- Theorem stating that the total budget for the outing is $40 -/
theorem total_budget_is_40 : total_budget = 40 := by
  sorry

end total_budget_is_40_l3558_355851


namespace correct_ranking_l3558_355891

-- Define the team members
inductive TeamMember
| David
| Emma
| Frank

-- Define the experience relation
def has_more_experience (a b : TeamMember) : Prop := sorry

-- Define the most experienced member
def is_most_experienced (m : TeamMember) : Prop :=
  ∀ x : TeamMember, x ≠ m → has_more_experience m x

-- Define the statements
def statement_I : Prop := has_more_experience TeamMember.Frank TeamMember.Emma
def statement_II : Prop := has_more_experience TeamMember.David TeamMember.Frank
def statement_III : Prop := is_most_experienced TeamMember.Frank

-- Define the condition that exactly one statement is true
def exactly_one_true : Prop :=
  (statement_I ∧ ¬statement_II ∧ ¬statement_III) ∨
  (¬statement_I ∧ statement_II ∧ ¬statement_III) ∨
  (¬statement_I ∧ ¬statement_II ∧ statement_III)

-- The theorem to prove
theorem correct_ranking (h : exactly_one_true) :
  has_more_experience TeamMember.David TeamMember.Emma ∧
  has_more_experience TeamMember.Emma TeamMember.Frank :=
sorry

end correct_ranking_l3558_355891


namespace expression_evaluation_l3558_355833

theorem expression_evaluation :
  let x : ℝ := 2
  let y : ℝ := -1
  (6 * x^2 * y * (-2 * x * y + y^3)) / (x * y^2) = -36 := by
  sorry

end expression_evaluation_l3558_355833


namespace fraction_product_l3558_355825

theorem fraction_product : (2/3 : ℚ) * (5/11 : ℚ) * (3/8 : ℚ) = (5/44 : ℚ) := by
  sorry

end fraction_product_l3558_355825


namespace lunes_area_equals_rectangle_area_l3558_355866

/-- Given a rectangle with sides a and b, with half-circles drawn outward on each side
    and a circumscribing circle, the area of the lunes (crescent shapes) is equal to
    the area of the rectangle. -/
theorem lunes_area_equals_rectangle_area (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let semicircle_area := π * (a^2 + b^2) / 4
  let circumscribed_circle_area := π * (a^2 + b^2) / 4
  let rectangle_area := a * b
  let lunes_area := semicircle_area + rectangle_area - circumscribed_circle_area
  lunes_area = rectangle_area :=
by sorry

end lunes_area_equals_rectangle_area_l3558_355866


namespace only_6_8_10_is_right_triangle_l3558_355876

-- Define a function to check if three numbers can form a right triangle
def isRightTriangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

-- Theorem stating that only (6, 8, 10) forms a right triangle among the given sets
theorem only_6_8_10_is_right_triangle :
  ¬(isRightTriangle 4 5 6) ∧
  ¬(isRightTriangle 5 7 9) ∧
  isRightTriangle 6 8 10 ∧
  ¬(isRightTriangle 7 8 9) :=
sorry

end only_6_8_10_is_right_triangle_l3558_355876


namespace first_term_of_arithmetic_sequence_l3558_355807

def arithmetic_sequence (a d : ℚ) (n : ℕ) : ℚ := a + (n - 1 : ℚ) * d

def sum_arithmetic_sequence (a d : ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * a + (n - 1 : ℚ) * d)

theorem first_term_of_arithmetic_sequence :
  ∃ (a d : ℚ),
    sum_arithmetic_sequence a d 30 = 300 ∧
    sum_arithmetic_sequence (arithmetic_sequence a d 31) d 40 = 2200 ∧
    a = -121 / 14 := by
  sorry

end first_term_of_arithmetic_sequence_l3558_355807


namespace exists_universal_source_l3558_355888

/-- A directed graph where every pair of vertices is connected by a directed edge. -/
structure CompleteDigraph (V : Type*) [Fintype V] [DecidableEq V] :=
  (edge : V → V → Prop)
  (complete : ∀ (u v : V), u ≠ v → edge u v ∨ edge v u)

/-- A path of length at most 2 between two vertices. -/
def PathOfLengthAtMostTwo {V : Type*} (edge : V → V → Prop) (u v : V) : Prop :=
  edge u v ∨ ∃ w, edge u w ∧ edge w v

/-- 
In a complete directed graph, there exists a vertex from which 
every other vertex can be reached by a path of length at most 2.
-/
theorem exists_universal_source {V : Type*} [Fintype V] [DecidableEq V] 
  (G : CompleteDigraph V) : 
  ∃ (u : V), ∀ (v : V), u ≠ v → PathOfLengthAtMostTwo G.edge u v :=
sorry

end exists_universal_source_l3558_355888


namespace largest_number_in_set_l3558_355854

theorem largest_number_in_set (a : ℝ) (h : a = -3) :
  -2 * a = max (-2 * a) (max (5 * a) (max (36 / a) (max (a ^ 3) 2))) := by
  sorry

end largest_number_in_set_l3558_355854


namespace units_digit_problem_l3558_355806

def units_digit (n : ℤ) : ℕ := n.natAbs % 10

theorem units_digit_problem : units_digit (8 * 19 * 1978 - 8^3) = 4 := by
  sorry

end units_digit_problem_l3558_355806


namespace geometric_sequence_min_value_l3558_355808

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_min_value (a : ℕ → ℝ) (m n : ℕ) :
  is_geometric_sequence a →
  (∀ k, a k > 0) →
  a 3 = a 2 + 2 * a 1 →
  Real.sqrt (a m * a n) = 4 * a 1 →
  (1 : ℝ) / m + 4 / n ≥ 3 / 2 :=
by sorry

end geometric_sequence_min_value_l3558_355808


namespace least_positive_integer_for_multiple_of_five_l3558_355821

theorem least_positive_integer_for_multiple_of_five : ∃ (n : ℕ), n > 0 ∧ (365 + n) % 5 = 0 ∧ ∀ (m : ℕ), m > 0 ∧ (365 + m) % 5 = 0 → n ≤ m := by
  sorry

end least_positive_integer_for_multiple_of_five_l3558_355821


namespace smallest_four_digit_divisible_by_4_and_5_l3558_355864

theorem smallest_four_digit_divisible_by_4_and_5 :
  ∃ n : ℕ, 
    (1000 ≤ n ∧ n < 10000) ∧  -- four-digit number
    (n % 4 = 0) ∧             -- divisible by 4
    (n % 5 = 0) ∧             -- divisible by 5
    (∀ m : ℕ, (1000 ≤ m ∧ m < 10000) ∧ (m % 4 = 0) ∧ (m % 5 = 0) → n ≤ m) ∧  -- smallest such number
    n = 1020 :=
by sorry

end smallest_four_digit_divisible_by_4_and_5_l3558_355864


namespace x_y_inequalities_l3558_355877

theorem x_y_inequalities (x y : ℝ) (h1 : x - y > x + 1) (h2 : x + y < y - 2) :
  x < -2 ∧ y < -1 := by sorry

end x_y_inequalities_l3558_355877


namespace arithmetic_sequence_problem_l3558_355846

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a where a₃ = -6 and a₇ = a₅ + 4, prove that a₁ = -10 -/
theorem arithmetic_sequence_problem (a : ℕ → ℤ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_a3 : a 3 = -6) 
  (h_a7 : a 7 = a 5 + 4) : 
  a 1 = -10 := by
sorry

end arithmetic_sequence_problem_l3558_355846


namespace shiela_neighbors_l3558_355831

theorem shiela_neighbors (total_drawings : ℕ) (drawings_per_neighbor : ℕ) 
  (h1 : total_drawings = 54)
  (h2 : drawings_per_neighbor = 9)
  : total_drawings / drawings_per_neighbor = 6 := by
  sorry

end shiela_neighbors_l3558_355831


namespace cubic_product_theorem_l3558_355827

theorem cubic_product_theorem : 
  (2^3 - 1) / (2^3 + 1) * 
  (3^3 - 1) / (3^3 + 1) * 
  (4^3 - 1) / (4^3 + 1) * 
  (5^3 - 1) / (5^3 + 1) * 
  (6^3 - 1) / (6^3 + 1) * 
  (7^3 - 1) / (7^3 + 1) = 19 / 56 := by
  sorry

end cubic_product_theorem_l3558_355827


namespace julie_newspaper_sheets_l3558_355899

/-- The number of sheets used to print one newspaper -/
def sheets_per_newspaper (boxes : ℕ) (packages_per_box : ℕ) (sheets_per_package : ℕ) (total_newspapers : ℕ) : ℕ :=
  (boxes * packages_per_box * sheets_per_package) / total_newspapers

/-- Proof that Julie uses 25 sheets to print one newspaper -/
theorem julie_newspaper_sheets : 
  sheets_per_newspaper 2 5 250 100 = 25 := by
  sorry

end julie_newspaper_sheets_l3558_355899


namespace problem_solving_probability_l3558_355884

theorem problem_solving_probability 
  (kyle_prob : ℚ) 
  (david_prob : ℚ) 
  (catherine_prob : ℚ) 
  (h1 : kyle_prob = 1/3) 
  (h2 : david_prob = 2/7) 
  (h3 : catherine_prob = 5/9) : 
  kyle_prob * catherine_prob * (1 - david_prob) = 25/189 := by
sorry

end problem_solving_probability_l3558_355884


namespace original_magazine_cost_l3558_355832

/-- The original cost of a magazine can be determined from the number of magazines, 
    selling price, and total profit. -/
theorem original_magazine_cost 
  (num_magazines : ℕ) 
  (selling_price : ℚ) 
  (total_profit : ℚ) : 
  num_magazines = 10 → 
  selling_price = 7/2 → 
  total_profit = 5 → 
  (num_magazines : ℚ) * selling_price - total_profit = 30 ∧ 
  ((num_magazines : ℚ) * selling_price - total_profit) / num_magazines = 3 :=
by sorry

end original_magazine_cost_l3558_355832


namespace sector_angle_l3558_355818

theorem sector_angle (R : ℝ) (α : ℝ) : 
  R > 0 ∧ 2 * R + α * R = 6 ∧ (1/2) * R^2 * α = 2 → α = 1 ∨ α = 4 := by
  sorry

end sector_angle_l3558_355818


namespace starting_lineup_count_l3558_355880

theorem starting_lineup_count :
  let team_size : ℕ := 12
  let lineup_size : ℕ := 5
  let captain_count : ℕ := 1
  let other_players_count : ℕ := lineup_size - captain_count
  team_size * Nat.choose (team_size - captain_count) other_players_count = 3960 :=
by sorry

end starting_lineup_count_l3558_355880


namespace natalie_shopping_result_l3558_355875

/-- Calculates the amount of money Natalie has left after shopping -/
def money_left (initial_amount jumper_price tshirt_price heels_price jumper_discount_rate sales_tax_rate : ℚ) : ℚ :=
  let discounted_jumper_price := jumper_price * (1 - jumper_discount_rate)
  let total_before_tax := discounted_jumper_price + tshirt_price + heels_price
  let total_after_tax := total_before_tax * (1 + sales_tax_rate)
  initial_amount - total_after_tax

/-- Theorem stating that Natalie has $18.62 left after shopping -/
theorem natalie_shopping_result : 
  money_left 100 25 15 40 (10/100) (5/100) = 18.62 := by
  sorry

end natalie_shopping_result_l3558_355875


namespace triangle_properties_l3558_355859

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

/-- The given triangle satisfies the specified conditions -/
def satisfies_conditions (t : Triangle) : Prop :=
  t.a * Real.cos t.B = (2 * t.c - t.b) * Real.cos t.A ∧
  t.a = Real.sqrt 7 ∧
  (1 / 2) * t.b * t.c * Real.sin t.A = (3 * Real.sqrt 3) / 2

/-- The theorem to be proved -/
theorem triangle_properties (t : Triangle) 
  (h : satisfies_conditions t) : 
  t.A = π / 3 ∧ t.a + t.b + t.c = 5 + Real.sqrt 7 :=
sorry

end triangle_properties_l3558_355859


namespace machine_purchase_price_machine_purchase_price_is_14000_l3558_355897

/-- Proves that the purchase price of a machine is 14000 given the specified conditions -/
theorem machine_purchase_price : ℝ → Prop :=
  fun purchase_price =>
    let repair_cost : ℝ := 5000
    let transport_cost : ℝ := 1000
    let profit_percentage : ℝ := 50
    let selling_price : ℝ := 30000
    let total_cost : ℝ := purchase_price + repair_cost + transport_cost
    let profit_multiplier : ℝ := (100 + profit_percentage) / 100
    selling_price = profit_multiplier * total_cost →
    purchase_price = 14000

/-- The purchase price of the machine is 14000 -/
theorem machine_purchase_price_is_14000 : machine_purchase_price 14000 := by
  sorry

end machine_purchase_price_machine_purchase_price_is_14000_l3558_355897


namespace sum_equals_twelve_l3558_355872

theorem sum_equals_twelve (a b c : ℕ) (h : 28 * a + 30 * b + 31 * c = 365) : a + b + c = 12 := by
  sorry

end sum_equals_twelve_l3558_355872


namespace vector_equations_true_l3558_355842

-- Define a vector space over the real numbers
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define vectors a and b
variable (a b : V)

-- Define points A, B, and C
variable (A B C : V)

-- Theorem statement
theorem vector_equations_true :
  (a + b = b + a) ∧
  (-(-a) = a) ∧
  ((B - A) + (C - B) + (A - C) = 0) ∧
  (a + (-a) = 0) := by
  sorry

end vector_equations_true_l3558_355842


namespace unique_polynomial_composition_l3558_355883

theorem unique_polynomial_composition (a b c : ℝ) (n : ℕ) (h : a ≠ 0) :
  ∃! Q : Polynomial ℝ,
    (Polynomial.degree Q = n) ∧
    (∀ x : ℝ, Q.eval (a * x^2 + b * x + c) = a * (Q.eval x)^2 + b * (Q.eval x) + c) := by
  sorry

end unique_polynomial_composition_l3558_355883


namespace event_selection_methods_l3558_355857

def total_students : ℕ := 5
def selected_students : ℕ := 4
def num_days : ℕ := 3
def friday_attendees : ℕ := 2
def saturday_attendees : ℕ := 1
def sunday_attendees : ℕ := 1

theorem event_selection_methods :
  (Nat.choose total_students friday_attendees) *
  (Nat.choose (total_students - friday_attendees) saturday_attendees) *
  (Nat.choose (total_students - friday_attendees - saturday_attendees) sunday_attendees) = 60 := by
  sorry

end event_selection_methods_l3558_355857


namespace lily_bouquet_cost_l3558_355805

/-- The cost of a bouquet is directly proportional to the number of lilies it contains. -/
def DirectlyProportional (cost : ℝ → ℝ) (lilies : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, cost x = k * lilies x

theorem lily_bouquet_cost 
  (cost : ℝ → ℝ) 
  (lilies : ℝ → ℝ) 
  (h_prop : DirectlyProportional cost lilies)
  (h_18 : cost 18 = 30)
  (h_pos : ∀ x, lilies x > 0) :
  cost 27 = 45 := by
sorry

end lily_bouquet_cost_l3558_355805


namespace pauls_books_count_paul_has_151_books_l3558_355835

/-- Calculates the total number of books Paul has after buying new ones -/
def total_books (initial_books new_books : ℕ) : ℕ :=
  initial_books + new_books

/-- Theorem: Paul's total books equal the sum of initial and new books -/
theorem pauls_books_count (initial_books new_books : ℕ) :
  total_books initial_books new_books = initial_books + new_books :=
by sorry

/-- Theorem: Paul now has 151 books -/
theorem paul_has_151_books :
  total_books 50 101 = 151 :=
by sorry

end pauls_books_count_paul_has_151_books_l3558_355835


namespace regular_hexagon_interior_angle_measure_l3558_355828

/-- The measure of an interior angle of a regular hexagon -/
def regular_hexagon_interior_angle : ℝ := 120

/-- A regular hexagon has 6 sides -/
def regular_hexagon_sides : ℕ := 6

/-- Theorem: The measure of each interior angle of a regular hexagon is 120 degrees -/
theorem regular_hexagon_interior_angle_measure :
  regular_hexagon_interior_angle = (((regular_hexagon_sides - 2) * 180) : ℝ) / regular_hexagon_sides :=
by sorry

end regular_hexagon_interior_angle_measure_l3558_355828


namespace exists_point_on_h_with_sum_40_l3558_355892

-- Define the function g
variable (g : ℝ → ℝ)

-- Define the function h in terms of g
def h (g : ℝ → ℝ) (x : ℝ) : ℝ := (g x - 2)^2

-- Theorem statement
theorem exists_point_on_h_with_sum_40 (g : ℝ → ℝ) (h : ℝ → ℝ) 
  (h_def : ∀ x, h x = (g x - 2)^2) (g_val : g 4 = 8) :
  ∃ x y, h x = y ∧ x + y = 40 := by
  sorry

end exists_point_on_h_with_sum_40_l3558_355892


namespace adams_pants_l3558_355816

/-- The number of pairs of pants Adam initially took out -/
def P : ℕ := 31

/-- The number of jumpers Adam took out -/
def jumpers : ℕ := 4

/-- The number of pajama sets Adam took out -/
def pajama_sets : ℕ := 4

/-- The number of t-shirts Adam took out -/
def tshirts : ℕ := 20

/-- The number of friends who donate the same amount as Adam -/
def friends : ℕ := 3

/-- The total number of articles of clothing being donated -/
def total_donated : ℕ := 126

theorem adams_pants :
  P = 31 ∧
  (4 * (P + jumpers + 2 * pajama_sets + tshirts) / 2 = total_donated) :=
sorry

end adams_pants_l3558_355816


namespace reciprocal_sum_equals_one_l3558_355834

theorem reciprocal_sum_equals_one : 
  1/2 + 1/3 + 1/12 + 1/18 + 1/72 + 1/108 + 1/216 = 1 := by
  sorry

end reciprocal_sum_equals_one_l3558_355834


namespace two_numbers_with_specific_means_l3558_355837

theorem two_numbers_with_specific_means : ∃ a b : ℝ, 
  a > 0 ∧ b > 0 ∧ 
  Real.sqrt (a * b) = Real.sqrt 5 ∧
  2 / (1 / a + 1 / b) = 2 ∧
  a = (5 + Real.sqrt 5) / 2 ∧ 
  b = (5 - Real.sqrt 5) / 2 := by
sorry

end two_numbers_with_specific_means_l3558_355837


namespace math_team_combinations_l3558_355856

theorem math_team_combinations (girls : ℕ) (boys : ℕ) : girls = 4 → boys = 5 → (girls.choose 3) * (boys.choose 1) = 20 := by
  sorry

end math_team_combinations_l3558_355856


namespace probability_no_consecutive_ones_l3558_355894

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

/-- Number of valid sequences of length n -/
def validSequences (n : ℕ) : ℕ := fib (n + 2)

/-- Total number of possible sequences of length n -/
def totalSequences (n : ℕ) : ℕ := 2^n

/-- The probability of not having two consecutive 1s in a sequence of length 15 -/
theorem probability_no_consecutive_ones : 
  (validSequences 15 : ℚ) / (totalSequences 15 : ℚ) = 1597 / 32768 := by
  sorry

#eval validSequences 15
#eval totalSequences 15

end probability_no_consecutive_ones_l3558_355894


namespace condition_neither_sufficient_nor_necessary_l3558_355889

theorem condition_neither_sufficient_nor_necessary :
  ¬(∀ a b : ℝ, (a ≠ 5 ∧ b ≠ -5) → a + b ≠ 0) ∧
  ¬(∀ a b : ℝ, a + b ≠ 0 → (a ≠ 5 ∧ b ≠ -5)) :=
by sorry

end condition_neither_sufficient_nor_necessary_l3558_355889


namespace square_sum_from_system_l3558_355810

theorem square_sum_from_system (x y : ℝ) 
  (h1 : x * y = 12)
  (h2 : x^2 * y + x * y^2 + x + y = 120) :
  x^2 + y^2 = (14400 - 4056) / 169 := by
sorry

end square_sum_from_system_l3558_355810


namespace subset_complement_implies_a_negative_l3558_355861

theorem subset_complement_implies_a_negative 
  (I : Set ℝ) 
  (A B : Set ℝ) 
  (a : ℝ) 
  (h_I : I = Set.univ) 
  (h_A : A = {x : ℝ | x ≤ a + 1}) 
  (h_B : B = {x : ℝ | x ≥ 1}) 
  (h_subset : A ⊆ (I \ B)) : 
  a < 0 := by
sorry

end subset_complement_implies_a_negative_l3558_355861


namespace inequality_proof_l3558_355853

theorem inequality_proof (n : ℕ+) : (2*n+1)^(n:ℕ) ≥ (2*n)^(n:ℕ) + (2*n-1)^(n:ℕ) := by
  sorry

end inequality_proof_l3558_355853


namespace total_is_700_l3558_355824

/-- The number of magazines Marie sold -/
def magazines : ℕ := 425

/-- The number of newspapers Marie sold -/
def newspapers : ℕ := 275

/-- The total number of reading materials Marie sold -/
def total_reading_materials : ℕ := magazines + newspapers

/-- Proof that the total number of reading materials sold is 700 -/
theorem total_is_700 : total_reading_materials = 700 := by
  sorry

end total_is_700_l3558_355824


namespace retailer_profit_is_ten_percent_l3558_355867

/-- Calculates the profit percentage for a retailer selling pens --/
def profit_percentage (buy_quantity : ℕ) (buy_price : ℕ) (discount : ℚ) : ℚ :=
  let cost_price := buy_price
  let selling_price_per_pen := 1 - discount
  let total_selling_price := buy_quantity * selling_price_per_pen
  let profit := total_selling_price - cost_price
  (profit / cost_price) * 100

/-- Theorem stating that the profit percentage is 10% for the given conditions --/
theorem retailer_profit_is_ten_percent :
  profit_percentage 40 36 (1/100) = 10 := by
  sorry

#eval profit_percentage 40 36 (1/100)

end retailer_profit_is_ten_percent_l3558_355867


namespace range_of_m_l3558_355820

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∀ x : ℝ, x ∈ Set.Icc (-1) 2 → m ≤ x^2

def q (m l : ℝ) : Prop := ∀ x : ℝ, x^2 + m*x + l > 0

-- Theorem statement
theorem range_of_m (m l : ℝ) (h : p m ∧ q m l) : m ∈ Set.Ioo (-2) 1 := by
  sorry

end range_of_m_l3558_355820


namespace max_wooden_pencils_l3558_355812

theorem max_wooden_pencils :
  ∀ (m w : ℕ),
  m + w = 72 →
  ∃ (p : ℕ), Nat.Prime p ∧ m = w + p →
  w ≤ 35 :=
by sorry

end max_wooden_pencils_l3558_355812


namespace det_positive_for_special_matrix_l3558_355800

open Matrix

theorem det_positive_for_special_matrix {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ) 
  (h : A + Aᵀ = 1) : 
  0 < det A := by
  sorry

end det_positive_for_special_matrix_l3558_355800


namespace parentheses_removal_l3558_355852

theorem parentheses_removal (x y : ℝ) : x - 2 * (y - 1) = x - 2 * y + 2 := by
  sorry

end parentheses_removal_l3558_355852


namespace inequality_system_solution_l3558_355868

theorem inequality_system_solution (x a : ℝ) : 
  (1 - x < -1) ∧ (x - 1 > a) ∧ (∀ y, (1 - y < -1 ∧ y - 1 > a) ↔ y > 2) →
  a ≤ 1 := by
  sorry

end inequality_system_solution_l3558_355868


namespace unique_prime_in_form_l3558_355815

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → n % d ≠ 0

def number_form (A : ℕ) : ℕ := 305200 + A

theorem unique_prime_in_form :
  ∃! A : ℕ, A < 10 ∧ is_prime (number_form A) ∧ number_form A = 305201 :=
sorry

end unique_prime_in_form_l3558_355815


namespace second_team_odd_second_team_odd_approx_l3558_355865

/-- Calculates the odd for the second team in a four-team soccer bet -/
theorem second_team_odd (odd1 odd3 odd4 bet_amount expected_winnings : ℝ) : ℝ :=
  let total_odds := expected_winnings / bet_amount
  let second_team_odd := total_odds / (odd1 * odd3 * odd4)
  second_team_odd

/-- The calculated odd for the second team is approximately 5.23 -/
theorem second_team_odd_approx :
  let odd1 : ℝ := 1.28
  let odd3 : ℝ := 3.25
  let odd4 : ℝ := 2.05
  let bet_amount : ℝ := 5.00
  let expected_winnings : ℝ := 223.0072
  abs (second_team_odd odd1 odd3 odd4 bet_amount expected_winnings - 5.23) < 0.01 := by
  sorry

end second_team_odd_second_team_odd_approx_l3558_355865


namespace hcd_7350_150_minus_12_l3558_355838

theorem hcd_7350_150_minus_12 : Nat.gcd 7350 150 - 12 = 138 := by
  sorry

end hcd_7350_150_minus_12_l3558_355838


namespace cauchy_schwarz_like_inequality_l3558_355879

theorem cauchy_schwarz_like_inequality (a b c d : ℝ) :
  (a^2 + b^2) * (c^2 + d^2) ≥ (a*c + b*d)^2 := by
  sorry

end cauchy_schwarz_like_inequality_l3558_355879


namespace johns_furniture_purchase_l3558_355881

theorem johns_furniture_purchase (chair_price table_price couch_price total_price : ℝ) :
  chair_price > 0 ∧
  table_price = 3 * chair_price ∧
  couch_price = 5 * table_price ∧
  total_price = chair_price + table_price + couch_price ∧
  total_price = 380 →
  couch_price = 300 := by
  sorry

end johns_furniture_purchase_l3558_355881


namespace special_numbers_are_one_and_nine_l3558_355826

/-- The number of divisors of a natural number -/
def divisor_count (n : ℕ) : ℕ := sorry

/-- The set of natural numbers that are equal to the square of their divisor count -/
def special_numbers : Set ℕ := {n : ℕ | n = (divisor_count n)^2}

/-- Theorem stating that the set of special numbers is equal to {1, 9} -/
theorem special_numbers_are_one_and_nine : special_numbers = {1, 9} := by sorry

end special_numbers_are_one_and_nine_l3558_355826


namespace base_10_to_base_7_l3558_355893

theorem base_10_to_base_7 : 
  (1 * 7^4 + 0 * 7^3 + 2 * 7^2 + 2 * 7^1 + 4 * 7^0 : ℕ) = 2468 := by
  sorry

#eval 1 * 7^4 + 0 * 7^3 + 2 * 7^2 + 2 * 7^1 + 4 * 7^0

end base_10_to_base_7_l3558_355893


namespace min_value_x_plus_2y_l3558_355882

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2/x + 1/y = 4) :
  ∀ a b : ℝ, a > 0 → b > 0 → 2/a + 1/b = 4 → x + 2*y ≤ a + 2*b :=
by sorry

end min_value_x_plus_2y_l3558_355882


namespace surface_area_unchanged_l3558_355845

/-- Represents a rectangular prism --/
structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ
  length_pos : length > 0
  width_pos : width > 0
  height_pos : height > 0

/-- Calculates the surface area of a rectangular prism --/
def surfaceArea (prism : RectangularPrism) : ℝ :=
  2 * (prism.length * prism.width + prism.length * prism.height + prism.width * prism.height)

/-- Represents the result of cutting a unit cube from a rectangular prism --/
structure CutPrism where
  original : RectangularPrism
  cut_from_corner : Bool

/-- Calculates the surface area of a prism after a unit cube is cut from it --/
def surfaceAreaAfterCut (cut : CutPrism) : ℝ :=
  surfaceArea cut.original

theorem surface_area_unchanged (cut : CutPrism) :
  surfaceArea cut.original = surfaceAreaAfterCut cut :=
sorry

#check surface_area_unchanged

end surface_area_unchanged_l3558_355845


namespace money_never_equal_l3558_355840

/-- Represents the amount of money in Kiriels and Dariels -/
structure Money where
  kiriels : ℕ
  dariels : ℕ

/-- Represents a currency exchange operation -/
inductive Exchange
  | KirielToDariel : ℕ → Exchange
  | DarielToKiriel : ℕ → Exchange

/-- Applies a single exchange operation to a Money value -/
def applyExchange (m : Money) (e : Exchange) : Money :=
  match e with
  | Exchange.KirielToDariel n => 
      ⟨m.kiriels - n, m.dariels + 10 * n⟩
  | Exchange.DarielToKiriel n => 
      ⟨m.kiriels + 10 * n, m.dariels - n⟩

/-- Applies a sequence of exchanges to an initial Money value -/
def applyExchanges (initial : Money) : List Exchange → Money
  | [] => initial
  | e :: es => applyExchanges (applyExchange initial e) es

theorem money_never_equal :
  ∀ (exchanges : List Exchange),
    let final := applyExchanges ⟨0, 1⟩ exchanges
    final.kiriels ≠ final.dariels :=
  sorry


end money_never_equal_l3558_355840


namespace pond_to_field_ratio_l3558_355823

/-- Represents a rectangular field with a square pond inside -/
structure FieldWithPond where
  field_length : ℝ
  field_width : ℝ
  pond_side : ℝ
  length_double_width : field_length = 2 * field_width
  field_length_16 : field_length = 16
  pond_side_8 : pond_side = 8

/-- The ratio of the pond area to the field area is 1:2 -/
theorem pond_to_field_ratio (f : FieldWithPond) : 
  (f.pond_side ^ 2) / (f.field_length * f.field_width) = 1 / 2 := by
  sorry

#check pond_to_field_ratio

end pond_to_field_ratio_l3558_355823


namespace polynomial_equality_l3558_355874

theorem polynomial_equality (x y : ℝ) (h : x + y = -1) :
  x^4 + 5*x^3*y + x^2*y + 8*x^2*y^2 + x*y^2 + 5*x*y^3 + y^4 = 1 := by
  sorry

end polynomial_equality_l3558_355874


namespace book_arrangement_count_l3558_355830

def num_math_books : ℕ := 4
def num_history_books : ℕ := 6

def alternating_arrangement (m h : ℕ) : Prop :=
  m > 1 ∧ h > 0 ∧ m = h + 1

theorem book_arrangement_count :
  alternating_arrangement num_math_books num_history_books →
  (num_math_books * (num_math_books - 1) * (num_history_books.factorial / (num_history_books - (num_math_books - 1)).factorial)) = 2880 :=
by sorry

end book_arrangement_count_l3558_355830


namespace ellipse_axis_sum_l3558_355804

/-- Proves that for an ellipse with given conditions, a + b = 40 -/
theorem ellipse_axis_sum (M N a b : ℝ) : 
  M > 0 → 
  N > 0 → 
  M = π * a * b → 
  N = π * (a + b) → 
  M / N = 10 → 
  a = b → 
  a + b = 40 := by
sorry

end ellipse_axis_sum_l3558_355804


namespace quadratic_points_range_l3558_355858

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 4 * a * x + 3

-- Define the theorem
theorem quadratic_points_range (a m y₁ y₂ : ℝ) :
  a > 0 →
  y₁ < y₂ →
  f a (m - 1) = y₁ →
  f a m = y₂ →
  m > -3/2 :=
sorry

end quadratic_points_range_l3558_355858
