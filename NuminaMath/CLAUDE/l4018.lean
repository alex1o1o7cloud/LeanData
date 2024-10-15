import Mathlib

namespace NUMINAMATH_CALUDE_solve_equation_l4018_401879

theorem solve_equation (x : ℚ) : x / 4 - x - 3 / 6 = 1 → x = -2 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l4018_401879


namespace NUMINAMATH_CALUDE_cone_base_radius_l4018_401876

/-- Given a cone with slant height 5 cm and lateral surface area 15π cm², 
    prove that the radius of its base is 3 cm. -/
theorem cone_base_radius (l : ℝ) (L : ℝ) (π : ℝ) (r : ℝ) : 
  l = 5 →
  L = 15 * π →
  L = π * r * l →
  r = 3 := by sorry

end NUMINAMATH_CALUDE_cone_base_radius_l4018_401876


namespace NUMINAMATH_CALUDE_jims_journey_distance_l4018_401852

/-- The total distance of Jim's journey -/
def total_distance (driven : ℕ) (remaining : ℕ) : ℕ := driven + remaining

/-- Theorem stating the total distance of Jim's journey -/
theorem jims_journey_distance :
  total_distance 642 558 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_jims_journey_distance_l4018_401852


namespace NUMINAMATH_CALUDE_slices_per_pie_is_four_l4018_401875

/-- The number of slices in a whole pie at a pie shop -/
def slices_per_pie : ℕ := sorry

/-- The price of a single slice of pie in dollars -/
def price_per_slice : ℕ := 5

/-- The number of whole pies sold -/
def pies_sold : ℕ := 9

/-- The total revenue in dollars from selling all pies -/
def total_revenue : ℕ := 180

/-- Theorem stating that the number of slices per pie is 4 -/
theorem slices_per_pie_is_four :
  slices_per_pie = 4 :=
by sorry

end NUMINAMATH_CALUDE_slices_per_pie_is_four_l4018_401875


namespace NUMINAMATH_CALUDE_first_walk_time_l4018_401877

/-- Represents a walk with its speed and distance -/
structure Walk where
  speed : ℝ
  distance : ℝ

/-- Proves that the time taken for the first walk is 2 hours given the problem conditions -/
theorem first_walk_time (first_walk : Walk) (second_walk : Walk) 
  (h1 : first_walk.speed = 3)
  (h2 : second_walk.speed = 4)
  (h3 : second_walk.distance = first_walk.distance + 2)
  (h4 : first_walk.distance / first_walk.speed + second_walk.distance / second_walk.speed = 4) :
  first_walk.distance / first_walk.speed = 2 := by
  sorry


end NUMINAMATH_CALUDE_first_walk_time_l4018_401877


namespace NUMINAMATH_CALUDE_exists_number_with_digit_sum_property_l4018_401885

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem stating the existence of a number with specific digit sum properties -/
theorem exists_number_with_digit_sum_property :
  ∃ n : ℕ, sum_of_digits n = 1000 ∧ sum_of_digits (n^2) = sum_of_digits (1000^2) := by
  sorry

end NUMINAMATH_CALUDE_exists_number_with_digit_sum_property_l4018_401885


namespace NUMINAMATH_CALUDE_grape_juice_percentage_l4018_401822

/-- Calculates the percentage of grape juice in a mixture after adding pure grape juice -/
theorem grape_juice_percentage 
  (original_volume : ℝ) 
  (original_percentage : ℝ) 
  (added_volume : ℝ) : 
  original_volume = 40 →
  original_percentage = 0.1 →
  added_volume = 10 →
  (original_volume * original_percentage + added_volume) / (original_volume + added_volume) = 0.28 := by
sorry


end NUMINAMATH_CALUDE_grape_juice_percentage_l4018_401822


namespace NUMINAMATH_CALUDE_parabola_directrix_p_l4018_401804

/-- A parabola with equation y^2 = 2px and directrix x = -1 has p = 2 -/
theorem parabola_directrix_p (y x p : ℝ) : 
  (∀ y, y^2 = 2*p*x) →  -- Parabola equation
  (x = -1)             -- Directrix equation
  → p = 2 := by sorry

end NUMINAMATH_CALUDE_parabola_directrix_p_l4018_401804


namespace NUMINAMATH_CALUDE_min_value_of_expression_lower_bound_achievable_l4018_401869

theorem min_value_of_expression (x y : ℝ) : 
  (x^2 * y^2 - 1)^2 + (x^2 + y^2)^2 ≥ 1 := by
  sorry

theorem lower_bound_achievable : 
  ∃ x y : ℝ, (x^2 * y^2 - 1)^2 + (x^2 + y^2)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_lower_bound_achievable_l4018_401869


namespace NUMINAMATH_CALUDE_equation_solution_l4018_401817

theorem equation_solution : ∃ x : ℝ, (4 / (x - 1) + 1 / (1 - x) = 1) ∧ (x = 4) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l4018_401817


namespace NUMINAMATH_CALUDE_expression_evaluation_l4018_401891

theorem expression_evaluation : (4^4 - 4*(4-1)^4)^4 = 21381376 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4018_401891


namespace NUMINAMATH_CALUDE_pure_imaginary_ratio_l4018_401870

theorem pure_imaginary_ratio (p q : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) 
  (h : ∃ (y : ℝ), (3 - 4 * Complex.I) * (p + q * Complex.I) = y * Complex.I) : 
  p / q = -4 / 3 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_ratio_l4018_401870


namespace NUMINAMATH_CALUDE_choir_competition_score_l4018_401824

/-- Calculates the final score of a choir competition team given their individual scores and weights -/
def final_score (song_content : ℝ) (singing_skills : ℝ) (spirit : ℝ) : ℝ :=
  0.3 * song_content + 0.5 * singing_skills + 0.2 * spirit

/-- Theorem stating that the final score for the given team is 93 -/
theorem choir_competition_score :
  final_score 90 94 95 = 93 := by
  sorry

#eval final_score 90 94 95

end NUMINAMATH_CALUDE_choir_competition_score_l4018_401824


namespace NUMINAMATH_CALUDE_multiply_fractions_l4018_401831

theorem multiply_fractions : (12 : ℚ) * (1 / 17) * 34 = 24 := by
  sorry

end NUMINAMATH_CALUDE_multiply_fractions_l4018_401831


namespace NUMINAMATH_CALUDE_normal_probability_theorem_l4018_401872

/-- The standard normal cumulative distribution function -/
noncomputable def Φ : ℝ → ℝ := sorry

/-- Normal distribution probability density function -/
def normal_pdf (μ σ : ℝ) (x : ℝ) : ℝ := sorry

theorem normal_probability_theorem (ξ : ℝ → ℝ) (μ σ : ℝ) 
  (h_normal : ∀ x, normal_pdf μ σ x = sorry)  -- ξ follows N(μ, σ²)
  (h_mean : ∫ x, x * normal_pdf μ σ x = 3)    -- E[ξ] = 3
  (h_var : ∫ x, (x - μ)^2 * normal_pdf μ σ x = 1)  -- D[ξ] = 1
  : ∫ x in Set.Ioo (-1) 1, normal_pdf μ σ x = Φ (-4) - Φ (-2) :=
sorry

end NUMINAMATH_CALUDE_normal_probability_theorem_l4018_401872


namespace NUMINAMATH_CALUDE_friend_savings_rate_l4018_401810

/-- Proves that given the initial amounts and saving rates, after 25 weeks,
    both people will have the same amount of money if and only if the friend saves 5 dollars per week. -/
theorem friend_savings_rate (your_initial : ℕ) (friend_initial : ℕ) (your_weekly_savings : ℕ) (weeks : ℕ) :
  your_initial = 160 →
  friend_initial = 210 →
  your_weekly_savings = 7 →
  weeks = 25 →
  (your_initial + your_weekly_savings * weeks = friend_initial + 5 * weeks) :=
by sorry

end NUMINAMATH_CALUDE_friend_savings_rate_l4018_401810


namespace NUMINAMATH_CALUDE_pet_store_ratio_l4018_401803

theorem pet_store_ratio (num_cats : ℕ) (num_dogs : ℕ) : 
  (num_cats : ℚ) / num_dogs = 3 / 4 →
  num_cats = 18 →
  num_dogs = 24 := by
sorry

end NUMINAMATH_CALUDE_pet_store_ratio_l4018_401803


namespace NUMINAMATH_CALUDE_parallel_vectors_tan_double_angle_l4018_401864

/-- Given two vectors a and b in R², where a is parallel to b, 
    prove that tan(2θ) = -4/3 -/
theorem parallel_vectors_tan_double_angle (θ : ℝ) 
  (a b : ℝ × ℝ) 
  (ha : a = (Real.sin θ, 2)) 
  (hb : b = (Real.cos θ, 1)) 
  (hparallel : ∃ (k : ℝ), a = k • b) : 
  Real.tan (2 * θ) = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_tan_double_angle_l4018_401864


namespace NUMINAMATH_CALUDE_triangle_side_count_l4018_401895

/-- The number of integer values for the third side of a triangle with sides 15 and 40 -/
def triangleSideCount : ℕ := by
  sorry

theorem triangle_side_count :
  triangleSideCount = 29 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_count_l4018_401895


namespace NUMINAMATH_CALUDE_worker_c_time_l4018_401898

/-- The time taken by worker c to complete the work alone, given the conditions -/
def time_c (time_abc time_a time_b : ℚ) : ℚ :=
  1 / (1 / time_abc - 1 / time_a - 1 / time_b)

/-- Theorem stating that under given conditions, worker c takes 18 days to finish the work alone -/
theorem worker_c_time (time_abc time_a time_b : ℚ) 
  (h_abc : time_abc = 4)
  (h_a : time_a = 12)
  (h_b : time_b = 9) :
  time_c time_abc time_a time_b = 18 := by
  sorry

#eval time_c 4 12 9

end NUMINAMATH_CALUDE_worker_c_time_l4018_401898


namespace NUMINAMATH_CALUDE_election_majority_l4018_401894

theorem election_majority (total_votes : ℕ) (winning_percentage : ℚ) : 
  total_votes = 900 →
  winning_percentage = 70 / 100 →
  (total_votes : ℚ) * winning_percentage - (total_votes : ℚ) * (1 - winning_percentage) = 360 := by
sorry

end NUMINAMATH_CALUDE_election_majority_l4018_401894


namespace NUMINAMATH_CALUDE_max_sum_given_constraints_l4018_401853

theorem max_sum_given_constraints (x y : ℝ) 
  (h1 : x^2 + y^2 = 100) 
  (h2 : x * y = 40) : 
  x + y ≤ 6 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_max_sum_given_constraints_l4018_401853


namespace NUMINAMATH_CALUDE_product_of_fractions_l4018_401840

theorem product_of_fractions : 
  (((3^4 - 1) / (3^4 + 1)) * ((4^4 - 1) / (4^4 + 1)) * ((5^4 - 1) / (5^4 + 1)) * 
   ((6^4 - 1) / (6^4 + 1)) * ((7^4 - 1) / (7^4 + 1))) = 25 / 210 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l4018_401840


namespace NUMINAMATH_CALUDE_lineup_count_l4018_401865

/-- The number of ways to choose a starting lineup for a football team -/
def choose_lineup (total_members : ℕ) (offensive_linemen : ℕ) : ℕ :=
  offensive_linemen * (total_members - 1) * (total_members - 2) * (total_members - 3) * (total_members - 4)

/-- Theorem: The number of ways to choose a starting lineup for a team of 15 members
    with 5 offensive linemen is 109200 -/
theorem lineup_count :
  choose_lineup 15 5 = 109200 := by
  sorry

end NUMINAMATH_CALUDE_lineup_count_l4018_401865


namespace NUMINAMATH_CALUDE_special_square_midpoint_sum_l4018_401821

/-- A square in the first quadrant with specific points on its sides -/
structure SpecialSquare where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  in_first_quadrant : A.1 ≥ 0 ∧ A.2 ≥ 0 ∧ B.1 ≥ 0 ∧ B.2 ≥ 0 ∧ C.1 ≥ 0 ∧ C.2 ≥ 0 ∧ D.1 ≥ 0 ∧ D.2 ≥ 0
  is_square : (A.1 - B.1)^2 + (A.2 - B.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2 ∧
              (B.1 - C.1)^2 + (B.2 - C.2)^2 = (C.1 - D.1)^2 + (C.2 - D.2)^2 ∧
              (C.1 - D.1)^2 + (C.2 - D.2)^2 = (D.1 - A.1)^2 + (D.2 - A.2)^2
  point_on_AD : ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ (2, 0) = (t * A.1 + (1 - t) * D.1, t * A.2 + (1 - t) * D.2)
  point_on_BC : ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ (6, 0) = (t * B.1 + (1 - t) * C.1, t * B.2 + (1 - t) * C.2)
  point_on_AB : ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ (10, 0) = (t * A.1 + (1 - t) * B.1, t * A.2 + (1 - t) * B.2)
  point_on_CD : ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ (14, 0) = (t * C.1 + (1 - t) * D.1, t * C.2 + (1 - t) * D.2)

/-- The sum of coordinates of the midpoint of the special square is 10 -/
theorem special_square_midpoint_sum (sq : SpecialSquare) :
  (sq.A.1 + sq.C.1) / 2 + (sq.A.2 + sq.C.2) / 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_special_square_midpoint_sum_l4018_401821


namespace NUMINAMATH_CALUDE_initial_chickens_is_300_l4018_401880

/-- Represents the initial state and conditions of the poultry farm problem --/
structure PoultryFarm where
  initial_turkeys : ℕ
  initial_guinea_fowls : ℕ
  daily_loss_chickens : ℕ
  daily_loss_turkeys : ℕ
  daily_loss_guinea_fowls : ℕ
  duration_days : ℕ
  total_remaining : ℕ

/-- Calculates the initial number of chickens given the farm conditions --/
def calculate_initial_chickens (farm : PoultryFarm) : ℕ :=
  let remaining_turkeys := farm.initial_turkeys - farm.daily_loss_turkeys * farm.duration_days
  let remaining_guinea_fowls := farm.initial_guinea_fowls - farm.daily_loss_guinea_fowls * farm.duration_days
  let remaining_chickens := farm.total_remaining - remaining_turkeys - remaining_guinea_fowls
  remaining_chickens + farm.daily_loss_chickens * farm.duration_days

/-- Theorem stating that the initial number of chickens is 300 --/
theorem initial_chickens_is_300 (farm : PoultryFarm)
  (h1 : farm.initial_turkeys = 200)
  (h2 : farm.initial_guinea_fowls = 80)
  (h3 : farm.daily_loss_chickens = 20)
  (h4 : farm.daily_loss_turkeys = 8)
  (h5 : farm.daily_loss_guinea_fowls = 5)
  (h6 : farm.duration_days = 7)
  (h7 : farm.total_remaining = 349) :
  calculate_initial_chickens farm = 300 := by
  sorry

end NUMINAMATH_CALUDE_initial_chickens_is_300_l4018_401880


namespace NUMINAMATH_CALUDE_man_walking_speed_l4018_401832

/-- Calculates the walking speed of a man given the following conditions:
  * The man walks at a constant speed
  * He takes a 5-minute rest after every kilometer
  * He covers 5 kilometers in 50 minutes
-/
theorem man_walking_speed (total_time : ℝ) (total_distance : ℝ) (rest_time : ℝ) 
  (rest_frequency : ℝ) (h1 : total_time = 50) (h2 : total_distance = 5) 
  (h3 : rest_time = 5) (h4 : rest_frequency = 1) : 
  (total_distance / ((total_time - (rest_time * (total_distance - 1))) / 60)) = 10 := by
  sorry

#check man_walking_speed

end NUMINAMATH_CALUDE_man_walking_speed_l4018_401832


namespace NUMINAMATH_CALUDE_solve_star_equation_l4018_401828

-- Define the custom operation ※
def star (a b : ℚ) : ℚ := a + b

-- State the theorem
theorem solve_star_equation :
  ∃ x : ℚ, star 4 (star x 3) = 1 ∧ x = -6 := by
  sorry

end NUMINAMATH_CALUDE_solve_star_equation_l4018_401828


namespace NUMINAMATH_CALUDE_total_tickets_is_900_l4018_401867

/-- Represents the total number of tickets sold at a movie theater. -/
def total_tickets (adult_price child_price : ℕ) (total_revenue child_tickets : ℕ) : ℕ :=
  let adult_tickets := (total_revenue - child_price * child_tickets) / adult_price
  adult_tickets + child_tickets

/-- Theorem stating that the total number of tickets sold is 900. -/
theorem total_tickets_is_900 :
  total_tickets 7 4 5100 400 = 900 := by
  sorry

end NUMINAMATH_CALUDE_total_tickets_is_900_l4018_401867


namespace NUMINAMATH_CALUDE_total_boxes_theorem_l4018_401843

/-- Calculates the total number of boxes sold over three days given the conditions --/
def total_boxes_sold (friday_boxes : ℕ) : ℕ :=
  let saturday_boxes := friday_boxes + (friday_boxes * 50 / 100)
  let sunday_boxes := saturday_boxes - (saturday_boxes * 30 / 100)
  friday_boxes + saturday_boxes + sunday_boxes

/-- Proves that the total number of boxes sold over three days is 213 --/
theorem total_boxes_theorem : total_boxes_sold 60 = 213 := by
  sorry

#eval total_boxes_sold 60

end NUMINAMATH_CALUDE_total_boxes_theorem_l4018_401843


namespace NUMINAMATH_CALUDE_binary_to_hex_l4018_401862

-- Define the binary number
def binary_num : ℕ := 1011001

-- Define the hexadecimal number
def hex_num : ℕ := 0x59

-- Theorem stating that the binary number is equal to the hexadecimal number
theorem binary_to_hex : binary_num = hex_num := by
  sorry

end NUMINAMATH_CALUDE_binary_to_hex_l4018_401862


namespace NUMINAMATH_CALUDE_circle_equation_l4018_401888

/-- Given a circle with center (a, 5-3a) that passes through (0, 0) and (3, -1),
    prove that its equation is (x - 1)^2 + (y - 2)^2 = 5 -/
theorem circle_equation (a : ℝ) :
  (∀ x y : ℝ, (x - a)^2 + (y - (5 - 3*a))^2 = a^2 + (5 - 3*a)^2) →
  (a^2 + (5 - 3*a)^2 = 3^2 + (-1 - (5 - 3*a))^2) →
  (∀ x y : ℝ, (x - 1)^2 + (y - 2)^2 = 5) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l4018_401888


namespace NUMINAMATH_CALUDE_num_successful_sequences_l4018_401802

/-- Represents the number of cards in the game -/
def num_cards : ℕ := 13

/-- Represents the number of cards that need to be flipped for success -/
def cards_to_flip : ℕ := 12

/-- Represents the number of choices for each flip after the first -/
def choices_per_flip : ℕ := 2

/-- Represents the rules of the card flipping game -/
structure CardGame where
  cards : Fin num_cards → Bool
  is_valid_flip : Fin num_cards → Bool

/-- Theorem stating the number of successful flip sequences -/
theorem num_successful_sequences (game : CardGame) :
  (num_cards : ℕ) * (choices_per_flip ^ (cards_to_flip - 1) : ℕ) = 26624 := by
  sorry

end NUMINAMATH_CALUDE_num_successful_sequences_l4018_401802


namespace NUMINAMATH_CALUDE_terminal_side_half_angle_l4018_401818

def is_in_first_quadrant (α : Real) : Prop :=
  ∃ k : ℤ, 2 * k * Real.pi < α ∧ α < 2 * k * Real.pi + Real.pi / 2

def is_in_first_or_third_quadrant (α : Real) : Prop :=
  ∃ k : ℤ, k * Real.pi < α ∧ α < k * Real.pi + Real.pi / 2

theorem terminal_side_half_angle (α : Real) :
  is_in_first_quadrant α → is_in_first_or_third_quadrant (α / 2) :=
by sorry

end NUMINAMATH_CALUDE_terminal_side_half_angle_l4018_401818


namespace NUMINAMATH_CALUDE_triangle_inequality_l4018_401854

theorem triangle_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) :
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l4018_401854


namespace NUMINAMATH_CALUDE_trapezoid_garden_bases_l4018_401889

theorem trapezoid_garden_bases :
  let area : ℕ := 1350
  let altitude : ℕ := 45
  let valid_pair (b₁ b₂ : ℕ) : Prop :=
    area = (altitude * (b₁ + b₂)) / 2 ∧
    b₁ % 9 = 0 ∧
    b₂ % 9 = 0 ∧
    b₁ > 0 ∧
    b₂ > 0
  ∃! (pairs : Finset (ℕ × ℕ)), pairs.card = 3 ∧ ∀ (p : ℕ × ℕ), p ∈ pairs ↔ valid_pair p.1 p.2 :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_garden_bases_l4018_401889


namespace NUMINAMATH_CALUDE_min_value_theorem_min_value_achievable_l4018_401819

theorem min_value_theorem (x : ℝ) : 
  (x^2 + 19) / Real.sqrt (x^2 + 8) ≥ 2 * Real.sqrt 11 := by
  sorry

theorem min_value_achievable : 
  ∃ x : ℝ, (x^2 + 19) / Real.sqrt (x^2 + 8) = 2 * Real.sqrt 11 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_min_value_achievable_l4018_401819


namespace NUMINAMATH_CALUDE_probability_of_red_from_B_mutually_exclusive_events_l4018_401820

structure Bag where
  red : ℕ
  white : ℕ
  black : ℕ

def bagA : Bag := ⟨5, 2, 3⟩
def bagB : Bag := ⟨4, 3, 3⟩

def totalBalls (bag : Bag) : ℕ := bag.red + bag.white + bag.black

def P_A1 : ℚ := bagA.red / totalBalls bagA
def P_A2 : ℚ := bagA.white / totalBalls bagA
def P_A3 : ℚ := bagA.black / totalBalls bagA

def P_B_given_A1 : ℚ := (bagB.red + 1) / (totalBalls bagB + 1)
def P_B_given_A2 : ℚ := bagB.red / (totalBalls bagB + 1)
def P_B_given_A3 : ℚ := bagB.red / (totalBalls bagB + 1)

def P_B : ℚ := P_A1 * P_B_given_A1 + P_A2 * P_B_given_A2 + P_A3 * P_B_given_A3

theorem probability_of_red_from_B : P_B = 9 / 22 := by sorry

theorem mutually_exclusive_events : P_A1 + P_A2 + P_A3 = 1 := by sorry

end NUMINAMATH_CALUDE_probability_of_red_from_B_mutually_exclusive_events_l4018_401820


namespace NUMINAMATH_CALUDE_speed_increase_percentage_l4018_401871

def initial_speed : ℝ := 80
def training_weeks : ℕ := 16
def speed_gain_per_week : ℝ := 1

def final_speed : ℝ := initial_speed + (speed_gain_per_week * training_weeks)

theorem speed_increase_percentage :
  (final_speed - initial_speed) / initial_speed * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_speed_increase_percentage_l4018_401871


namespace NUMINAMATH_CALUDE_hyperbola_center_l4018_401805

/-- The center of a hyperbola is the midpoint of its foci -/
theorem hyperbola_center (f1 f2 : ℝ × ℝ) :
  let center := ((f1.1 + f2.1) / 2, (f1.2 + f2.2) / 2)
  f1 = (3, 2) ∧ f2 = (11, 6) →
  center = (7, 4) := by sorry

end NUMINAMATH_CALUDE_hyperbola_center_l4018_401805


namespace NUMINAMATH_CALUDE_power_equality_implies_n_equals_four_l4018_401874

theorem power_equality_implies_n_equals_four (n : ℕ) : 4^8 = 16^n → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_implies_n_equals_four_l4018_401874


namespace NUMINAMATH_CALUDE_rationalize_denominator_l4018_401887

theorem rationalize_denominator :
  ∀ x : ℝ, x > 0 → (30 : ℝ) / (5 - Real.sqrt x) = -30 - 6 * Real.sqrt x → x = 30 :=
by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l4018_401887


namespace NUMINAMATH_CALUDE_condition_relationship_l4018_401845

theorem condition_relationship (A B : Prop) 
  (h : (¬A → ¬B) ∧ ¬(¬B → ¬A)) : 
  (B → A) ∧ ¬(A → B) := by
  sorry

end NUMINAMATH_CALUDE_condition_relationship_l4018_401845


namespace NUMINAMATH_CALUDE_origin_outside_circle_l4018_401883

/-- The circle equation: x^2 + y^2 - 2ax - 2y + (a-1)^2 = 0 -/
def circle_equation (x y a : ℝ) : Prop :=
  x^2 + y^2 - 2*a*x - 2*y + (a-1)^2 = 0

/-- A point (x, y) is outside the circle if the left-hand side of the equation is positive -/
def is_outside_circle (x y a : ℝ) : Prop :=
  x^2 + y^2 - 2*a*x - 2*y + (a-1)^2 > 0

theorem origin_outside_circle (a : ℝ) (h : a > 1) :
  is_outside_circle 0 0 a :=
sorry

end NUMINAMATH_CALUDE_origin_outside_circle_l4018_401883


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l4018_401848

theorem point_in_second_quadrant (m : ℝ) :
  (m - 1 < 0 ∧ 3 > 0) → m < 1 := by
  sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l4018_401848


namespace NUMINAMATH_CALUDE_ellipse_sum_specific_l4018_401841

/-- Represents an ellipse with center (h, k) and semi-axes lengths a and b -/
structure Ellipse where
  h : ℝ
  k : ℝ
  a : ℝ
  b : ℝ

/-- The sum of h, k, a, and b for a specific ellipse -/
def ellipse_sum (e : Ellipse) : ℝ :=
  e.h + e.k + e.a + e.b

theorem ellipse_sum_specific : ∃ (e : Ellipse), 
  e.h = 3 ∧ 
  e.k = -1 ∧ 
  e.a = 6 ∧ 
  e.b = 4 ∧ 
  ellipse_sum e = 12 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_sum_specific_l4018_401841


namespace NUMINAMATH_CALUDE_A_completes_in_20_days_l4018_401846

/-- The number of days B takes to complete the project alone -/
def B_days : ℝ := 30

/-- The number of days A and B work together -/
def together_days : ℝ := 8

/-- The number of days B works alone after A quits -/
def B_alone_days : ℝ := 10

/-- The total amount of work (100% of the project) -/
def total_work : ℝ := 1

/-- Theorem stating that A can complete the project alone in 20 days -/
theorem A_completes_in_20_days :
  ∃ A_days : ℝ,
    A_days = 20 ∧
    together_days * (1 / A_days + 1 / B_days) + B_alone_days * (1 / B_days) = total_work :=
by sorry

end NUMINAMATH_CALUDE_A_completes_in_20_days_l4018_401846


namespace NUMINAMATH_CALUDE_unique_solution_factorial_equation_l4018_401893

theorem unique_solution_factorial_equation :
  ∃! (a b : ℕ), a^2 + 2 = Nat.factorial b :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_solution_factorial_equation_l4018_401893


namespace NUMINAMATH_CALUDE_complex_expression_evaluation_l4018_401826

theorem complex_expression_evaluation :
  ∀ (a b : ℂ),
  a = 5 - 3*I →
  b = 2 + 4*I →
  3*a - 4*b = 7 - 25*I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_expression_evaluation_l4018_401826


namespace NUMINAMATH_CALUDE_speech_arrangement_count_l4018_401825

theorem speech_arrangement_count :
  let total_male : ℕ := 4
  let total_female : ℕ := 3
  let selected_male : ℕ := 3
  let selected_female : ℕ := 2
  let total_selected : ℕ := selected_male + selected_female

  (Nat.choose total_male selected_male) *
  (Nat.choose total_female selected_female) *
  (Nat.factorial selected_male) *
  (Nat.factorial (total_selected - 1)) = 864 :=
by sorry

end NUMINAMATH_CALUDE_speech_arrangement_count_l4018_401825


namespace NUMINAMATH_CALUDE_seating_arrangements_l4018_401859

theorem seating_arrangements (n : ℕ) (h : n = 6) : Nat.factorial n = 720 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangements_l4018_401859


namespace NUMINAMATH_CALUDE_fraction_equality_l4018_401827

-- Define the @ operation
def at_op (a b : ℝ) : ℝ := a * b - b^3

-- Define the # operation
def hash_op (a b : ℝ) : ℝ := a + b - a * b^2

-- Theorem statement
theorem fraction_equality : 
  let a : ℝ := 3
  let b : ℝ := 2
  (at_op a b) / (hash_op a b) = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l4018_401827


namespace NUMINAMATH_CALUDE_f_monotonicity_and_a_range_l4018_401884

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x - a) / Real.log x

theorem f_monotonicity_and_a_range :
  (∀ x₁ x₂, e < x₁ ∧ x₁ < x₂ → f 0 x₁ < f 0 x₂) ∧ 
  (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 → f 0 x₁ > f 0 x₂) ∧
  (∀ x₁ x₂, 1 < x₁ ∧ x₁ < x₂ ∧ x₂ < e → f 0 x₁ > f 0 x₂) ∧
  (∀ a, (∀ x, 1 < x → f a x > Real.sqrt x) → a ≤ 1) :=
sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_a_range_l4018_401884


namespace NUMINAMATH_CALUDE_salary_increase_after_four_years_l4018_401892

theorem salary_increase_after_four_years (annual_raise : ℝ) (h : annual_raise = 0.1) :
  (1 + annual_raise)^4 - 1 > 0.45 := by sorry

end NUMINAMATH_CALUDE_salary_increase_after_four_years_l4018_401892


namespace NUMINAMATH_CALUDE_five_digit_reverse_multiply_nine_l4018_401837

theorem five_digit_reverse_multiply_nine :
  ∃! n : ℕ, 
    10000 ≤ n ∧ n < 100000 ∧
    (∃ a b c d e : ℕ,
      n = 10000 * a + 1000 * b + 100 * c + 10 * d + e ∧
      9 * n = 10000 * e + 1000 * d + 100 * c + 10 * b + a ∧
      a ≠ 0) ∧
    n = 10989 :=
by sorry

end NUMINAMATH_CALUDE_five_digit_reverse_multiply_nine_l4018_401837


namespace NUMINAMATH_CALUDE_yellow_apples_count_l4018_401850

theorem yellow_apples_count (green red total : ℕ) 
  (h1 : green = 2) 
  (h2 : red = 3) 
  (h3 : total = 19) : 
  total - (green + red) = 14 := by
  sorry

end NUMINAMATH_CALUDE_yellow_apples_count_l4018_401850


namespace NUMINAMATH_CALUDE_biff_break_even_time_biff_break_even_time_is_three_l4018_401809

/-- Calculates the break-even time for Biff's bus trip -/
theorem biff_break_even_time 
  (ticket_cost : ℝ) 
  (snacks_cost : ℝ) 
  (headphones_cost : ℝ) 
  (work_rate : ℝ) 
  (wifi_cost : ℝ) : ℝ :=
  let total_cost := ticket_cost + snacks_cost + headphones_cost
  let net_hourly_rate := work_rate - wifi_cost
  total_cost / net_hourly_rate

/-- Proves that Biff's break-even time is 3 hours given the specific costs and rates -/
theorem biff_break_even_time_is_three :
  biff_break_even_time 11 3 16 12 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_biff_break_even_time_biff_break_even_time_is_three_l4018_401809


namespace NUMINAMATH_CALUDE_min_value_theorem_l4018_401836

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + 1/b) * (b + 4/a) ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l4018_401836


namespace NUMINAMATH_CALUDE_parabola_transformation_l4018_401851

/-- A parabola is above a line if it opens upwards and doesn't intersect the line. -/
def parabola_above_line (a b c : ℝ) : Prop :=
  a > 0 ∧ (b - c)^2 < 4*a*c

theorem parabola_transformation (a b c : ℝ) 
  (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) 
  (above : parabola_above_line a b c) : 
  parabola_above_line c (-b) a :=
sorry

end NUMINAMATH_CALUDE_parabola_transformation_l4018_401851


namespace NUMINAMATH_CALUDE_fourth_row_middle_cells_l4018_401890

/-- Represents a letter in the grid -/
inductive Letter : Type
| A | B | C | D | E | F

/-- Represents a position in the grid -/
structure Position :=
  (row : Fin 6)
  (col : Fin 6)

/-- Represents the 6x6 grid -/
def Grid := Position → Letter

/-- Checks if a 2x3 rectangle is valid (no repeats) -/
def validRectangle (g : Grid) (topLeft : Position) : Prop :=
  ∀ (i j : Fin 2) (k : Fin 3),
    g ⟨topLeft.row + i, topLeft.col + k⟩ ≠ g ⟨topLeft.row + j, topLeft.col + k⟩ ∨ i = j

/-- Checks if the entire grid is valid -/
def validGrid (g : Grid) : Prop :=
  (∀ r : Fin 6, ∀ i j : Fin 6, g ⟨r, i⟩ ≠ g ⟨r, j⟩ ∨ i = j) ∧  -- No repeats in rows
  (∀ c : Fin 6, ∀ i j : Fin 6, g ⟨i, c⟩ ≠ g ⟨j, c⟩ ∨ i = j) ∧  -- No repeats in columns
  (∀ r c : Fin 2, validRectangle g ⟨3*r, 3*c⟩)                 -- Valid 2x3 rectangles

/-- The main theorem -/
theorem fourth_row_middle_cells (g : Grid) (h : validGrid g) :
  g ⟨3, 1⟩ = Letter.E ∧
  g ⟨3, 2⟩ = Letter.D ∧
  g ⟨3, 3⟩ = Letter.C ∧
  g ⟨3, 4⟩ = Letter.F :=
by sorry

end NUMINAMATH_CALUDE_fourth_row_middle_cells_l4018_401890


namespace NUMINAMATH_CALUDE_modulus_of_complex_l4018_401878

theorem modulus_of_complex (a b : ℝ) (i : ℂ) :
  i * i = -1 →
  a / (1 - i) = 1 - b * i →
  Complex.abs (a + b * i) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_l4018_401878


namespace NUMINAMATH_CALUDE_thirty_two_team_tournament_games_l4018_401813

/-- Represents a single-elimination tournament -/
structure SingleEliminationTournament where
  num_teams : ℕ

/-- The number of games played in a single-elimination tournament -/
def games_played (t : SingleEliminationTournament) : ℕ :=
  t.num_teams - 1

theorem thirty_two_team_tournament_games :
  ∀ (t : SingleEliminationTournament),
    t.num_teams = 32 →
    games_played t = 31 := by
  sorry

end NUMINAMATH_CALUDE_thirty_two_team_tournament_games_l4018_401813


namespace NUMINAMATH_CALUDE_factorization_constant_l4018_401873

theorem factorization_constant (c : ℝ) : 
  (∀ x, x^2 - 4*x + c = (x - 1) * (x - 3)) → c = 3 := by
  sorry

end NUMINAMATH_CALUDE_factorization_constant_l4018_401873


namespace NUMINAMATH_CALUDE_max_value_g_and_range_of_a_l4018_401844

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (a * x)

def g (a : ℝ) (x : ℝ) : ℝ := x^2 * f a x

def h (a : ℝ) (x : ℝ) : ℝ := x^2 / f a x - 1

theorem max_value_g_and_range_of_a :
  (∀ x > 0, g (-2) x ≤ Real.exp (-2)) ∧
  (∀ a : ℝ, (∃ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 16 ∧ h a x₁ = 0 ∧ h a x₂ = 0) →
    1/2 * Real.log 2 < a ∧ a < 2 / Real.exp 1) :=
by sorry

end NUMINAMATH_CALUDE_max_value_g_and_range_of_a_l4018_401844


namespace NUMINAMATH_CALUDE_sphere_radius_change_factor_l4018_401866

theorem sphere_radius_change_factor (initial_area new_area : ℝ) 
  (h1 : initial_area = 2464)
  (h2 : new_area = 9856) : 
  let factor := (new_area / initial_area).sqrt
  factor = 2 := by sorry

end NUMINAMATH_CALUDE_sphere_radius_change_factor_l4018_401866


namespace NUMINAMATH_CALUDE_trig_identities_l4018_401815

theorem trig_identities (α β γ : Real) (h : α + β + γ = Real.pi) :
  (Real.sin α)^2 + (Real.sin β)^2 - (Real.sin γ)^2 = 2 * Real.sin α * Real.sin β * Real.cos γ ∧
  (Real.cos α)^2 + (Real.cos β)^2 - (Real.cos γ)^2 = 1 - 2 * Real.sin α * Real.sin β * Real.cos γ := by
  sorry

end NUMINAMATH_CALUDE_trig_identities_l4018_401815


namespace NUMINAMATH_CALUDE_starting_lineup_count_l4018_401839

def total_players : ℕ := 20
def num_goalies : ℕ := 1
def num_forwards : ℕ := 6
def num_defenders : ℕ := 4

def starting_lineup_combinations : ℕ := 
  (total_players.choose num_goalies) * 
  ((total_players - num_goalies).choose num_forwards) * 
  ((total_players - num_goalies - num_forwards).choose num_defenders)

theorem starting_lineup_count : starting_lineup_combinations = 387889200 := by
  sorry

end NUMINAMATH_CALUDE_starting_lineup_count_l4018_401839


namespace NUMINAMATH_CALUDE_line_equations_correct_l4018_401835

/-- Triangle ABC with vertices A(4,0), B(8,10), and C(0,6) -/
structure Triangle :=
  (A : ℝ × ℝ)
  (B : ℝ × ℝ)
  (C : ℝ × ℝ)

/-- Definition of the specific triangle in the problem -/
def triangle : Triangle :=
  { A := (4, 0),
    B := (8, 10),
    C := (0, 6) }

/-- Equation of a line in the form ax + by + c = 0 -/
structure LineEquation :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)

/-- The equation of the line passing through A and parallel to BC -/
def line_parallel_to_BC : LineEquation :=
  { a := 1,
    b := -1,
    c := -4 }

/-- The equation of the line containing the altitude on edge AC -/
def altitude_on_AC : LineEquation :=
  { a := 2,
    b := -3,
    c := -8 }

/-- Theorem stating the correctness of the line equations -/
theorem line_equations_correct (t : Triangle) :
  t = triangle →
  (line_parallel_to_BC.a * t.A.1 + line_parallel_to_BC.b * t.A.2 + line_parallel_to_BC.c = 0) ∧
  (altitude_on_AC.a * t.B.1 + altitude_on_AC.b * t.B.2 + altitude_on_AC.c = 0) :=
by sorry

end NUMINAMATH_CALUDE_line_equations_correct_l4018_401835


namespace NUMINAMATH_CALUDE_triangle_ratio_l4018_401861

theorem triangle_ratio (A B C : ℝ) (a b c : ℝ) : 
  A = π/3 → b = 1 → c = 4 → 
  (a + b + c) / (Real.sin A + Real.sin B + Real.sin C) = 2 * Real.sqrt 39 / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_ratio_l4018_401861


namespace NUMINAMATH_CALUDE_terry_age_l4018_401812

/-- Given the following conditions:
    1. In 10 years, Terry will be 4 times Nora's current age.
    2. Nora is currently 10 years old.
    3. In 5 years, Nora will be half Sam's age.
    4. Sam is currently 6 years older than Terry.
    Prove that Terry is currently 19 years old. -/
theorem terry_age (nora_age : ℕ) (terry_future_age : ℕ → ℕ) (sam_age : ℕ → ℕ) :
  nora_age = 10 ∧
  terry_future_age 10 = 4 * nora_age ∧
  sam_age 5 = 2 * (nora_age + 5) ∧
  sam_age 0 = terry_future_age 0 + 6 →
  terry_future_age 0 = 19 := by
  sorry

end NUMINAMATH_CALUDE_terry_age_l4018_401812


namespace NUMINAMATH_CALUDE_arrow_connections_theorem_l4018_401863

/-- The number of ways to connect 2n points on a circle with n arrows -/
def arrow_connections (n : ℕ) : ℕ := Nat.choose (2 * n) n

/-- Theorem statement for the arrow connection problem -/
theorem arrow_connections_theorem (n : ℕ) (h : n > 0) :
  arrow_connections n = Nat.choose (2 * n) n :=
by sorry

end NUMINAMATH_CALUDE_arrow_connections_theorem_l4018_401863


namespace NUMINAMATH_CALUDE_policeman_hats_l4018_401847

theorem policeman_hats (simpson_hats : ℕ) (obrien_hats_after : ℕ) : 
  simpson_hats = 15 →
  obrien_hats_after = 34 →
  ∃ (obrien_hats_before : ℕ), 
    obrien_hats_before > 2 * simpson_hats ∧
    obrien_hats_before = obrien_hats_after + 1 ∧
    obrien_hats_before - 2 * simpson_hats = 5 :=
by sorry

end NUMINAMATH_CALUDE_policeman_hats_l4018_401847


namespace NUMINAMATH_CALUDE_max_regions_is_nine_l4018_401830

/-- Represents a square in a 2D plane -/
structure Square where
  -- We don't need to define the internals of the square for this problem

/-- The number of regions created by two intersecting squares -/
def num_regions (s1 s2 : Square) : ℕ := sorry

/-- The maximum number of regions that can be created by two intersecting squares -/
def max_regions : ℕ := sorry

/-- Theorem: The maximum number of regions created by two intersecting squares is 9 -/
theorem max_regions_is_nine : max_regions = 9 := by sorry

end NUMINAMATH_CALUDE_max_regions_is_nine_l4018_401830


namespace NUMINAMATH_CALUDE_complementary_angles_sum_l4018_401896

theorem complementary_angles_sum (a b : ℝ) : 
  a > 0 → b > 0 → a / b = 3 / 5 → a + b = 90 → a + b = 90 := by sorry

end NUMINAMATH_CALUDE_complementary_angles_sum_l4018_401896


namespace NUMINAMATH_CALUDE_shoe_price_after_changes_lous_shoe_price_l4018_401897

/-- The price of shoes after a price increase followed by a discount -/
theorem shoe_price_after_changes (initial_price : ℝ) 
  (increase_percent : ℝ) (discount_percent : ℝ) : ℝ := by
  -- Define the price after increase
  let price_after_increase := initial_price * (1 + increase_percent / 100)
  -- Define the final price after discount
  let final_price := price_after_increase * (1 - discount_percent / 100)
  -- Prove that when initial_price = 40, increase_percent = 10, and discount_percent = 10,
  -- the final_price is 39.60
  sorry

/-- The specific case for Lou's Fine Shoes -/
theorem lous_shoe_price : 
  shoe_price_after_changes 40 10 10 = 39.60 := by sorry

end NUMINAMATH_CALUDE_shoe_price_after_changes_lous_shoe_price_l4018_401897


namespace NUMINAMATH_CALUDE_triangle_area_l4018_401814

theorem triangle_area (A B C : ℝ) (a b c : ℝ) : 
  b * Real.cos C = 3 * a * Real.cos B - c * Real.cos B →
  a * c * Real.cos B = 2 →
  (1/2) * a * c * Real.sin B = 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_l4018_401814


namespace NUMINAMATH_CALUDE_intersecting_lines_m_value_l4018_401899

/-- Given three lines that intersect at a single point, prove that the value of m is -22/7 -/
theorem intersecting_lines_m_value (x y : ℚ) :
  y = 4 * x - 8 ∧
  y = -3 * x + 9 ∧
  y = 2 * x + m →
  m = -22 / 7 := by
  sorry

end NUMINAMATH_CALUDE_intersecting_lines_m_value_l4018_401899


namespace NUMINAMATH_CALUDE_diamond_four_three_l4018_401800

def diamond (a b : ℝ) : ℝ := 4*a + 3*b - 2*a*b

theorem diamond_four_three : diamond 4 3 = 1 := by sorry

end NUMINAMATH_CALUDE_diamond_four_three_l4018_401800


namespace NUMINAMATH_CALUDE_no_unique_solution_implies_a_equals_four_l4018_401858

/-- Given two linear equations in two variables, this function determines if they have a unique solution. -/
def hasUniqueSolution (a k : ℝ) : Prop :=
  ∃! (x y : ℝ), a * (3 * x + 4 * y) = 36 ∧ k * x + 12 * y = 30

/-- The theorem states that when k = 9 and the equations don't have a unique solution, a must equal 4. -/
theorem no_unique_solution_implies_a_equals_four :
  ∀ (a : ℝ), (¬ hasUniqueSolution a 9) → a = 4 := by
  sorry

#check no_unique_solution_implies_a_equals_four

end NUMINAMATH_CALUDE_no_unique_solution_implies_a_equals_four_l4018_401858


namespace NUMINAMATH_CALUDE_smallest_multiple_l4018_401834

theorem smallest_multiple : ∃ n : ℕ, 
  n > 0 ∧ 
  19 ∣ n ∧ 
  n % 97 = 3 ∧ 
  ∀ m : ℕ, m > 0 → 19 ∣ m → m % 97 = 3 → n ≤ m ∧ 
  n = 494 := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_l4018_401834


namespace NUMINAMATH_CALUDE_library_book_distribution_l4018_401838

/-- The number of ways to distribute n identical objects between two locations,
    with at least one object in each location. -/
def distributionWays (n : ℕ) : ℕ :=
  if n ≥ 2 then n - 1 else 0

/-- The problem statement as a theorem -/
theorem library_book_distribution :
  distributionWays 8 = 7 := by
  sorry

end NUMINAMATH_CALUDE_library_book_distribution_l4018_401838


namespace NUMINAMATH_CALUDE_optimal_z_maximizes_optimal_z_satisfies_condition_l4018_401849

open Complex

/-- The complex number that maximizes the given expression -/
def optimal_z : ℂ := -4 + I

theorem optimal_z_maximizes (z : ℂ) (h : arg (z + 3) = Real.pi * (3 / 4)) :
  1 / (abs (z + 6) + abs (z - 3 * I)) ≤ 1 / (abs (optimal_z + 6) + abs (optimal_z - 3 * I)) :=
by sorry

theorem optimal_z_satisfies_condition :
  arg (optimal_z + 3) = Real.pi * (3 / 4) :=
by sorry

end NUMINAMATH_CALUDE_optimal_z_maximizes_optimal_z_satisfies_condition_l4018_401849


namespace NUMINAMATH_CALUDE_min_value_of_expression_l4018_401808

theorem min_value_of_expression (x y k : ℝ) : 
  (x * y + k)^2 + (x - y)^2 ≥ 0 ∧ 
  ∃ (x y k : ℝ), (x * y + k)^2 + (x - y)^2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l4018_401808


namespace NUMINAMATH_CALUDE_discriminant_of_specific_quadratic_l4018_401811

/-- The discriminant of a quadratic equation ax^2 + bx + c -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- Theorem: The discriminant of 5x^2 - 9x + 1 is 61 -/
theorem discriminant_of_specific_quadratic : discriminant 5 (-9) 1 = 61 := by
  sorry

end NUMINAMATH_CALUDE_discriminant_of_specific_quadratic_l4018_401811


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_one_minimum_a_for_always_greater_than_three_l4018_401816

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + x + a

-- Theorem 1
theorem solution_set_when_a_is_one :
  {x : ℝ | f 1 x ≥ 3} = {x : ℝ | x ≥ 1 ∨ x ≤ -2} := by sorry

-- Theorem 2
theorem minimum_a_for_always_greater_than_three :
  (∀ x : ℝ, f a x ≥ 3) ↔ a ≥ 13/4 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_one_minimum_a_for_always_greater_than_three_l4018_401816


namespace NUMINAMATH_CALUDE_polynomial_factorization_l4018_401860

theorem polynomial_factorization (x : ℝ) :
  x^2 + 6*x + 9 - 100*x^4 = (-10*x^2 + x + 3) * (10*x^2 + x + 3) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l4018_401860


namespace NUMINAMATH_CALUDE_six_students_adjacent_permutations_l4018_401829

/-- The number of permutations of n elements where two specific elements must be adjacent -/
def adjacent_permutations (n : ℕ) : ℕ :=
  2 * Nat.factorial (n - 1)

/-- Theorem: The number of permutations of 6 students where 2 specific students
    must stand next to each other is 240 -/
theorem six_students_adjacent_permutations :
  adjacent_permutations 6 = 240 := by
  sorry

#eval adjacent_permutations 6

end NUMINAMATH_CALUDE_six_students_adjacent_permutations_l4018_401829


namespace NUMINAMATH_CALUDE_proposition_implication_l4018_401801

theorem proposition_implication (P : ℕ → Prop) 
  (h1 : ∀ k : ℕ, k > 0 → (P k → P (k + 1)))
  (h2 : ¬ P 4) : 
  ¬ P 3 := by
  sorry

end NUMINAMATH_CALUDE_proposition_implication_l4018_401801


namespace NUMINAMATH_CALUDE_ascent_speed_l4018_401868

/-- 
Given a round trip journey with:
- Total time of 8 hours
- Ascent time of 5 hours
- Descent time of 3 hours
- Average speed for the entire journey of 3 km/h
Prove that the average speed during the ascent is 2.4 km/h
-/
theorem ascent_speed (total_time : ℝ) (ascent_time : ℝ) (descent_time : ℝ) (avg_speed : ℝ) :
  total_time = 8 →
  ascent_time = 5 →
  descent_time = 3 →
  avg_speed = 3 →
  (avg_speed * total_time / 2) / ascent_time = 2.4 := by
  sorry

end NUMINAMATH_CALUDE_ascent_speed_l4018_401868


namespace NUMINAMATH_CALUDE_no_real_roots_condition_l4018_401823

/-- A quadratic equation of the form ax^2 + bx + c = 0 has no real roots if and only if its discriminant is negative. -/
axiom no_real_roots_iff_neg_discriminant {a b c : ℝ} (ha : a ≠ 0) :
  (∀ x, a * x^2 + b * x + c ≠ 0) ↔ b^2 - 4*a*c < 0

/-- For the quadratic equation 6x^2 - 5x + a = 0 to have no real roots, a must be greater than 25/24. -/
theorem no_real_roots_condition (a : ℝ) :
  (∀ x, 6 * x^2 - 5 * x + a ≠ 0) ↔ a > 25/24 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_condition_l4018_401823


namespace NUMINAMATH_CALUDE_area_ADBC_l4018_401855

/-- Given a triangle ABC in the xy-plane where:
    A is at the origin (0, 0)
    B lies on the positive x-axis
    C is in the upper right quadrant
    ∠A = 30°, ∠B = 60°, ∠C = 90°
    Length BC = 1
    D is the intersection of the angle bisector of ∠C with the y-axis

    The area of quadrilateral ADBC is (5√3 + 9) / 4 -/
theorem area_ADBC (A B C D : ℝ × ℝ) : 
  A = (0, 0) →
  B.1 > 0 ∧ B.2 = 0 →
  C.1 > 0 ∧ C.2 > 0 →
  Real.cos (π/6) * (C.1 - A.1) = Real.sin (π/6) * (C.2 - A.2) →
  Real.cos (π/3) * (C.1 - B.1) = Real.sin (π/3) * (C.2 - B.2) →
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = 1 →
  D.1 = 0 →
  (C.2 - D.2) / (C.1 - D.1) = (C.2 - A.2) / (C.1 - A.1) →
  let area := abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)) / 2 +
               abs ((C.1 - A.1) * (D.2 - A.2) - (D.1 - A.1) * (C.2 - A.2)) / 2
  area = (5 * Real.sqrt 3 + 9) / 4 := by
  sorry


end NUMINAMATH_CALUDE_area_ADBC_l4018_401855


namespace NUMINAMATH_CALUDE_common_chord_length_l4018_401881

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 4*y = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 12 = 0

-- Define the common chord
def common_chord (x y : ℝ) : Prop := x - 2*y + 6 = 0

-- Theorem statement
theorem common_chord_length : 
  ∃ (a b c d : ℝ), 
    (circle1 a b ∧ circle1 c d ∧ common_chord a b ∧ common_chord c d) →
    ((a - c)^2 + (b - d)^2)^(1/2 : ℝ) = 4 * 2^(1/2 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_common_chord_length_l4018_401881


namespace NUMINAMATH_CALUDE_one_zero_quadratic_l4018_401842

def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * x + 1

theorem one_zero_quadratic (a : ℝ) :
  (∃! x, f a x = 0) → (a = 0 ∨ a = 1) :=
by sorry

end NUMINAMATH_CALUDE_one_zero_quadratic_l4018_401842


namespace NUMINAMATH_CALUDE_min_distance_point_l4018_401857

/-- Given a triangle ABC with vertices (x₁, y₁), (x₂, y₂), (x₃, y₃), 
    the point P that minimizes the sum of squares of distances from P to the three vertices 
    has coordinates ((x₁ + x₂ + x₃) / 3, (y₁ + y₂ + y₃) / 3) -/
theorem min_distance_point (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) :
  let dist_sum_sq (x y : ℝ) := 
    (x - x₁)^2 + (y - y₁)^2 + (x - x₂)^2 + (y - y₂)^2 + (x - x₃)^2 + (y - y₃)^2
  ∃ (x y : ℝ), (∀ (u v : ℝ), dist_sum_sq x y ≤ dist_sum_sq u v) ∧ 
    x = (x₁ + x₂ + x₃) / 3 ∧ y = (y₁ + y₂ + y₃) / 3 := by
  sorry

end NUMINAMATH_CALUDE_min_distance_point_l4018_401857


namespace NUMINAMATH_CALUDE_tank_emptying_time_l4018_401882

/-- Proves that a tank with given properties empties in 6 hours due to a leak alone -/
theorem tank_emptying_time (tank_capacity : ℝ) (inlet_rate : ℝ) (emptying_time_with_inlet : ℝ) :
  tank_capacity = 4320 →
  inlet_rate = 3 →
  emptying_time_with_inlet = 8 →
  ∃ (leak_rate : ℝ),
    leak_rate > 0 ∧
    (leak_rate - inlet_rate) * (emptying_time_with_inlet * 60) = tank_capacity ∧
    tank_capacity / leak_rate / 60 = 6 :=
by sorry

end NUMINAMATH_CALUDE_tank_emptying_time_l4018_401882


namespace NUMINAMATH_CALUDE_largest_number_in_ratio_l4018_401886

theorem largest_number_in_ratio (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  (b / a = 4 / 3) →
  (c / a = 6 / 3) →
  a * b * c = 1944 →
  max a (max b c) = 18 := by
sorry

end NUMINAMATH_CALUDE_largest_number_in_ratio_l4018_401886


namespace NUMINAMATH_CALUDE_remainder_of_587421_div_6_l4018_401807

theorem remainder_of_587421_div_6 : 587421 % 6 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_587421_div_6_l4018_401807


namespace NUMINAMATH_CALUDE_min_values_ab_and_a_plus_2b_l4018_401833

theorem min_values_ab_and_a_plus_2b (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_eq : a * b = 2 * a + b) :
  (∀ x y, 0 < x ∧ 0 < y ∧ x * y = 2 * x + y → 8 ≤ x * y) ∧
  (∃ x y, 0 < x ∧ 0 < y ∧ x * y = 2 * x + y ∧ x * y = 8) ∧
  (∀ x y, 0 < x ∧ 0 < y ∧ x * y = 2 * x + y → 9 ≤ x + 2 * y) ∧
  (∃ x y, 0 < x ∧ 0 < y ∧ x * y = 2 * x + y ∧ x + 2 * y = 9) := by
sorry

end NUMINAMATH_CALUDE_min_values_ab_and_a_plus_2b_l4018_401833


namespace NUMINAMATH_CALUDE_w_squared_value_l4018_401806

theorem w_squared_value (w : ℝ) (h : (2*w + 10)^2 = (5*w + 15)*(w + 6)) : 
  w^2 = (90 + 10*Real.sqrt 65) / 4 := by
sorry

end NUMINAMATH_CALUDE_w_squared_value_l4018_401806


namespace NUMINAMATH_CALUDE_max_silver_tokens_l4018_401856

/-- Represents the number of tokens Kevin has -/
structure TokenCount where
  red : ℕ
  blue : ℕ
  silver : ℕ

/-- Represents an exchange booth -/
structure ExchangeBooth where
  input_color : String
  input_amount : ℕ
  output_silver : ℕ
  output_other_color : String
  output_other_amount : ℕ

/-- Function to perform a single exchange -/
def exchange (tokens : TokenCount) (booth : ExchangeBooth) : TokenCount :=
  sorry

/-- Function to check if an exchange is possible -/
def can_exchange (tokens : TokenCount) (booth : ExchangeBooth) : Bool :=
  sorry

/-- Function to perform all possible exchanges -/
def perform_all_exchanges (tokens : TokenCount) (booths : List ExchangeBooth) : TokenCount :=
  sorry

/-- The main theorem stating the maximum number of silver tokens Kevin can obtain -/
theorem max_silver_tokens : 
  let initial_tokens : TokenCount := ⟨100, 100, 0⟩
  let booth1 : ExchangeBooth := ⟨"red", 3, 1, "blue", 2⟩
  let booth2 : ExchangeBooth := ⟨"blue", 4, 1, "red", 2⟩
  let final_tokens := perform_all_exchanges initial_tokens [booth1, booth2]
  final_tokens.silver = 132 :=
sorry

end NUMINAMATH_CALUDE_max_silver_tokens_l4018_401856
