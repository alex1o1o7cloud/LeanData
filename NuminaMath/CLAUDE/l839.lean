import Mathlib

namespace series_sum_equals_one_over_200_l839_83927

/-- The nth term of the series -/
def seriesTerm (n : ℕ) : ℚ :=
  (4 * n + 3) / ((4 * n + 1)^2 * (4 * n + 5)^2)

/-- The sum of the series -/
noncomputable def seriesSum : ℚ := ∑' n, seriesTerm n

/-- Theorem stating that the sum of the series is 1/200 -/
theorem series_sum_equals_one_over_200 : seriesSum = 1 / 200 := by
  sorry

end series_sum_equals_one_over_200_l839_83927


namespace opposite_sides_equal_implies_parallelogram_condition_b_implies_parallelogram_l839_83990

/-- A quadrilateral in a 2D plane --/
structure Quadrilateral (V : Type*) [AddCommGroup V] [Module ℝ V] :=
  (A B C D : V)

/-- Definition of a parallelogram --/
def is_parallelogram {V : Type*} [AddCommGroup V] [Module ℝ V] (q : Quadrilateral V) : Prop :=
  q.A - q.B = q.D - q.C ∧ q.A - q.D = q.B - q.C

/-- Theorem: If opposite sides of a quadrilateral are equal, it is a parallelogram --/
theorem opposite_sides_equal_implies_parallelogram 
  {V : Type*} [AddCommGroup V] [Module ℝ V] (q : Quadrilateral V) :
  q.A - q.D = q.B - q.C → q.A - q.B = q.D - q.C → is_parallelogram q :=
by sorry

/-- Main theorem: If AD=BC and AB=DC, then ABCD is a parallelogram --/
theorem condition_b_implies_parallelogram 
  {V : Type*} [AddCommGroup V] [Module ℝ V] (q : Quadrilateral V) :
  q.A - q.D = q.B - q.C → q.A - q.B = q.D - q.C → is_parallelogram q :=
by sorry

end opposite_sides_equal_implies_parallelogram_condition_b_implies_parallelogram_l839_83990


namespace father_twice_as_old_father_four_times_now_l839_83978

/-- Represents the current age of the father -/
def father_age : ℕ := 40

/-- Represents the current age of the daughter -/
def daughter_age : ℕ := 10

/-- Represents the number of years until the father is twice as old as the daughter -/
def years_until_twice : ℕ := 20

/-- Theorem stating that after the specified number of years, the father will be twice as old as the daughter -/
theorem father_twice_as_old :
  father_age + years_until_twice = 2 * (daughter_age + years_until_twice) :=
sorry

/-- Theorem stating that the father is currently 4 times as old as the daughter -/
theorem father_four_times_now :
  father_age = 4 * daughter_age :=
sorry

end father_twice_as_old_father_four_times_now_l839_83978


namespace cube_root_of_a_plus_one_l839_83934

theorem cube_root_of_a_plus_one (a : ℕ) (x : ℝ) (h : x ^ 2 = a) :
  (a + 1 : ℝ) ^ (1/3) = (x ^ 2 + 1) ^ (1/3) :=
by sorry

end cube_root_of_a_plus_one_l839_83934


namespace relationship_abc_l839_83900

theorem relationship_abc : ∀ (a b c : ℕ),
  a = 5^140 ∧ b = 3^210 ∧ c = 2^280 →
  c < a ∧ a < b := by
sorry

end relationship_abc_l839_83900


namespace maze_navigation_ways_l839_83922

/-- Converts a list of digits in base 6 to a number in base 10 -/
def base6ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

/-- The number of ways the dog can navigate through the maze in base 6 -/
def mazeWaysBase6 : List Nat := [4, 1, 2, 5]

/-- Theorem: The number of ways the dog can navigate through the maze
    is 1162 when converted from base 6 to base 10 -/
theorem maze_navigation_ways :
  base6ToBase10 mazeWaysBase6 = 1162 := by
  sorry

end maze_navigation_ways_l839_83922


namespace fifth_day_is_tuesday_l839_83946

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a day in a month -/
structure DayInMonth where
  day : Nat
  dayOfWeek : DayOfWeek

/-- Returns the day of the week for a given number of days after a reference day -/
def dayAfter (startDay : DayOfWeek) (daysAfter : Int) : DayOfWeek :=
  sorry

theorem fifth_day_is_tuesday
  (month : List DayInMonth)
  (h : ∃ d ∈ month, d.day = 20 ∧ d.dayOfWeek = DayOfWeek.Wednesday) :
  ∃ d ∈ month, d.day = 5 ∧ d.dayOfWeek = DayOfWeek.Tuesday :=
sorry

end fifth_day_is_tuesday_l839_83946


namespace cake_after_four_trips_l839_83914

/-- The fraction of cake remaining after a given number of trips to the pantry -/
def cakeRemaining (trips : ℕ) : ℚ :=
  (1 : ℚ) / 2^trips

/-- The theorem stating that after 4 trips, 1/16 of the cake remains -/
theorem cake_after_four_trips :
  cakeRemaining 4 = (1 : ℚ) / 16 := by
  sorry

#eval cakeRemaining 4

end cake_after_four_trips_l839_83914


namespace smallest_multiple_of_5_and_21_l839_83908

theorem smallest_multiple_of_5_and_21 : ∃ b : ℕ+, 
  (∀ k : ℕ+, 5 ∣ k ∧ 21 ∣ k → b ≤ k) ∧ 5 ∣ b ∧ 21 ∣ b ∧ b = 105 := by
  sorry

end smallest_multiple_of_5_and_21_l839_83908


namespace triangle_max_perimeter_l839_83962

theorem triangle_max_perimeter :
  ∀ (x : ℕ),
  x > 0 →
  x < 18 →
  x + 4*x > 18 →
  x + 18 > 4*x →
  4*x + 18 > x →
  (∀ y : ℕ, y > x → y + 4*y ≤ 18 ∨ y + 18 ≤ 4*y ∨ 4*y + 18 ≤ y) →
  x + 4*x + 18 = 38 :=
by
  sorry

#check triangle_max_perimeter

end triangle_max_perimeter_l839_83962


namespace area_ratio_in_special_triangle_l839_83976

-- Define the triangle ABC and point D
variable (A B C D : ℝ × ℝ)

-- Define the properties of the triangle and point D
def is_equilateral (A B C : ℝ × ℝ) : Prop := sorry

def on_side (D A C : ℝ × ℝ) : Prop := sorry

def angle_measure (B D C : ℝ × ℝ) : ℝ := sorry

def triangle_area (A B C : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem area_ratio_in_special_triangle 
  (h_equilateral : is_equilateral A B C)
  (h_on_side : on_side D A C)
  (h_angle : angle_measure B D C = 30) :
  triangle_area A D B / triangle_area C D B = 1 / Real.sqrt 3 := by
  sorry

end area_ratio_in_special_triangle_l839_83976


namespace binomial_divisibility_l839_83960

theorem binomial_divisibility (p : ℕ) (h_prime : Nat.Prime p) (h_odd : p % 2 = 1) :
  p^2 ∣ (Nat.choose (2*p - 1) (p - 1) - 1) := by
  sorry

end binomial_divisibility_l839_83960


namespace reciprocal_roots_condition_l839_83911

/-- The quadratic equation 5x^2 + 7x + k = 0 has reciprocal roots if and only if k = 5 -/
theorem reciprocal_roots_condition (k : ℝ) : 
  (∃ x y : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ 5 * x^2 + 7 * x + k = 0 ∧ 5 * y^2 + 7 * y + k = 0 ∧ x * y = 1) ↔ 
  k = 5 := by
sorry

end reciprocal_roots_condition_l839_83911


namespace train_speed_l839_83974

/-- The speed of a train given its length and time to pass a stationary point -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 240) (h2 : time = 6) :
  length / time = 40 := by
  sorry

end train_speed_l839_83974


namespace square_gt_necessary_not_sufficient_l839_83901

theorem square_gt_necessary_not_sufficient (a : ℝ) :
  (∀ a, a > 1 → a^2 > a) ∧ 
  (∃ a, a^2 > a ∧ ¬(a > 1)) :=
sorry

end square_gt_necessary_not_sufficient_l839_83901


namespace simplify_expression_l839_83929

theorem simplify_expression (x : ℝ) : 2 * (x - 3) - (-x + 4) = 3 * x - 10 := by
  sorry

end simplify_expression_l839_83929


namespace sin_90_degrees_l839_83987

theorem sin_90_degrees : Real.sin (π / 2) = 1 := by
  sorry

end sin_90_degrees_l839_83987


namespace least_positive_angle_phi_l839_83940

theorem least_positive_angle_phi : 
  ∃ φ : Real, φ > 0 ∧ φ ≤ π/2 ∧ 
  Real.cos (15 * π/180) = Real.sin (45 * π/180) + Real.sin φ ∧
  ∀ ψ : Real, ψ > 0 ∧ ψ < φ → 
    Real.cos (15 * π/180) ≠ Real.sin (45 * π/180) + Real.sin ψ ∧
  φ = 15 * π/180 :=
by sorry

end least_positive_angle_phi_l839_83940


namespace calculation_proof_l839_83955

theorem calculation_proof : (2.5 - 0.3) * 0.25 = 0.55 := by
  sorry

end calculation_proof_l839_83955


namespace prove_not_p_or_not_q_l839_83980

theorem prove_not_p_or_not_q (h1 : ¬(p ∧ q)) (h2 : p ∨ q) : ¬p ∨ ¬q := by
  sorry

end prove_not_p_or_not_q_l839_83980


namespace prob_different_suits_enlarged_deck_l839_83992

/-- A deck of cards with five suits -/
structure Deck :=
  (total_cards : ℕ)
  (num_suits : ℕ)
  (cards_per_suit : ℕ)
  (h1 : total_cards = num_suits * cards_per_suit)
  (h2 : num_suits = 5)

/-- The probability of drawing two cards of different suits -/
def prob_different_suits (d : Deck) : ℚ :=
  (d.total_cards - d.cards_per_suit) / (d.total_cards - 1)

/-- The main theorem -/
theorem prob_different_suits_enlarged_deck :
  ∃ d : Deck, d.total_cards = 65 ∧ prob_different_suits d = 13 / 16 := by
  sorry

end prob_different_suits_enlarged_deck_l839_83992


namespace sqrt_y_fourth_power_l839_83920

theorem sqrt_y_fourth_power (y : ℝ) (h : (Real.sqrt y) ^ 4 = 256) : y = 16 := by
  sorry

end sqrt_y_fourth_power_l839_83920


namespace smallest_value_x_squared_plus_8x_l839_83913

theorem smallest_value_x_squared_plus_8x :
  (∀ x : ℝ, x^2 + 8*x ≥ -16) ∧ (∃ x : ℝ, x^2 + 8*x = -16) := by
  sorry

end smallest_value_x_squared_plus_8x_l839_83913


namespace ellipse_intersection_slope_l839_83919

/-- Given an ellipse mx^2 + ny^2 = 1 intersecting with y = 1 - x at A and B,
    if the slope of the line through origin and midpoint of AB is √2, then m/n = √2 -/
theorem ellipse_intersection_slope (m n : ℝ) (A B : ℝ × ℝ) :
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  (m * x₁^2 + n * y₁^2 = 1) →
  (m * x₂^2 + n * y₂^2 = 1) →
  (y₁ = 1 - x₁) →
  (y₂ = 1 - x₂) →
  ((y₁ + y₂) / (x₁ + x₂) = Real.sqrt 2) →
  m / n = Real.sqrt 2 := by
  sorry

end ellipse_intersection_slope_l839_83919


namespace money_division_l839_83933

/-- Given a division of money among three people a, b, and c, where b's share is 65% of a's
    and c's share is 40% of a's, and c's share is 64 rupees, prove that the total sum is 328 rupees. -/
theorem money_division (a b c : ℝ) : 
  (b = 0.65 * a) →  -- b's share is 65% of a's
  (c = 0.40 * a) →  -- c's share is 40% of a's
  (c = 64) →        -- c's share is 64 rupees
  (a + b + c = 328) -- total sum is 328 rupees
:= by sorry

end money_division_l839_83933


namespace intersection_count_l839_83942

/-- The number of distinct intersection points between two algebraic curves -/
def num_intersections (f g : ℝ → ℝ → ℝ) : ℕ :=
  sorry

/-- First curve equation -/
def curve1 (x y : ℝ) : ℝ :=
  (x - y + 3) * (3 * x + y - 7)

/-- Second curve equation -/
def curve2 (x y : ℝ) : ℝ :=
  (x + y - 3) * (2 * x - 5 * y + 12)

theorem intersection_count :
  num_intersections curve1 curve2 = 4 := by sorry

end intersection_count_l839_83942


namespace arctangent_inequalities_l839_83910

theorem arctangent_inequalities (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (Real.arctan x + Real.arctan y < π / 2 ↔ x * y < 1) ∧
  (Real.arctan x + Real.arctan y + Real.arctan z < π ↔ x * y * z < x + y + z) := by
  sorry

end arctangent_inequalities_l839_83910


namespace second_book_cost_l839_83966

/-- Proves that the cost of the second book is $4 given the conditions of Shelby's book fair purchases. -/
theorem second_book_cost (initial_amount : ℕ) (first_book_cost : ℕ) (poster_cost : ℕ) (posters_bought : ℕ) :
  initial_amount = 20 →
  first_book_cost = 8 →
  poster_cost = 4 →
  posters_bought = 2 →
  ∃ (second_book_cost : ℕ),
    second_book_cost + first_book_cost + (poster_cost * posters_bought) = initial_amount ∧
    second_book_cost = 4 :=
by
  sorry

end second_book_cost_l839_83966


namespace triangle_side_c_equals_two_l839_83949

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Sides

-- State the theorem
theorem triangle_side_c_equals_two (ABC : Triangle) 
  (h1 : ABC.B = 2 * ABC.A)  -- B = 2A
  (h2 : ABC.a = 1)          -- a = 1
  (h3 : ABC.b = Real.sqrt 3)  -- b = √3
  : ABC.c = 2 := by
  sorry

end triangle_side_c_equals_two_l839_83949


namespace distance_a_travels_is_60km_l839_83969

/-- Represents the movement of two objects towards each other with doubling speed -/
structure DoubleSpeedMeeting where
  initial_distance : ℝ
  initial_speed_a : ℝ
  initial_speed_b : ℝ

/-- Calculates the distance traveled by object a until meeting object b -/
def distance_traveled_by_a (meeting : DoubleSpeedMeeting) : ℝ :=
  sorry

/-- Theorem stating that given the specific initial conditions, a travels 60 km until meeting b -/
theorem distance_a_travels_is_60km :
  let meeting := DoubleSpeedMeeting.mk 90 10 5
  distance_traveled_by_a meeting = 60 := by
  sorry

end distance_a_travels_is_60km_l839_83969


namespace least_perimeter_l839_83981

/-- Triangle DEF with given cosine values -/
structure TriangleDEF where
  d : ℕ
  e : ℕ
  f : ℕ
  cos_d : Real
  cos_e : Real
  cos_f : Real
  h_cos_d : cos_d = 8 / 17
  h_cos_e : cos_e = 15 / 17
  h_cos_f : cos_f = -5 / 13

/-- The perimeter of triangle DEF -/
def perimeter (t : TriangleDEF) : ℕ := t.d + t.e + t.f

/-- The least possible perimeter of triangle DEF is 503 -/
theorem least_perimeter (t : TriangleDEF) : 
  (∀ t' : TriangleDEF, perimeter t ≤ perimeter t') → perimeter t = 503 := by
  sorry

end least_perimeter_l839_83981


namespace average_rate_of_change_specific_average_rate_of_change_l839_83958

def f (x : ℝ) : ℝ := x^2 + 2*x + 3

theorem average_rate_of_change (a b : ℝ) (h : a < b) :
  (f b - f a) / (b - a) = ((b + a) + 2) :=
sorry

theorem specific_average_rate_of_change :
  (f 3 - f 1) / (3 - 1) = 6 :=
sorry

end average_rate_of_change_specific_average_rate_of_change_l839_83958


namespace stratified_sampling_theorem_l839_83964

/-- Represents the size of each stratum in the population -/
structure StratumSize where
  under30 : ℕ
  between30and40 : ℕ
  over40 : ℕ

/-- Represents the sample size for each stratum -/
structure StratumSample where
  under30 : ℕ
  between30and40 : ℕ
  over40 : ℕ

/-- Calculates the stratified sample size for a given population and total sample size -/
def stratifiedSample (populationSize : ℕ) (sampleSize : ℕ) (strata : StratumSize) : StratumSample :=
  { under30 := sampleSize * strata.under30 / populationSize,
    between30and40 := sampleSize * strata.between30and40 / populationSize,
    over40 := sampleSize * strata.over40 / populationSize }

theorem stratified_sampling_theorem (populationSize : ℕ) (sampleSize : ℕ) (strata : StratumSize) :
  populationSize = 100 →
  sampleSize = 20 →
  strata.under30 = 20 →
  strata.between30and40 = 60 →
  strata.over40 = 20 →
  let sample := stratifiedSample populationSize sampleSize strata
  sample.under30 = 4 ∧ sample.between30and40 = 12 ∧ sample.over40 = 4 :=
by
  sorry

end stratified_sampling_theorem_l839_83964


namespace simplify_expression_l839_83986

theorem simplify_expression (x y : ℝ) : 4 * x^2 + 3 * y^2 - 2 * x^2 - 4 * y^2 = 2 * x^2 - y^2 := by
  sorry

end simplify_expression_l839_83986


namespace james_marbles_l839_83988

theorem james_marbles (total_marbles : ℕ) (num_bags : ℕ) (marbles_per_bag : ℕ) :
  total_marbles = 28 →
  num_bags = 4 →
  marbles_per_bag * num_bags = total_marbles →
  total_marbles - marbles_per_bag = 21 :=
by sorry

end james_marbles_l839_83988


namespace tomato_weight_l839_83995

/-- Calculates the weight of a tomato based on grocery shopping information. -/
theorem tomato_weight (meat_price meat_weight buns_price lettuce_price pickle_price pickle_discount tomato_price_per_pound paid change : ℝ) :
  meat_price = 3.5 →
  meat_weight = 2 →
  buns_price = 1.5 →
  lettuce_price = 1 →
  pickle_price = 2.5 →
  pickle_discount = 1 →
  tomato_price_per_pound = 2 →
  paid = 20 →
  change = 6 →
  (paid - change - (meat_price * meat_weight + buns_price + lettuce_price + (pickle_price - pickle_discount))) / tomato_price_per_pound = 1.5 := by
sorry

end tomato_weight_l839_83995


namespace extremum_point_monotonicity_positive_when_m_leq_2_l839_83996

-- Define the function f(x)
noncomputable def f (x m : ℝ) : ℝ := Real.exp x - Real.log (x + m)

-- Theorem for the extremum point condition
theorem extremum_point (m : ℝ) : 
  (∃ ε > 0, ∀ x ∈ Set.Ioo (-ε) ε, f x m ≥ f 0 m ∨ f x m ≤ f 0 m) → 
  (deriv (f · m)) 0 = 0 := 
sorry

-- Theorem for monotonicity of f(x)
theorem monotonicity (m : ℝ) : 
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → (f x₁ m < f x₂ m ∨ f x₁ m > f x₂ m) := 
sorry

-- Theorem for f(x) > 0 when m ≤ 2
theorem positive_when_m_leq_2 (x m : ℝ) : 
  m ≤ 2 → f x m > 0 := 
sorry

end extremum_point_monotonicity_positive_when_m_leq_2_l839_83996


namespace not_perfect_square_l839_83935

theorem not_perfect_square (n : ℕ) (a : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ 9) : 
  ¬ ∃ k : ℕ, a * 10^(n+1) + 9 = k^2 :=
sorry

end not_perfect_square_l839_83935


namespace larger_number_from_sum_and_difference_l839_83989

theorem larger_number_from_sum_and_difference (x y : ℝ) 
  (sum_eq : x + y = 40)
  (diff_eq : x - y = 6) :
  max x y = 23 := by
  sorry

end larger_number_from_sum_and_difference_l839_83989


namespace perpendicular_lines_imply_a_equals_one_l839_83925

-- Define the lines
def line1 (a x y : ℝ) : Prop := (3*a + 2)*x - 3*y + 8 = 0
def line2 (a x y : ℝ) : Prop := 3*x + (a + 4)*y - 7 = 0

-- Define perpendicularity condition
def perpendicular (a : ℝ) : Prop := 
  (3*a + 2) * 3 + (-3) * (a + 4) = 0

-- Theorem statement
theorem perpendicular_lines_imply_a_equals_one :
  ∀ a : ℝ, perpendicular a → a = 1 := by sorry

end perpendicular_lines_imply_a_equals_one_l839_83925


namespace inequality_proof_l839_83983

theorem inequality_proof (x y z w : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0) 
  (h_eq : (x^3 + y^3)^4 = z^3 + w^3) : 
  x^4*z + y^4*w ≥ z*w := by
  sorry

end inequality_proof_l839_83983


namespace function_extrema_implies_a_range_l839_83905

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + (a + 6)*x + 1

-- State the theorem
theorem function_extrema_implies_a_range (a : ℝ) :
  (∃ (x_max x_min : ℝ), ∀ (x : ℝ), f a x ≤ f a x_max ∧ f a x_min ≤ f a x) →
  (a > 6 ∨ a < -3) :=
sorry

end function_extrema_implies_a_range_l839_83905


namespace original_price_calculation_l839_83945

theorem original_price_calculation (selling_price : ℝ) (profit_percentage : ℝ) 
  (h1 : selling_price = 600) 
  (h2 : profit_percentage = 20) : 
  ∃ original_price : ℝ, 
    selling_price = original_price * (1 + profit_percentage / 100) ∧ 
    original_price = 500 := by
  sorry

end original_price_calculation_l839_83945


namespace inverse_f_zero_l839_83951

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := 1 / (2 * a * x + 3 * b)

theorem inverse_f_zero (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ∃ x, f a b x = 1 / (3 * b) ∧ (∀ y, f a b y = x → y = 0) :=
by sorry

end inverse_f_zero_l839_83951


namespace toy_store_shelves_l839_83932

def number_of_shelves (initial_stock new_shipment bears_per_shelf : ℕ) : ℕ :=
  (initial_stock + new_shipment) / bears_per_shelf

theorem toy_store_shelves : 
  number_of_shelves 4 10 7 = 2 := by
  sorry

end toy_store_shelves_l839_83932


namespace monthly_compounding_greater_than_annual_l839_83972

theorem monthly_compounding_greater_than_annual : 
  (1 + 0.04 / 12) ^ 12 > 1 + 0.04 := by
  sorry

end monthly_compounding_greater_than_annual_l839_83972


namespace tree_space_for_given_conditions_l839_83918

/-- Calculates the sidewalk space taken by each tree given the street length, number of trees, and space between trees. -/
def tree_space (street_length : ℕ) (num_trees : ℕ) (space_between : ℕ) : ℚ :=
  let total_gap_space := (num_trees - 1) * space_between
  let total_tree_space := street_length - total_gap_space
  (total_tree_space : ℚ) / num_trees

/-- Theorem stating that for a 151-foot street with 16 trees and 9 feet between each tree, each tree takes up 1 square foot of sidewalk space. -/
theorem tree_space_for_given_conditions :
  tree_space 151 16 9 = 1 := by
  sorry

end tree_space_for_given_conditions_l839_83918


namespace expression_evaluation_l839_83909

theorem expression_evaluation : 2 * (3 * 4 * 5) * (1/3 + 1/4 + 1/5) = 94 := by
  sorry

end expression_evaluation_l839_83909


namespace tina_customers_l839_83979

/-- Calculates the number of customers Tina sold books to -/
def number_of_customers (selling_price cost_price total_profit books_per_customer : ℚ) : ℚ :=
  (total_profit / (selling_price - cost_price)) / books_per_customer

/-- Theorem: Given the conditions, Tina sold books to 4 customers -/
theorem tina_customers :
  let selling_price : ℚ := 20
  let cost_price : ℚ := 5
  let total_profit : ℚ := 120
  let books_per_customer : ℚ := 2
  number_of_customers selling_price cost_price total_profit books_per_customer = 4 := by
  sorry

#eval number_of_customers 20 5 120 2

end tina_customers_l839_83979


namespace solve_ttakji_problem_l839_83906

def ttakji_problem (initial_large : ℕ) (initial_small : ℕ) (final_total : ℕ) : Prop :=
  ∃ (lost_large : ℕ),
    initial_large ≥ lost_large ∧
    initial_small ≥ 3 * lost_large ∧
    initial_large + initial_small - lost_large - 3 * lost_large = final_total ∧
    lost_large = 4

theorem solve_ttakji_problem :
  ttakji_problem 12 34 30 := by sorry

end solve_ttakji_problem_l839_83906


namespace exists_player_in_win_range_l839_83936

/-- Represents a chess tournament with 2n+1 players -/
structure ChessTournament (n : ℕ) where
  /-- The number of games won by lower-rated players -/
  k : ℕ
  /-- Player ratings, assumed to be unique -/
  ratings : Fin (2*n+1) → ℕ
  ratings_unique : ∀ i j, i ≠ j → ratings i ≠ ratings j

/-- The number of wins for each player -/
def wins (t : ChessTournament n) : Fin (2*n+1) → ℕ :=
  sorry

theorem exists_player_in_win_range (n : ℕ) (t : ChessTournament n) :
  ∃ p : Fin (2*n+1), 
    (n : ℝ) - Real.sqrt (2 * t.k) ≤ wins t p ∧ 
    wins t p ≤ (n : ℝ) + Real.sqrt (2 * t.k) :=
  sorry

end exists_player_in_win_range_l839_83936


namespace power_calculation_l839_83944

theorem power_calculation : 10^6 * (10^2)^3 / 10^4 = 10^8 := by
  sorry

end power_calculation_l839_83944


namespace seth_candy_bars_l839_83939

theorem seth_candy_bars (max_candy_bars : ℕ) (seth_candy_bars : ℕ) : 
  max_candy_bars = 24 →
  seth_candy_bars = 3 * max_candy_bars + 6 →
  seth_candy_bars = 78 :=
by sorry

end seth_candy_bars_l839_83939


namespace worker_a_alone_time_l839_83921

/-- Represents the efficiency of a worker -/
structure WorkerEfficiency where
  rate : ℝ
  rate_pos : rate > 0

/-- Represents a job to be completed -/
structure Job where
  total_work : ℝ
  total_work_pos : total_work > 0

theorem worker_a_alone_time 
  (job : Job) 
  (a b : WorkerEfficiency) 
  (h1 : a.rate = 2 * b.rate) 
  (h2 : job.total_work / (a.rate + b.rate) = 20) : 
  job.total_work / a.rate = 30 := by
  sorry

end worker_a_alone_time_l839_83921


namespace x_minus_y_equals_one_l839_83907

-- Define x and y based on the given conditions
def x : Int := 2 - 4 + 6
def y : Int := 1 - 3 + 5

-- State the theorem to be proved
theorem x_minus_y_equals_one : x - y = 1 := by
  sorry

end x_minus_y_equals_one_l839_83907


namespace sum_ab_over_2b_plus_1_geq_1_l839_83967

variables (a b c : ℝ)

theorem sum_ab_over_2b_plus_1_geq_1
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_sum : a + b + c = 3) :
  (a * b) / (2 * b + 1) + (b * c) / (2 * c + 1) + (c * a) / (2 * a + 1) ≥ 1 :=
sorry

end sum_ab_over_2b_plus_1_geq_1_l839_83967


namespace rhombus_area_from_quadratic_roots_l839_83937

theorem rhombus_area_from_quadratic_roots : ∀ (d₁ d₂ : ℝ),
  d₁^2 - 10*d₁ + 24 = 0 →
  d₂^2 - 10*d₂ + 24 = 0 →
  d₁ ≠ d₂ →
  (1/2) * d₁ * d₂ = 12 := by
sorry

end rhombus_area_from_quadratic_roots_l839_83937


namespace female_workers_count_l839_83957

/-- Represents the number of workers of each type and their wages --/
structure WorkforceData where
  male_workers : ℕ
  female_workers : ℕ
  child_workers : ℕ
  male_wage : ℕ
  female_wage : ℕ
  child_wage : ℕ
  average_wage : ℕ

/-- Calculates the total daily wage for all workers --/
def total_daily_wage (data : WorkforceData) : ℕ :=
  data.male_workers * data.male_wage +
  data.female_workers * data.female_wage +
  data.child_workers * data.child_wage

/-- Calculates the total number of workers --/
def total_workers (data : WorkforceData) : ℕ :=
  data.male_workers + data.female_workers + data.child_workers

/-- Theorem stating that the number of female workers is 15 --/
theorem female_workers_count (data : WorkforceData)
  (h1 : data.male_workers = 20)
  (h2 : data.child_workers = 5)
  (h3 : data.male_wage = 25)
  (h4 : data.female_wage = 20)
  (h5 : data.child_wage = 8)
  (h6 : data.average_wage = 21)
  (h7 : (total_daily_wage data) / (total_workers data) = data.average_wage) :
  data.female_workers = 15 :=
sorry

end female_workers_count_l839_83957


namespace inequalities_theorem_l839_83924

theorem inequalities_theorem (a b : ℝ) (h : (1 / a) < (1 / b) ∧ (1 / b) < 0) :
  (abs a < abs b) ∧ (a > b) ∧ (a + b > a * b) ∧ (a^3 > b^3) := by sorry

end inequalities_theorem_l839_83924


namespace complex_number_in_third_quadrant_l839_83952

theorem complex_number_in_third_quadrant : 
  let z : ℂ := (1 - Complex.I) / Complex.I
  (z.re < 0) ∧ (z.im < 0) :=
by sorry

end complex_number_in_third_quadrant_l839_83952


namespace solve_complex_equation_l839_83941

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation (1+i)Z = 2
def equation (Z : ℂ) : Prop := (1 + i) * Z = 2

-- Theorem statement
theorem solve_complex_equation :
  ∀ Z : ℂ, equation Z → Z = 1 - i :=
by sorry

end solve_complex_equation_l839_83941


namespace compare_cubic_and_mixed_terms_l839_83953

theorem compare_cubic_and_mixed_terms {a b : ℝ} (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  a^3 + b^3 > a^2 * b + a * b^2 := by
  sorry

end compare_cubic_and_mixed_terms_l839_83953


namespace triangle_abc_properties_l839_83923

theorem triangle_abc_properties (A B C : ℝ) (AB AC BC : ℝ) 
  (h_triangle : A + B + C = π)
  (h_AB : AB = 2)
  (h_AC : AC = 3)
  (h_BC : BC = Real.sqrt 7) : 
  A = π / 3 ∧ Real.cos (B - C) = 11 / 14 := by
  sorry

end triangle_abc_properties_l839_83923


namespace max_different_ages_l839_83963

theorem max_different_ages 
  (average_age : ℝ) 
  (std_dev : ℝ) 
  (average_age_eq : average_age = 31) 
  (std_dev_eq : std_dev = 8) : 
  ∃ (max_ages : ℕ), 
    max_ages = 17 ∧ 
    ∀ (age : ℕ), 
      (↑age ≥ average_age - std_dev ∧ ↑age ≤ average_age + std_dev) ↔ 
      (age ≥ 23 ∧ age ≤ 39) :=
by sorry

end max_different_ages_l839_83963


namespace more_girls_than_boys_l839_83959

/-- Represents the number of students in Mrs. Smith's chemistry class -/
def total_students : ℕ := 42

/-- Represents the ratio of boys in the class -/
def boys_ratio : ℕ := 3

/-- Represents the ratio of girls in the class -/
def girls_ratio : ℕ := 4

/-- Calculates the number of boys in the class -/
def num_boys : ℕ := (total_students * boys_ratio) / (boys_ratio + girls_ratio)

/-- Calculates the number of girls in the class -/
def num_girls : ℕ := (total_students * girls_ratio) / (boys_ratio + girls_ratio)

/-- Proves that there are 6 more girls than boys in the class -/
theorem more_girls_than_boys : num_girls - num_boys = 6 := by
  sorry

end more_girls_than_boys_l839_83959


namespace weight_loss_challenge_l839_83968

theorem weight_loss_challenge (initial_weight : ℝ) (h_initial_weight_pos : initial_weight > 0) :
  let weight_after_loss := initial_weight * (1 - 0.11)
  let measured_weight_loss_percentage := 0.0922
  ∃ (clothes_weight_percentage : ℝ),
    weight_after_loss * (1 + clothes_weight_percentage) = initial_weight * (1 - measured_weight_loss_percentage) ∧
    clothes_weight_percentage = 0.02 :=
by sorry

end weight_loss_challenge_l839_83968


namespace tan_negative_five_pi_thirds_equals_sqrt_three_l839_83917

theorem tan_negative_five_pi_thirds_equals_sqrt_three :
  Real.tan (-5 * π / 3) = Real.sqrt 3 := by
  sorry

end tan_negative_five_pi_thirds_equals_sqrt_three_l839_83917


namespace knights_round_table_l839_83994

theorem knights_round_table (n : ℕ) (h : n = 25) :
  let total_arrangements := n * (n + 1) * (n + 2) / 6
  let non_adjacent_arrangements := n * (n - 3) * (n - 4) / 2
  (total_arrangements - non_adjacent_arrangements : ℚ) / total_arrangements = 11 / 46 :=
by sorry

end knights_round_table_l839_83994


namespace inequalities_theorem_l839_83902

theorem inequalities_theorem :
  (∀ a b c d : ℝ, a > b → c > d → a + c > b + d) ∧
  (∃ a b c d : ℝ, a > b ∧ c > d ∧ a * c ≤ b * d) ∧
  (∃ a b c : ℝ, a < b ∧ a * c^2 ≥ b * c^2) ∧
  (∀ a b c : ℝ, a > b → b > 0 → c < 0 → c / a > c / b) :=
by sorry

end inequalities_theorem_l839_83902


namespace circle_line_intersection_l839_83954

/-- The necessary and sufficient condition for a circle and a line to have common points -/
theorem circle_line_intersection (k : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 = 1 ∧ y = k*x - 3) ↔ -Real.sqrt 8 ≤ k ∧ k ≤ Real.sqrt 8 := by
  sorry

end circle_line_intersection_l839_83954


namespace smallest_y_squared_value_l839_83961

/-- Represents an isosceles trapezoid EFGH with a tangent circle -/
structure IsoscelesTrapezoidWithTangentCircle where
  EF : ℝ
  GH : ℝ
  y : ℝ
  is_isosceles : EF > GH
  tangent_circle : Bool

/-- The smallest possible value of y^2 in the given configuration -/
def smallest_y_squared (t : IsoscelesTrapezoidWithTangentCircle) : ℝ := sorry

/-- Theorem stating the smallest possible value of y^2 -/
theorem smallest_y_squared_value 
  (t : IsoscelesTrapezoidWithTangentCircle) 
  (h1 : t.EF = 102) 
  (h2 : t.GH = 26) 
  (h3 : t.tangent_circle = true) : 
  smallest_y_squared t = 1938 := by sorry

end smallest_y_squared_value_l839_83961


namespace crackers_per_friend_l839_83965

theorem crackers_per_friend (initial_crackers : ℕ) (friends : ℕ) (remaining_crackers : ℕ) :
  initial_crackers = 23 →
  friends = 2 →
  remaining_crackers = 11 →
  (initial_crackers - remaining_crackers) / friends = 6 :=
by sorry

end crackers_per_friend_l839_83965


namespace power_division_equality_l839_83931

theorem power_division_equality : (3^3)^2 / 3^2 = 81 := by sorry

end power_division_equality_l839_83931


namespace percentage_of_A_students_l839_83997

theorem percentage_of_A_students (total_students : ℕ) (failed_students : ℕ) 
  (h1 : total_students = 32)
  (h2 : failed_students = 18)
  (h3 : ∃ (A : ℕ) (B_C : ℕ), 
    A + B_C + failed_students = total_students ∧ 
    B_C = (total_students - failed_students - A) / 4) :
  (((total_students - failed_students) : ℚ) / total_students) * 100 = 43.75 := by
  sorry

end percentage_of_A_students_l839_83997


namespace decimal_to_scientific_notation_l839_83916

/-- Expresses a given decimal number in scientific notation -/
def scientific_notation (x : ℝ) : ℝ × ℤ :=
  sorry

theorem decimal_to_scientific_notation :
  scientific_notation 0.00000011 = (1.1, -7) :=
sorry

end decimal_to_scientific_notation_l839_83916


namespace sugar_salt_difference_l839_83904

/-- 
Given a recipe that calls for specific amounts of ingredients and Mary's actions,
prove that the difference between the required cups of sugar and salt is 2.
-/
theorem sugar_salt_difference (sugar_required flour_required salt_required flour_added : ℕ) 
  (h1 : sugar_required = 11)
  (h2 : flour_required = 6)
  (h3 : salt_required = 9)
  (h4 : flour_added = 12) :
  sugar_required - salt_required = 2 := by
  sorry

end sugar_salt_difference_l839_83904


namespace six_at_three_equals_six_l839_83928

/-- The @ operation for positive integers a and b where a > b -/
def at_op (a b : ℕ+) (h : a > b) : ℚ :=
  (a * b : ℚ) / (a - b)

/-- Theorem: 6 @ 3 = 6 -/
theorem six_at_three_equals_six :
  ∀ (h : (6 : ℕ+) > (3 : ℕ+)), at_op 6 3 h = 6 := by sorry

end six_at_three_equals_six_l839_83928


namespace circle_properties_l839_83938

/-- Circle C in the Cartesian coordinate system -/
def circle_C (x y b : ℝ) : Prop := x^2 + y^2 - 6*x - 4*y + b = 0

/-- Point A -/
def point_A : ℝ × ℝ := (0, 3)

/-- Radius of circle C -/
def radius : ℝ := 1

theorem circle_properties :
  ∃ (b : ℝ), 
    (∀ x y, circle_C x y b → (x - 3)^2 + (y - 2)^2 = 1) ∧ 
    (b < 13) ∧
    (∃ (k : ℝ), k = -3/4 ∧ ∀ x y, 3*x + 4*y - 12 = 0 → circle_C x y b) ∧
    (∀ y, circle_C 0 3 b → y = 3) :=
by sorry

end circle_properties_l839_83938


namespace like_terms_imply_m_minus_2n_equals_1_l839_83982

/-- Two monomials are like terms if they have the same variables with the same exponents. -/
def are_like_terms (m n : ℕ) : Prop :=
  m = 3 ∧ n = 1

/-- The theorem states that if 3x^m*y and -5x^3*y^n are like terms, then m - 2n = 1. -/
theorem like_terms_imply_m_minus_2n_equals_1 (m n : ℕ) :
  are_like_terms m n → m - 2*n = 1 := by
  sorry

end like_terms_imply_m_minus_2n_equals_1_l839_83982


namespace sum_of_coefficients_l839_83947

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (1 + 2*x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₀ + a₂ + a₄ = 121 := by
sorry

end sum_of_coefficients_l839_83947


namespace parabola_shift_l839_83973

/-- Given a parabola y = x^2 + 2, shifting it 3 units left and 4 units down results in y = (x + 3)^2 - 2 -/
theorem parabola_shift (x y : ℝ) : 
  (y = x^2 + 2) → 
  (y = (x + 3)^2 - 2) ↔ 
  (y + 4 = ((x + 3) + 3)^2 + 2) :=
by sorry

end parabola_shift_l839_83973


namespace club_female_count_l839_83956

theorem club_female_count (total : ℕ) (difference : ℕ) (female : ℕ) : 
  total = 82 →
  difference = 6 →
  female = total / 2 + difference / 2 →
  female = 44 := by
sorry

end club_female_count_l839_83956


namespace student_distribution_proof_l839_83971

def distribute_students (n : ℕ) (k : ℕ) : ℕ := sorry

theorem student_distribution_proof : 
  distribute_students 24 3 = 475 := by sorry

end student_distribution_proof_l839_83971


namespace K_set_equals_target_set_l839_83993

/-- The set of natural numbers K satisfying the given conditions for a fixed h = 2^r -/
def K_set (r : ℕ) : Set ℕ :=
  {K : ℕ | ∃ (m n : ℕ), m > 1 ∧ Odd m ∧
    K ∣ (m^(2^r) - 1) ∧
    K ∣ (n^((m^(2^r) - 1) / K) + 1)}

/-- The set of numbers of the form 2^(r+s) * t where t is odd -/
def target_set (r : ℕ) : Set ℕ :=
  {K : ℕ | ∃ (s t : ℕ), K = 2^(r+s) * t ∧ Odd t}

/-- The main theorem stating that K_set equals target_set for any non-negative integer r -/
theorem K_set_equals_target_set (r : ℕ) : K_set r = target_set r := by
  sorry

end K_set_equals_target_set_l839_83993


namespace program_result_l839_83970

/-- The smallest positive integer n for which n² + 4n ≥ 10000 -/
def smallest_n : ℕ := 99

/-- The function that computes x given n -/
def x (n : ℕ) : ℕ := 3 + 2 * n

/-- The function that computes S given n -/
def S (n : ℕ) : ℕ := n^2 + 4*n

theorem program_result :
  (∀ m : ℕ, m < smallest_n → S m < 10000) ∧
  S smallest_n ≥ 10000 ∧
  x smallest_n = 201 := by sorry

end program_result_l839_83970


namespace complex_ratio_theorem_l839_83984

theorem complex_ratio_theorem (x y : ℂ) 
  (h1 : (x^2 + y^2) / (x + y) = 4)
  (h2 : (x^4 + y^4) / (x^3 + y^3) = 2) :
  (x^6 + y^6) / (x^5 + y^5) = 10 + 2 * Real.sqrt 17 ∨
  (x^6 + y^6) / (x^5 + y^5) = 10 - 2 * Real.sqrt 17 :=
by sorry

end complex_ratio_theorem_l839_83984


namespace alice_fruit_consumption_impossible_l839_83930

/-- Represents the number of each type of fruit in the basket -/
structure FruitBasket :=
  (apples : ℕ)
  (pears : ℕ)
  (oranges : ℕ)

/-- Represents Alice's fruit consumption for a day -/
inductive DailyConsumption
  | AP  -- Apple and Pear
  | AO  -- Apple and Orange
  | PO  -- Pear and Orange

def initial_basket : FruitBasket :=
  { apples := 5, pears := 8, oranges := 11 }

def consume_fruits (basket : FruitBasket) (consumption : DailyConsumption) : FruitBasket :=
  match consumption with
  | DailyConsumption.AP => { apples := basket.apples - 1, pears := basket.pears - 1, oranges := basket.oranges }
  | DailyConsumption.AO => { apples := basket.apples - 1, pears := basket.pears, oranges := basket.oranges - 1 }
  | DailyConsumption.PO => { apples := basket.apples, pears := basket.pears - 1, oranges := basket.oranges - 1 }

def fruits_equal (basket : FruitBasket) : Prop :=
  basket.apples = basket.pears ∧ basket.pears = basket.oranges

theorem alice_fruit_consumption_impossible :
  ∀ (days : ℕ) (consumptions : List DailyConsumption),
    days = consumptions.length →
    ¬(fruits_equal (consumptions.foldl consume_fruits initial_basket)) :=
  sorry


end alice_fruit_consumption_impossible_l839_83930


namespace prob_black_second_draw_l839_83943

/-- Represents the color of a ball -/
inductive Color
| Red
| Black

/-- Represents the state of the box -/
structure Box :=
  (red : ℕ)
  (black : ℕ)

/-- Calculates the probability of drawing a black ball -/
def prob_black (b : Box) : ℚ :=
  b.black / (b.red + b.black)

/-- Adds balls to the box based on the color drawn -/
def add_balls (b : Box) (c : Color) : Box :=
  match c with
  | Color.Red => Box.mk (b.red + 3) b.black
  | Color.Black => Box.mk b.red (b.black + 3)

/-- The main theorem to prove -/
theorem prob_black_second_draw (initial_box : Box) 
  (h1 : initial_box.red = 4)
  (h2 : initial_box.black = 5) : 
  (prob_black initial_box * prob_black (add_balls initial_box Color.Black) +
   (1 - prob_black initial_box) * prob_black (add_balls initial_box Color.Red)) = 5/9 :=
by sorry

end prob_black_second_draw_l839_83943


namespace sum_in_D_l839_83948

-- Define the sets A, B, C, and D
def A : Set Int := {x | ∃ k : Int, x = 4 * k}
def B : Set Int := {x | ∃ m : Int, x = 4 * m + 1}
def C : Set Int := {x | ∃ n : Int, x = 4 * n + 2}
def D : Set Int := {x | ∃ t : Int, x = 4 * t + 3}

-- State the theorem
theorem sum_in_D (a b : Int) (ha : a ∈ B) (hb : b ∈ C) : a + b ∈ D := by
  sorry

end sum_in_D_l839_83948


namespace solve_linear_equation_l839_83977

theorem solve_linear_equation (x : ℝ) : 2*x - 3*x + 4*x = 150 → x = 50 := by
  sorry

end solve_linear_equation_l839_83977


namespace room_length_proof_l839_83975

theorem room_length_proof (width : ℝ) (total_cost : ℝ) (paving_rate : ℝ) :
  width = 4.75 →
  total_cost = 29925 →
  paving_rate = 900 →
  (total_cost / paving_rate) / width = 7 := by
  sorry

end room_length_proof_l839_83975


namespace decimal_places_theorem_l839_83998

def first_1000_decimal_places (x : ℝ) : List ℕ :=
  sorry

theorem decimal_places_theorem :
  (∀ d ∈ first_1000_decimal_places ((6 + Real.sqrt 35) ^ 1999), d = 9) ∧
  (∀ d ∈ first_1000_decimal_places ((6 + Real.sqrt 37) ^ 1999), d = 0) ∧
  (∀ d ∈ first_1000_decimal_places ((6 + Real.sqrt 37) ^ 2000), d = 9) :=
by sorry

end decimal_places_theorem_l839_83998


namespace negative_abs_equals_opposite_l839_83991

theorem negative_abs_equals_opposite (x : ℝ) : x < 0 → |x| = -x := by
  sorry

end negative_abs_equals_opposite_l839_83991


namespace min_area_triangle_abc_l839_83985

/-- The minimum area of a triangle ABC where A = (0, 0), B = (30, 16), and C has integer coordinates --/
theorem min_area_triangle_abc : 
  ∀ (p q : ℤ), 
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (30, 16)
  let C : ℝ × ℝ := (p, q)
  let area := (1/2 : ℝ) * |16 * p - 30 * q|
  1 ≤ area ∧ (∃ (p' q' : ℤ), (1/2 : ℝ) * |16 * p' - 30 * q'| = 1) :=
by sorry

end min_area_triangle_abc_l839_83985


namespace expression_value_l839_83915

theorem expression_value (m n : ℝ) 
  (h1 : m^2 + 2*m*n = 384) 
  (h2 : 3*m*n + 2*n^2 = 560) : 
  2*m^2 + 13*m*n + 6*n^2 - 444 = 2004 := by
  sorry

end expression_value_l839_83915


namespace handshake_arrangement_theorem_l839_83912

-- Define the number of people in the group
def num_people : ℕ := 12

-- Define the number of handshakes per person
def handshakes_per_person : ℕ := 3

-- Define the function to calculate the number of distinct handshaking arrangements
def num_arrangements (n : ℕ) (k : ℕ) : ℕ := sorry

-- Define the function to calculate the remainder when divided by 1000
def remainder_mod_1000 (x : ℕ) : ℕ := x % 1000

-- Theorem statement
theorem handshake_arrangement_theorem :
  num_arrangements num_people handshakes_per_person = 680680 ∧
  remainder_mod_1000 (num_arrangements num_people handshakes_per_person) = 680 := by
  sorry

end handshake_arrangement_theorem_l839_83912


namespace remainder_3056_div_78_l839_83950

theorem remainder_3056_div_78 : 3056 % 78 = 14 := by
  sorry

end remainder_3056_div_78_l839_83950


namespace midpoint_quadrilateral_area_l839_83999

/-- A parallelogram in a 2D plane -/
structure Parallelogram where
  area : ℝ

/-- A quadrilateral formed by joining the midpoints of a parallelogram's sides -/
def midpoint_quadrilateral (p : Parallelogram) : Parallelogram :=
  { area := sorry }

/-- The area of the midpoint quadrilateral is 1/4 of the original parallelogram's area -/
theorem midpoint_quadrilateral_area (p : Parallelogram) :
  (midpoint_quadrilateral p).area = p.area / 4 := by
  sorry

end midpoint_quadrilateral_area_l839_83999


namespace mabel_tomatoes_l839_83903

/-- The number of tomato plants Mabel planted -/
def num_plants : ℕ := 4

/-- The number of tomatoes on the first plant -/
def first_plant_tomatoes : ℕ := 8

/-- The number of additional tomatoes on the second plant compared to the first -/
def second_plant_additional : ℕ := 4

/-- The factor by which the remaining plants' tomatoes exceed the sum of the first two plants -/
def remaining_plants_factor : ℕ := 3

/-- The total number of tomatoes Mabel has -/
def total_tomatoes : ℕ := 140

theorem mabel_tomatoes :
  let second_plant_tomatoes := first_plant_tomatoes + second_plant_additional
  let first_two_plants := first_plant_tomatoes + second_plant_tomatoes
  let remaining_plants_tomatoes := 2 * (remaining_plants_factor * first_two_plants)
  first_plant_tomatoes + second_plant_tomatoes + remaining_plants_tomatoes = total_tomatoes :=
by sorry

end mabel_tomatoes_l839_83903


namespace average_annual_decrease_rate_optimal_price_reduction_l839_83926

-- Part 1: Average annual percentage decrease
def initial_price : ℝ := 200
def final_price : ℝ := 162
def num_years : ℕ := 2

-- Part 2: Unit price reduction
def selling_price : ℝ := 200
def initial_daily_sales : ℕ := 20
def price_decrease_step : ℝ := 3
def sales_increase_step : ℕ := 6
def target_daily_profit : ℝ := 1150

-- Theorem for Part 1
theorem average_annual_decrease_rate (x : ℝ) :
  initial_price * (1 - x)^num_years = final_price →
  x = 0.1 := by sorry

-- Theorem for Part 2
theorem optimal_price_reduction (m : ℝ) :
  (selling_price - m - 162) * (initial_daily_sales + 2 * m) = target_daily_profit →
  m = 15 := by sorry

end average_annual_decrease_rate_optimal_price_reduction_l839_83926
