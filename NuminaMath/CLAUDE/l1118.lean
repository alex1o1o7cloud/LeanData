import Mathlib

namespace chocolate_cost_l1118_111873

theorem chocolate_cost (candy_cost : ℕ) (candy_count : ℕ) (chocolate_count : ℕ) (price_difference : ℕ) :
  candy_cost = 530 →
  candy_count = 12 →
  chocolate_count = 8 →
  price_difference = 5400 →
  candy_count * candy_cost = chocolate_count * (candy_count * candy_cost / chocolate_count - price_difference / chocolate_count) + price_difference →
  candy_count * candy_cost / chocolate_count - price_difference / chocolate_count = 120 :=
by
  sorry

end chocolate_cost_l1118_111873


namespace weight_loss_difference_l1118_111810

/-- Given Barbi's and Luca's weight loss rates and durations, prove the difference in their total weight losses -/
theorem weight_loss_difference (barbi_monthly_loss : ℝ) (barbi_months : ℕ) 
  (luca_yearly_loss : ℝ) (luca_years : ℕ) : 
  barbi_monthly_loss = 1.5 → 
  barbi_months = 12 → 
  luca_yearly_loss = 9 → 
  luca_years = 11 → 
  luca_yearly_loss * luca_years - barbi_monthly_loss * barbi_months = 81 := by
  sorry

end weight_loss_difference_l1118_111810


namespace tim_running_hours_l1118_111846

/-- The number of days Tim originally ran per week -/
def original_days : ℕ := 3

/-- The number of extra days Tim added to her running schedule -/
def extra_days : ℕ := 2

/-- The number of hours Tim runs in the morning each day she runs -/
def morning_hours : ℕ := 1

/-- The number of hours Tim runs in the evening each day she runs -/
def evening_hours : ℕ := 1

/-- Theorem stating that Tim now runs 10 hours a week -/
theorem tim_running_hours : 
  (original_days + extra_days) * (morning_hours + evening_hours) = 10 := by
  sorry

end tim_running_hours_l1118_111846


namespace unique_solution_value_l1118_111828

theorem unique_solution_value (p : ℝ) : 
  (∃! x : ℝ, x ≠ 0 ∧ (1 : ℝ) / (3 * x) = (p - x) / 4) ↔ p = 4 / 3 :=
by sorry

end unique_solution_value_l1118_111828


namespace total_spent_candy_and_chocolate_l1118_111892

/-- The total amount spent on a candy bar and chocolate -/
def total_spent (candy_bar_cost chocolate_cost : ℕ) : ℕ :=
  candy_bar_cost + chocolate_cost

/-- Theorem: The total amount spent on a candy bar costing $7 and chocolate costing $6 is $13 -/
theorem total_spent_candy_and_chocolate :
  total_spent 7 6 = 13 := by
  sorry

end total_spent_candy_and_chocolate_l1118_111892


namespace fraction_meaningful_l1118_111808

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x - 4)) ↔ x ≠ 4 := by
sorry

end fraction_meaningful_l1118_111808


namespace passion_fruit_crates_l1118_111822

theorem passion_fruit_crates (total_crates grapes_crates mangoes_crates : ℕ) 
  (h1 : total_crates = 50)
  (h2 : grapes_crates = 13)
  (h3 : mangoes_crates = 20) :
  total_crates - (grapes_crates + mangoes_crates) = 17 := by
  sorry

end passion_fruit_crates_l1118_111822


namespace amit_work_days_l1118_111858

theorem amit_work_days (amit_rate : ℚ) (ananthu_rate : ℚ) : 
  ananthu_rate = 1 / 45 →
  amit_rate * 3 + ananthu_rate * 36 = 1 →
  amit_rate = 1 / 15 :=
by
  sorry

end amit_work_days_l1118_111858


namespace calculation_proof_l1118_111899

theorem calculation_proof : 15 * 30 + 45 * 15 - 15 * 10 = 975 := by
  sorry

end calculation_proof_l1118_111899


namespace quadratic_roots_condition_l1118_111878

theorem quadratic_roots_condition (p q : ℝ) : 
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    x₁^2 + p*x₁ + q = 0 ∧ 
    x₂^2 + p*x₂ + q = 0 ∧
    x₁ = 2*p ∧ 
    x₂ = p + q) →
  p = 2/3 ∧ q = -8/3 := by
sorry

end quadratic_roots_condition_l1118_111878


namespace macaroon_problem_l1118_111845

theorem macaroon_problem (total_macaroons : ℕ) (weight_per_macaroon : ℕ) (total_bags : ℕ) (remaining_weight : ℕ) :
  total_macaroons = 12 →
  weight_per_macaroon = 5 →
  total_bags = 4 →
  remaining_weight = 45 →
  total_macaroons % total_bags = 0 →
  (total_macaroons * weight_per_macaroon - remaining_weight) / (total_macaroons / total_bags * weight_per_macaroon) = 1 :=
by
  sorry

end macaroon_problem_l1118_111845


namespace forty_coins_impossible_l1118_111834

/-- Represents the contents of Bethany's purse -/
structure Purse where
  pound_coins : ℕ
  twenty_pence : ℕ
  fifty_pence : ℕ

/-- Calculates the total value of coins in pence -/
def total_value (p : Purse) : ℕ :=
  100 * p.pound_coins + 20 * p.twenty_pence + 50 * p.fifty_pence

/-- Calculates the total number of coins -/
def total_coins (p : Purse) : ℕ :=
  p.pound_coins + p.twenty_pence + p.fifty_pence

/-- Represents Bethany's purse with the given conditions -/
def bethany_purse : Purse :=
  { pound_coins := 11
  , twenty_pence := 0  -- placeholder, actual value unknown
  , fifty_pence := 0 } -- placeholder, actual value unknown

/-- The mean value of coins in pence -/
def mean_value : ℚ := 52

theorem forty_coins_impossible :
  ∀ p : Purse,
    p.pound_coins = 11 →
    (total_value p : ℚ) / (total_coins p : ℚ) = mean_value →
    total_coins p ≠ 40 :=
by sorry

end forty_coins_impossible_l1118_111834


namespace multiply_specific_numbers_l1118_111890

theorem multiply_specific_numbers : 469160 * 999999 = 469159530840 := by
  sorry

end multiply_specific_numbers_l1118_111890


namespace maggie_tractor_hours_l1118_111838

/-- Represents Maggie's work schedule and income for a week. -/
structure WorkWeek where
  tractorHours : ℕ
  officeHours : ℕ
  deliveryHours : ℕ
  totalIncome : ℕ

/-- Checks if a work week satisfies the given conditions. -/
def isValidWorkWeek (w : WorkWeek) : Prop :=
  w.officeHours = 2 * w.tractorHours ∧
  w.deliveryHours = w.officeHours - 3 ∧
  w.totalIncome = 10 * w.officeHours + 12 * w.tractorHours + 15 * w.deliveryHours

/-- Theorem stating that given the conditions, Maggie spent 15 hours driving the tractor. -/
theorem maggie_tractor_hours :
  ∃ (w : WorkWeek), isValidWorkWeek w ∧ w.totalIncome = 820 → w.tractorHours = 15 :=
by sorry


end maggie_tractor_hours_l1118_111838


namespace midway_point_distance_yendor_midway_distance_l1118_111817

/-- Represents an elliptical orbit -/
structure EllipticalOrbit where
  /-- Length of the major axis -/
  major_axis : ℝ
  /-- Distance between the foci -/
  focal_distance : ℝ
  /-- Assumption that the focal distance is less than the major axis -/
  h_focal_lt_major : focal_distance < major_axis

/-- A point on the elliptical orbit -/
structure OrbitPoint (orbit : EllipticalOrbit) where
  /-- Distance from the point to the first focus -/
  dist_focus1 : ℝ
  /-- Distance from the point to the second focus -/
  dist_focus2 : ℝ
  /-- The sum of distances to foci equals the major axis -/
  h_sum_dist : dist_focus1 + dist_focus2 = orbit.major_axis

/-- Theorem: For a point midway along the orbit, its distance to either focus is half the major axis -/
theorem midway_point_distance (orbit : EllipticalOrbit) 
    (point : OrbitPoint orbit) 
    (h_midway : point.dist_focus1 = point.dist_focus2) : 
    point.dist_focus1 = orbit.major_axis / 2 := by sorry

/-- The specific orbit from the problem -/
def yendor_orbit : EllipticalOrbit where
  major_axis := 18
  focal_distance := 12
  h_focal_lt_major := by norm_num

/-- Theorem: In Yendor's orbit, a midway point is 9 AU from each focus -/
theorem yendor_midway_distance (point : OrbitPoint yendor_orbit) 
    (h_midway : point.dist_focus1 = point.dist_focus2) : 
    point.dist_focus1 = 9 ∧ point.dist_focus2 = 9 := by sorry

end midway_point_distance_yendor_midway_distance_l1118_111817


namespace l_shape_area_l1118_111857

/-- Represents a rectangle with given width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- Represents the cut-out position -/
structure CutOutPosition where
  fromRight : ℝ
  fromBottom : ℝ

theorem l_shape_area (large : Rectangle) (cutOut : Rectangle) (pos : CutOutPosition) : 
  large.width = 12 →
  large.height = 7 →
  cutOut.width = 4 →
  cutOut.height = 3 →
  pos.fromRight = large.width / 2 →
  pos.fromBottom = large.height / 2 →
  large.area - cutOut.area = 72 := by
  sorry

end l_shape_area_l1118_111857


namespace binomial_coefficient_equality_l1118_111844

theorem binomial_coefficient_equality (n : ℕ) (h : n ≥ 6) :
  (3^5 : ℚ) * (Nat.choose n 5) = (3^6 : ℚ) * (Nat.choose n 6) ↔ n = 7 := by
  sorry

end binomial_coefficient_equality_l1118_111844


namespace puppies_per_cage_l1118_111881

/-- Given a pet store scenario with initial puppies, bought puppies, and cages used,
    calculate the number of puppies per cage. -/
theorem puppies_per_cage
  (initial_puppies : ℝ)
  (bought_puppies : ℝ)
  (cages_used : ℝ)
  (h1 : initial_puppies = 18.0)
  (h2 : bought_puppies = 3.0)
  (h3 : cages_used = 4.2) :
  (initial_puppies + bought_puppies) / cages_used = 5.0 := by
  sorry

end puppies_per_cage_l1118_111881


namespace chess_pawn_placement_l1118_111860

theorem chess_pawn_placement (n : ℕ) (hn : n = 5) : 
  (Finset.card (Finset.univ : Finset (Fin n → Fin n))) * 
  (Finset.card (Finset.univ : Finset (Equiv.Perm (Fin n)))) = 14400 :=
by sorry

end chess_pawn_placement_l1118_111860


namespace quadratic_complete_square_l1118_111848

theorem quadratic_complete_square (x : ℝ) : 
  (∃ r s : ℝ, (6 * x^2 - 24 * x - 54 = 0) ↔ ((x + r)^2 = s)) → 
  (∃ r s : ℝ, (6 * x^2 - 24 * x - 54 = 0) ↔ ((x + r)^2 = s) ∧ r + s = 11) :=
by sorry

end quadratic_complete_square_l1118_111848


namespace cubic_expression_zero_l1118_111866

theorem cubic_expression_zero (x : ℝ) (h : x^2 + 3*x - 3 = 0) : 
  x^3 + 2*x^2 - 6*x + 3 = 0 := by
  sorry

end cubic_expression_zero_l1118_111866


namespace union_M_N_complement_M_U_l1118_111867

-- Define the universe set U
def U : Set Nat := {1, 2, 3, 4, 5, 6}

-- Define set M
def M : Set Nat := {2, 3, 4}

-- Define set N
def N : Set Nat := {4, 5}

-- Theorem for the union of M and N
theorem union_M_N : M ∪ N = {2, 3, 4, 5} := by sorry

-- Theorem for the complement of M with respect to U
theorem complement_M_U : (U \ M) = {1, 5, 6} := by sorry

end union_M_N_complement_M_U_l1118_111867


namespace square_tiles_problem_l1118_111830

theorem square_tiles_problem (n : ℕ) : 
  (4 * n - 4 = 52) → n = 14 := by
  sorry

end square_tiles_problem_l1118_111830


namespace no_quadratic_composition_l1118_111836

/-- A quadratic polynomial is a polynomial of degree 2 -/
def IsQuadratic (p : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x, p x = a * x^2 + b * x + c

/-- The theorem states that there do not exist quadratic polynomials f and g
    such that their composition equals x^4 - 3x^3 + 3x^2 - x for all x -/
theorem no_quadratic_composition :
  ¬ ∃ (f g : ℝ → ℝ), IsQuadratic f ∧ IsQuadratic g ∧
    (∀ x, f (g x) = x^4 - 3*x^3 + 3*x^2 - x) :=
by sorry

end no_quadratic_composition_l1118_111836


namespace number_difference_l1118_111869

theorem number_difference (a b : ℕ) 
  (sum_eq : a + b = 21308)
  (a_ends_in_5 : a % 10 = 5)
  (b_derivation : b = 50 + (a - 5) / 10) :
  b - a = 17344 := by
sorry

end number_difference_l1118_111869


namespace average_weight_abc_l1118_111887

theorem average_weight_abc (a b c : ℝ) : 
  (a + b) / 2 = 40 →
  (b + c) / 2 = 45 →
  b = 35 →
  (a + b + c) / 3 = 45 := by
sorry

end average_weight_abc_l1118_111887


namespace final_sum_after_transformation_l1118_111894

theorem final_sum_after_transformation (x y S : ℝ) : 
  x + y = S → 3 * (x + 5) + 3 * (y + 5) = 3 * S + 30 := by
  sorry

end final_sum_after_transformation_l1118_111894


namespace book_arrangement_theorem_l1118_111870

/-- Represents the number of ways to arrange books of two subjects. -/
def arrange_books (total : ℕ) (subject1 : ℕ) (subject2 : ℕ) : ℕ :=
  2 * 2 * 2

/-- Theorem stating that arranging 4 books (2 Chinese and 2 math) 
    such that books of the same subject are not adjacent 
    results in 8 possible arrangements. -/
theorem book_arrangement_theorem :
  arrange_books 4 2 2 = 8 := by
  sorry

end book_arrangement_theorem_l1118_111870


namespace vector_addition_l1118_111837

def vector1 : Fin 2 → ℤ := ![5, -9]
def vector2 : Fin 2 → ℤ := ![-8, 14]

theorem vector_addition :
  (vector1 + vector2) = ![(-3 : ℤ), 5] := by
  sorry

end vector_addition_l1118_111837


namespace ratio_problem_l1118_111827

theorem ratio_problem (w x y z : ℝ) (hw : w ≠ 0) 
  (h1 : w / x = 2 / 3) 
  (h2 : w / y = 6 / 15) 
  (h3 : w / z = 4 / 5) : 
  (x + y) / z = 16 / 5 := by
  sorry

end ratio_problem_l1118_111827


namespace inequality_solution_set_l1118_111819

theorem inequality_solution_set : 
  {x : ℝ | (1/2 - x) * (x - 1/3) > 0} = {x : ℝ | 1/3 < x ∧ x < 1/2} :=
by sorry

end inequality_solution_set_l1118_111819


namespace exponent_division_l1118_111861

theorem exponent_division (a : ℝ) : a^8 / a^2 = a^6 := by
  sorry

end exponent_division_l1118_111861


namespace solve_cookies_problem_l1118_111806

def cookies_problem (total_cookies : ℕ) (cookies_per_guest : ℕ) : Prop :=
  total_cookies = 10 ∧ cookies_per_guest = 2 →
  total_cookies / cookies_per_guest = 5

theorem solve_cookies_problem : cookies_problem 10 2 := by
  sorry

end solve_cookies_problem_l1118_111806


namespace triathlon_bike_speed_l1118_111862

/-- Triathlon problem -/
theorem triathlon_bike_speed 
  (total_time : ℝ) 
  (swim_distance swim_speed : ℝ) 
  (run_distance run_speed : ℝ) 
  (bike_distance : ℝ) 
  (h1 : total_time = 3)
  (h2 : swim_distance = 0.5)
  (h3 : swim_speed = 1)
  (h4 : run_distance = 5)
  (h5 : run_speed = 5)
  (h6 : bike_distance = 20) :
  (bike_distance / (total_time - (swim_distance / swim_speed + run_distance / run_speed))) = 40 / 3 := by
  sorry

end triathlon_bike_speed_l1118_111862


namespace min_value_of_expression_min_value_achieved_l1118_111821

theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2*a + b = a*b) :
  ∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 2*x + y = x*y → 1/(x-1) + 2/(y-2) ≥ 2 :=
by
  sorry

theorem min_value_achieved (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2*a + b = a*b) :
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 2*x + y = x*y ∧ 1/(x-1) + 2/(y-2) = 2 :=
by
  sorry

end min_value_of_expression_min_value_achieved_l1118_111821


namespace isosceles_triangle_perimeter_l1118_111820

/-- An isosceles triangle with two sides of length 8 and one side of length 4 has a perimeter of 20 -/
theorem isosceles_triangle_perimeter : ∀ (a b c : ℝ), 
  a = 8 → b = 8 → c = 4 → a + b + c = 20 := by
  sorry

end isosceles_triangle_perimeter_l1118_111820


namespace binomial_expansion_property_l1118_111882

/-- Given (√x + 2/√x)^n, where the binomial coefficients of the second, third, and fourth terms 
    in its expansion form an arithmetic sequence, prove that n = 7 and the expansion does not 
    contain a constant term. -/
theorem binomial_expansion_property (x : ℝ) (n : ℕ) 
  (h : (Nat.choose n 2) * 2 = (Nat.choose n 1) + (Nat.choose n 3)) : 
  (n = 7) ∧ (∀ k : ℕ, 2 * k ≠ n) := by
  sorry

end binomial_expansion_property_l1118_111882


namespace functional_equation_not_surjective_l1118_111832

/-- A function from reals to natural numbers satisfying a specific functional equation -/
def FunctionalEquation (f : ℝ → ℕ) : Prop :=
  ∀ x y : ℝ, f (x + 1 / (f y : ℝ)) = f (y + 1 / (f x : ℝ))

/-- Theorem stating that a function satisfying the functional equation cannot map onto all natural numbers -/
theorem functional_equation_not_surjective (f : ℝ → ℕ) (h : FunctionalEquation f) : 
  ¬(Set.range f = Set.univ) := by
  sorry

end functional_equation_not_surjective_l1118_111832


namespace ball_probabilities_l1118_111875

def total_balls : ℕ := 6
def red_balls : ℕ := 3
def white_balls : ℕ := 2
def black_balls : ℕ := 1
def drawn_balls : ℕ := 3

def prob_one_red_one_white : ℚ := 3 / 10
def prob_at_least_two_red : ℚ := 1 / 2
def prob_no_black : ℚ := 1 / 2

theorem ball_probabilities :
  (total_balls = red_balls + white_balls + black_balls) →
  (drawn_balls ≤ total_balls) →
  (prob_one_red_one_white = 3 / 10) ∧
  (prob_at_least_two_red = 1 / 2) ∧
  (prob_no_black = 1 / 2) := by
  sorry

end ball_probabilities_l1118_111875


namespace jane_babysitting_problem_l1118_111811

/-- Represents the problem of determining when Jane stopped babysitting --/
theorem jane_babysitting_problem (jane_start_age : ℕ) (jane_current_age : ℕ) (oldest_babysat_current_age : ℕ) :
  jane_start_age = 20 →
  jane_current_age = 32 →
  oldest_babysat_current_age = 22 →
  (∀ (jane_age : ℕ) (child_age : ℕ),
    jane_start_age ≤ jane_age →
    jane_age ≤ jane_current_age →
    child_age ≤ oldest_babysat_current_age →
    child_age ≤ jane_age / 2) →
  jane_current_age - jane_start_age = 12 :=
by sorry

end jane_babysitting_problem_l1118_111811


namespace hyperbola_condition_l1118_111876

/-- For the equation x²/(2+m) - y²/(m+1) = 1 to represent a hyperbola, 
    m must satisfy: m > -1 or m < -2 -/
theorem hyperbola_condition (m : ℝ) : 
  (∃ x y : ℝ, x^2 / (2 + m) - y^2 / (m + 1) = 1 ∧ 
   (2 + m ≠ 0 ∧ m + 1 ≠ 0)) ↔ (m > -1 ∨ m < -2) :=
by sorry

end hyperbola_condition_l1118_111876


namespace intersection_M_N_l1118_111812

def U : Set Int := {-2, -1, 0, 1, 2}

def M : Set Int := {x ∈ U | x^2 ≤ x}

def N : Set Int := {x ∈ U | x^3 - 3*x^2 + 2*x = 0}

theorem intersection_M_N : M ∩ N = {0, 1} := by
  sorry

end intersection_M_N_l1118_111812


namespace cubic_roots_inequality_l1118_111871

theorem cubic_roots_inequality (a b c : ℝ) : 
  (∃ x y z : ℝ, ∀ t : ℝ, t^3 + a*t^2 + b*t + c = (t - x) * (t - y) * (t - z)) → 
  3*b ≤ a^2 := by
sorry

end cubic_roots_inequality_l1118_111871


namespace banana_distribution_l1118_111852

theorem banana_distribution (B N : ℕ) : 
  B = 2 * N ∧ B = 4 * (N - 320) → N = 640 := by
  sorry

end banana_distribution_l1118_111852


namespace log_expression_equals_two_l1118_111840

-- Define the logarithm base 10 function
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_expression_equals_two :
  (log10 2)^2 + log10 2 * log10 50 + log10 25 = 2 := by
  sorry

end log_expression_equals_two_l1118_111840


namespace tadpole_fish_ratio_l1118_111815

/-- The ratio of initial tadpoles to initial fish in a pond -/
theorem tadpole_fish_ratio :
  ∀ (initial_tadpoles : ℕ) (initial_fish : ℕ),
  initial_fish = 50 →
  ∃ (remaining_fish : ℕ) (remaining_tadpoles : ℕ),
  remaining_fish = initial_fish - 7 ∧
  remaining_tadpoles = initial_tadpoles / 2 ∧
  remaining_tadpoles = remaining_fish + 32 →
  (initial_tadpoles : ℚ) / initial_fish = 3 / 1 :=
by sorry

end tadpole_fish_ratio_l1118_111815


namespace chloe_trivia_points_l1118_111833

/-- Chloe's trivia game points calculation -/
theorem chloe_trivia_points 
  (first_round : ℕ) 
  (last_round_loss : ℕ) 
  (total_points : ℕ) 
  (h1 : first_round = 40)
  (h2 : last_round_loss = 4)
  (h3 : total_points = 86) :
  ∃ (second_round : ℕ), 
    first_round + second_round - last_round_loss = total_points ∧ 
    second_round = 50 := by
sorry

end chloe_trivia_points_l1118_111833


namespace certain_number_problem_l1118_111804

theorem certain_number_problem : 
  (∃ n : ℕ, (∀ m > n, ¬∃ p q : ℕ+, 
    p > m ∧ 
    q > m ∧ 
    17 * (p + 1) = 28 * (q + 1) ∧ 
    p + q = 43) ∧
  (∃ p q : ℕ+, 
    p > n ∧ 
    q > n ∧ 
    17 * (p + 1) = 28 * (q + 1) ∧ 
    p + q = 43)) ∧
  (∀ n' > n, ¬∃ p q : ℕ+, 
    p > n' ∧ 
    q > n' ∧ 
    17 * (p + 1) = 28 * (q + 1) ∧ 
    p + q = 43) →
  n = 15 := by sorry

end certain_number_problem_l1118_111804


namespace second_interest_rate_is_ten_percent_l1118_111856

/-- Proves that given specific investment conditions, the second interest rate is 10% -/
theorem second_interest_rate_is_ten_percent 
  (total_investment : ℝ)
  (first_investment : ℝ)
  (first_rate : ℝ)
  (h_total : total_investment = 5400)
  (h_first : first_investment = 3000)
  (h_first_rate : first_rate = 0.08)
  (h_equal_interest : first_investment * first_rate = 
    (total_investment - first_investment) * (10 / 100)) :
  (10 : ℝ) / 100 = (first_investment * first_rate) / (total_investment - first_investment) :=
sorry

end second_interest_rate_is_ten_percent_l1118_111856


namespace lg_45_equals_1_minus_m_plus_2n_l1118_111826

-- Define the logarithm base 10 function
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem lg_45_equals_1_minus_m_plus_2n (m n : ℝ) (h1 : lg 2 = m) (h2 : lg 3 = n) :
  lg 45 = 1 - m + 2 * n := by
  sorry

end lg_45_equals_1_minus_m_plus_2n_l1118_111826


namespace division_problem_l1118_111841

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) :
  dividend = 271 →
  divisor = 30 →
  remainder = 1 →
  dividend = divisor * quotient + remainder →
  quotient = 9 := by
sorry

end division_problem_l1118_111841


namespace final_row_ordered_l1118_111893

variable (m n : ℕ)
variable (C : ℕ → ℕ → ℕ)

-- C[i][j] represents the card number at row i and column j
axiom row_ordered : ∀ i j k, j < k → C i j < C i k
axiom col_ordered : ∀ i j k, i < k → C i j < C k j

theorem final_row_ordered :
  ∀ i j k, j < k → C i j < C i k :=
sorry

end final_row_ordered_l1118_111893


namespace product_of_repeating_decimal_and_twelve_l1118_111883

def repeating_decimal : ℚ := 356 / 999

theorem product_of_repeating_decimal_and_twelve :
  repeating_decimal * 12 = 1424 / 333 := by
  sorry

end product_of_repeating_decimal_and_twelve_l1118_111883


namespace polynomial_M_proof_l1118_111896

-- Define the polynomial M as a function of x and y
def M (x y : ℝ) : ℝ := 2 * x * y - 1

-- Theorem statement
theorem polynomial_M_proof :
  -- Given condition
  (∀ x y : ℝ, M x y + (2 * x^2 * y - 3 * x * y + 1) = 2 * x^2 * y - x * y) →
  -- Conclusion 1: M is correctly defined
  (∀ x y : ℝ, M x y = 2 * x * y - 1) ∧
  -- Conclusion 2: M(-1, 2) = -5
  (M (-1) 2 = -5) := by
sorry

end polynomial_M_proof_l1118_111896


namespace years_until_arun_36_l1118_111843

/-- Proves the number of years that will pass before Arun's age is 36 years -/
theorem years_until_arun_36 (arun_age deepak_age : ℕ) (future_arun_age : ℕ) : 
  arun_age / deepak_age = 5 / 7 →
  deepak_age = 42 →
  future_arun_age = 36 →
  future_arun_age - arun_age = 6 := by
  sorry

end years_until_arun_36_l1118_111843


namespace gcd_of_three_numbers_l1118_111847

theorem gcd_of_three_numbers : Nat.gcd 8885 (Nat.gcd 4514 5246) = 1 := by
  sorry

end gcd_of_three_numbers_l1118_111847


namespace even_games_player_exists_l1118_111898

/-- Represents a player in the chess tournament -/
structure Player where
  id : Nat
  gamesPlayed : Nat

/-- Represents the state of a round-robin chess tournament -/
structure ChessTournament where
  players : Finset Player
  numPlayers : Nat
  h_numPlayers : numPlayers = 17

/-- The main theorem to prove -/
theorem even_games_player_exists (tournament : ChessTournament) :
  ∃ p ∈ tournament.players, Even p.gamesPlayed :=
sorry

end even_games_player_exists_l1118_111898


namespace seed_distribution_l1118_111868

theorem seed_distribution (total_seeds : ℕ) (num_pots : ℕ) 
  (h1 : total_seeds = 10) 
  (h2 : num_pots = 4) : 
  ∃ (pot1 pot2 pot3 pot4 : ℕ), 
    pot1 = 2 * pot2 ∧ 
    pot3 = pot2 + 1 ∧ 
    pot1 + pot2 + pot3 + pot4 = total_seeds ∧ 
    pot4 = 1 := by
  sorry

end seed_distribution_l1118_111868


namespace group_photo_arrangements_l1118_111884

theorem group_photo_arrangements :
  let total_volunteers : ℕ := 6
  let male_volunteers : ℕ := 4
  let female_volunteers : ℕ := 2
  let elderly_people : ℕ := 2
  let elderly_arrangements : ℕ := 2  -- Number of ways to arrange elderly people
  let female_arrangements : ℕ := 2   -- Number of ways to arrange female volunteers
  let male_arrangements : ℕ := 24    -- Number of ways to arrange male volunteers (4!)
  
  total_volunteers = male_volunteers + female_volunteers + elderly_people →
  (elderly_arrangements * female_arrangements * male_arrangements) = 96 :=
by
  sorry

end group_photo_arrangements_l1118_111884


namespace sector_area_90_degrees_l1118_111849

/-- The area of a sector with radius 2 and central angle 90° is π. -/
theorem sector_area_90_degrees : 
  let r : ℝ := 2
  let angle_degrees : ℝ := 90
  let angle_radians : ℝ := angle_degrees * (π / 180)
  let sector_area : ℝ := (1/2) * r^2 * angle_radians
  sector_area = π := by
  sorry

end sector_area_90_degrees_l1118_111849


namespace smallest_multiple_40_over_100_l1118_111874

theorem smallest_multiple_40_over_100 : ∀ n : ℕ, n > 0 ∧ 40 ∣ n ∧ n > 100 → n ≥ 120 := by
  sorry

end smallest_multiple_40_over_100_l1118_111874


namespace ellipse_m_value_l1118_111816

/-- An ellipse with equation x²/(10-m) + y²/(m-2) = 1, major axis along y-axis, and focal length 4 -/
structure Ellipse (m : ℝ) where
  eq : ∀ (x y : ℝ), x^2 / (10 - m) + y^2 / (m - 2) = 1
  major_axis_y : m - 2 > 10 - m
  focal_length : ∃ (a b : ℝ), a^2 - b^2 = 16 ∧ a^2 = m - 2 ∧ b^2 = 10 - m

theorem ellipse_m_value (m : ℝ) (e : Ellipse m) : m = 8 := by
  sorry

end ellipse_m_value_l1118_111816


namespace find_missing_mark_l1118_111850

/-- Represents the marks obtained in a subject, ranging from 0 to 100. -/
def Marks := Fin 101

/-- Calculates the sum of marks for the given subjects. -/
def sum_marks (marks : List Marks) : Nat :=
  marks.foldl (fun acc m => acc + m.val) 0

/-- Represents the problem of finding the missing subject mark. -/
theorem find_missing_mark (english : Marks) (math : Marks) (physics : Marks) (chemistry : Marks)
    (average : Nat) (h_average : average = 69) (h_english : english.val = 66)
    (h_math : math.val = 65) (h_physics : physics.val = 77) (h_chemistry : chemistry.val = 62) :
    ∃ (biology : Marks), sum_marks [english, math, physics, chemistry, biology] / 5 = average :=
  sorry

end find_missing_mark_l1118_111850


namespace function_inequality_l1118_111897

theorem function_inequality (a b : ℝ) (f g : ℝ → ℝ) 
  (h₁ : a ≤ b)
  (h₂ : DifferentiableOn ℝ f (Set.Icc a b))
  (h₃ : DifferentiableOn ℝ g (Set.Icc a b))
  (h₄ : ∀ x ∈ Set.Icc a b, deriv f x > deriv g x)
  (h₅ : f a = g a) :
  ∀ x ∈ Set.Icc a b, f x ≥ g x :=
by sorry

end function_inequality_l1118_111897


namespace only_valid_pythagorean_triple_l1118_111889

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

theorem only_valid_pythagorean_triple :
  ¬ is_pythagorean_triple 3 4 5 ∧
  ¬ is_pythagorean_triple 2 2 (2 * 2) ∧
  ¬ is_pythagorean_triple 4 5 6 ∧
  is_pythagorean_triple 5 12 13 :=
by sorry

end only_valid_pythagorean_triple_l1118_111889


namespace intersection_implies_b_range_l1118_111824

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p | p.1^2 + 2*p.2^2 = 3}
def N (m b : ℝ) : Set (ℝ × ℝ) := {p | p.2 = m*p.1 + b}

-- State the theorem
theorem intersection_implies_b_range :
  (∀ m : ℝ, ∃ p : ℝ × ℝ, p ∈ M ∩ N m b) →
  b ∈ Set.Icc (-Real.sqrt 6 / 2) (Real.sqrt 6 / 2) := by
  sorry

end intersection_implies_b_range_l1118_111824


namespace find_larger_number_l1118_111886

theorem find_larger_number (L S : ℕ) (h1 : L - S = 1515) (h2 : L = 16 * S + 15) : L = 1615 := by
  sorry

end find_larger_number_l1118_111886


namespace direct_square_root_most_suitable_l1118_111829

/-- The quadratic equation to be solved -/
def quadratic_equation (x : ℝ) : Prop := (x - 1)^2 = 4

/-- Possible solution methods for quadratic equations -/
inductive SolutionMethod
  | CompletingSquare
  | QuadraticFormula
  | Factoring
  | DirectSquareRoot

/-- Predicate to determine if a method is the most suitable for solving a given equation -/
def is_most_suitable_method (eq : ℝ → Prop) (method : SolutionMethod) : Prop :=
  ∀ other_method : SolutionMethod, method = other_method ∨ 
    (∃ (complexity_measure : SolutionMethod → ℕ), 
      complexity_measure method < complexity_measure other_method)

/-- Theorem stating that the direct square root method is the most suitable for the given equation -/
theorem direct_square_root_most_suitable :
  is_most_suitable_method quadratic_equation SolutionMethod.DirectSquareRoot :=
sorry

end direct_square_root_most_suitable_l1118_111829


namespace find_f_one_l1118_111879

/-- A function with the property f(x + y) = f(x) + f(y) + 7xy + 4 -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ x y, f (x + y) = f x + f y + 7 * x * y + 4

theorem find_f_one (f : ℝ → ℝ) (h1 : special_function f) (h2 : f 2 + f 5 = 125) :
  f 1 = 4 := by
  sorry

end find_f_one_l1118_111879


namespace santa_candy_remainders_l1118_111803

theorem santa_candy_remainders (N : ℕ) (x : ℕ) (h : N = 35 * x + 7) :
  N % 15 ∈ ({2, 7, 12} : Finset ℕ) := by
  sorry

end santa_candy_remainders_l1118_111803


namespace limit_a_minus_log_n_eq_zero_l1118_111818

noncomputable def a : ℕ → ℝ
  | 0 => 1
  | n + 1 => a n + Real.exp (-a n)

theorem limit_a_minus_log_n_eq_zero :
  ∃ L : ℝ, L = 0 ∧ ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a n - Real.log n - L| < ε :=
sorry

end limit_a_minus_log_n_eq_zero_l1118_111818


namespace apple_bag_price_l1118_111839

-- Define the given quantities
def total_harvest : ℕ := 405
def juice_amount : ℕ := 90
def restaurant_amount : ℕ := 60
def bag_size : ℕ := 5
def total_revenue : ℕ := 408

-- Define the selling price of one bag
def selling_price : ℚ := 8

-- Theorem to prove
theorem apple_bag_price :
  (total_harvest - juice_amount - restaurant_amount) / bag_size * selling_price = total_revenue :=
by sorry

end apple_bag_price_l1118_111839


namespace angle_bisector_slope_l1118_111864

/-- The slope of the angle bisector of the acute angle formed at the origin
    by the lines y = x and y = 4x is -5/3 + √2. -/
theorem angle_bisector_slope : ℝ := by
  -- Define the slopes of the two lines
  let m₁ : ℝ := 1
  let m₂ : ℝ := 4

  -- Define the slope of the angle bisector
  let k : ℝ := (m₁ + m₂ + Real.sqrt (1 + m₁^2 + m₂^2)) / (1 - m₁ * m₂)

  -- Prove that k equals -5/3 + √2
  sorry

end angle_bisector_slope_l1118_111864


namespace smallest_sum_of_primes_and_composites_l1118_111831

def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 0 → d < n → n % d ≠ 0

def isComposite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

def formNumbers (digits : List ℕ) : List ℕ :=
  sorry

theorem smallest_sum_of_primes_and_composites (digits : List ℕ) :
  digits = [1, 2, 3, 4, 5, 6, 7, 8, 9] →
  (∃ nums : List ℕ,
    nums = formNumbers digits ∧
    (∀ n ∈ nums, isPrime n) ∧
    nums.sum = 318 ∧
    (∀ otherNums : List ℕ,
      otherNums = formNumbers digits →
      (∀ n ∈ otherNums, isPrime n) →
      otherNums.sum ≥ 318)) ∧
  (∃ nums : List ℕ,
    nums = formNumbers digits ∧
    (∀ n ∈ nums, isComposite n) ∧
    nums.sum = 127 ∧
    (∀ otherNums : List ℕ,
      otherNums = formNumbers digits →
      (∀ n ∈ otherNums, isComposite n) →
      otherNums.sum ≥ 127)) :=
by sorry


end smallest_sum_of_primes_and_composites_l1118_111831


namespace greatest_common_measure_l1118_111801

theorem greatest_common_measure (a b c : ℕ) (ha : a = 729000) (hb : b = 1242500) (hc : c = 32175) :
  Nat.gcd a (Nat.gcd b c) = 225 := by
  sorry

end greatest_common_measure_l1118_111801


namespace triangle_max_area_l1118_111877

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a * cos(B) + b * cos(A) = √3 and the area of its circumcircle is π,
    then the maximum area of triangle ABC is 3√3/4. -/
theorem triangle_max_area (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  a * Real.cos B + b * Real.cos A = Real.sqrt 3 →
  (π * (a / (2 * Real.sin A))^2) = π →
  ∃ (S : ℝ), S = (1/2) * a * b * Real.sin C ∧
              S ≤ (3 * Real.sqrt 3) / 4 ∧
              (∀ (S' : ℝ), S' = (1/2) * a * b * Real.sin C → S' ≤ S) :=
by sorry

end triangle_max_area_l1118_111877


namespace third_person_teeth_removal_l1118_111800

theorem third_person_teeth_removal (total_teeth : ℕ) (total_removed : ℕ) 
  (first_person_fraction : ℚ) (second_person_fraction : ℚ) (last_person_removed : ℕ) :
  total_teeth = 32 →
  total_removed = 40 →
  first_person_fraction = 1/4 →
  second_person_fraction = 3/8 →
  last_person_removed = 4 →
  (total_removed - 
    (first_person_fraction * total_teeth + 
     second_person_fraction * total_teeth + 
     last_person_removed)) / total_teeth = 1/2 := by
  sorry

#check third_person_teeth_removal

end third_person_teeth_removal_l1118_111800


namespace watch_price_equation_l1118_111842

/-- The original cost price of a watch satisfies the equation relating its discounted price and taxed price with profit. -/
theorem watch_price_equation (C : ℝ) : C > 0 → 0.855 * C + 540 = 1.2096 * C := by
  sorry

end watch_price_equation_l1118_111842


namespace logarithm_equation_l1118_111805

theorem logarithm_equation (a b c : ℝ) 
  (eq1 : Real.log 3 = 2*a - b)
  (eq2 : Real.log 5 = a + c)
  (eq3 : Real.log 8 = 3 - 3*a - 3*c)
  (eq4 : Real.log 9 = 4*a - 2*b) :
  Real.log 15 = 3*a - b + c := by sorry

end logarithm_equation_l1118_111805


namespace new_student_weight_l1118_111863

/-- The weight of the new student given the conditions of the problem -/
theorem new_student_weight (n : ℕ) (initial_weight replaced_weight new_weight : ℝ) 
  (h1 : n = 4)
  (h2 : replaced_weight = 96)
  (h3 : (initial_weight - replaced_weight + new_weight) / n = initial_weight / n - 8) :
  new_weight = 64 := by
  sorry

end new_student_weight_l1118_111863


namespace complex_power_six_l1118_111880

theorem complex_power_six (i : ℂ) (h : i^2 = -1) : (1 + i)^6 = -8*i := by
  sorry

end complex_power_six_l1118_111880


namespace work_completion_time_l1118_111865

/-- Given that A can do a work in 8 days and A and B together can do the work in 16/3 days,
    prove that B can do the work alone in 16 days. -/
theorem work_completion_time (a b : ℝ) (ha : a = 8) (hab : 1 / a + 1 / b = 3 / 16) :
  b = 16 := by sorry

end work_completion_time_l1118_111865


namespace parallel_line_through_point_l1118_111823

/-- Given a line y = kx + b that is parallel to y = 2x - 3 and passes through (1, -5),
    prove that its equation is y = 2x - 7 -/
theorem parallel_line_through_point (k b : ℝ) : 
  (∀ x y, y = k * x + b ↔ y = 2 * x - 3) →  -- parallelism condition
  (-5 : ℝ) = k * 1 + b →                   -- point condition
  ∀ x y, y = k * x + b ↔ y = 2 * x - 7 :=   -- conclusion
by sorry

end parallel_line_through_point_l1118_111823


namespace initial_watermelons_l1118_111807

theorem initial_watermelons (eaten : ℕ) (left : ℕ) (initial : ℕ) : 
  eaten = 3 → left = 1 → initial = eaten + left → initial = 4 := by
  sorry

end initial_watermelons_l1118_111807


namespace lcm_18_24_l1118_111854

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  sorry

end lcm_18_24_l1118_111854


namespace bartender_cheating_l1118_111814

theorem bartender_cheating (total_cost : ℚ) (whiskey_cost pipe_cost : ℕ) : 
  total_cost = 11.80 ∧ whiskey_cost = 3 ∧ pipe_cost = 6 → ¬(∃ n : ℕ, total_cost = n * 3) :=
by sorry

end bartender_cheating_l1118_111814


namespace sum_of_two_numbers_l1118_111835

theorem sum_of_two_numbers (x y : ℤ) : y = 2 * x - 3 → x = 14 → x + y = 39 := by
  sorry

end sum_of_two_numbers_l1118_111835


namespace sum_of_medians_is_63_l1118_111855

/-- Represents the scores of a basketball player -/
def Scores := List ℕ

/-- Calculates the median of a list of scores -/
def median (scores : Scores) : ℚ :=
  sorry

/-- Player A's scores -/
def scoresA : Scores :=
  sorry

/-- Player B's scores -/
def scoresB : Scores :=
  sorry

/-- The sum of median scores of players A and B is 63 -/
theorem sum_of_medians_is_63 :
  median scoresA + median scoresB = 63 :=
by sorry

end sum_of_medians_is_63_l1118_111855


namespace geometric_series_sum_l1118_111853

theorem geometric_series_sum : 
  let a : ℚ := 1/4
  let r : ℚ := 1/4
  let n : ℕ := 5
  let series_sum := (a * (1 - r^n)) / (1 - r)
  series_sum = 341/1024 := by sorry

end geometric_series_sum_l1118_111853


namespace expected_steps_is_five_l1118_111825

/-- The coloring process on the unit interval [0,1] --/
structure ColoringProcess where
  /-- The random selection of x in [0,1] --/
  select_x : Unit → Real
  /-- The coloring rule for x ≤ 1/2 --/
  color_left (x : Real) : Set Real := { y | x ≤ y ∧ y ≤ x + 1/2 }
  /-- The coloring rule for x > 1/2 --/
  color_right (x : Real) : Set Real := { y | x ≤ y ∧ y ≤ 1 } ∪ { y | 0 ≤ y ∧ y ≤ x - 1/2 }

/-- The expected number of steps to color the entire interval --/
def expected_steps (process : ColoringProcess) : Real :=
  5  -- The actual value we want to prove

/-- The theorem stating that the expected number of steps is 5 --/
theorem expected_steps_is_five (process : ColoringProcess) :
  expected_steps process = 5 := by sorry

end expected_steps_is_five_l1118_111825


namespace trigonometric_simplification_l1118_111888

theorem trigonometric_simplification (θ : Real) : 
  (Real.sin (2 * Real.pi - θ) * Real.cos (Real.pi + θ) * Real.cos (Real.pi / 2 + θ) * Real.cos (11 * Real.pi / 2 - θ)) / 
  (Real.cos (Real.pi - θ) * Real.sin (3 * Real.pi - θ) * Real.sin (-Real.pi - θ) * Real.sin (9 * Real.pi / 2 + θ)) = 
  -Real.tan θ := by
  sorry

end trigonometric_simplification_l1118_111888


namespace ladder_slide_l1118_111813

theorem ladder_slide (ladder_length : Real) (initial_distance : Real) (top_slip : Real) (foot_slide : Real) : 
  ladder_length = 30 ∧ 
  initial_distance = 8 ∧ 
  top_slip = 4 ∧ 
  foot_slide = 2 →
  (ladder_length ^ 2 = initial_distance ^ 2 + (Real.sqrt (ladder_length ^ 2 - initial_distance ^ 2)) ^ 2) ∧
  (ladder_length ^ 2 = (initial_distance + foot_slide) ^ 2 + (Real.sqrt (ladder_length ^ 2 - initial_distance ^ 2) - top_slip) ^ 2) :=
by sorry

end ladder_slide_l1118_111813


namespace quadratic_equation_root_l1118_111802

theorem quadratic_equation_root (x : ℝ) : x^2 + 6*x + 4 = 0 ↔ x = Real.sqrt 5 - 3 := by
  sorry

end quadratic_equation_root_l1118_111802


namespace max_stick_length_l1118_111872

theorem max_stick_length (a b c : ℕ) (ha : a = 24) (hb : b = 32) (hc : c = 44) :
  Nat.gcd a (Nat.gcd b c) = 4 := by
  sorry

end max_stick_length_l1118_111872


namespace matrix_power_result_l1118_111885

theorem matrix_power_result (B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : B.mulVec ![3, -1] = ![6, -2]) : 
  (B ^ 3).mulVec ![3, -1] = ![24, -8] := by
  sorry

end matrix_power_result_l1118_111885


namespace completing_square_equivalence_l1118_111809

theorem completing_square_equivalence (x : ℝ) :
  x^2 - 6*x - 10 = 0 ↔ (x - 3)^2 = 19 := by
  sorry

end completing_square_equivalence_l1118_111809


namespace john_weekly_loss_l1118_111895

/-- Represents John's tire production and sales scenario -/
structure TireProduction where
  daily_production : ℕ
  production_cost : ℚ
  selling_price_multiplier : ℚ
  potential_daily_sales : ℕ

/-- Calculates the weekly loss due to production limitations -/
def weekly_loss (t : TireProduction) : ℚ :=
  let profit_per_tire := t.production_cost * (t.selling_price_multiplier - 1)
  let daily_loss := profit_per_tire * (t.potential_daily_sales - t.daily_production)
  7 * daily_loss

/-- Theorem stating that given John's production scenario, the weekly loss is $175,000 -/
theorem john_weekly_loss :
  let john_production : TireProduction := {
    daily_production := 1000,
    production_cost := 250,
    selling_price_multiplier := 1.5,
    potential_daily_sales := 1200
  }
  weekly_loss john_production = 175000 := by
  sorry

end john_weekly_loss_l1118_111895


namespace trapezoid_division_common_side_l1118_111859

theorem trapezoid_division_common_side
  (a b k p : ℝ)
  (h1 : a > b)
  (h2 : k > 0)
  (h3 : p > 0) :
  let x := Real.sqrt ((k * a^2 + p * b^2) / (p + k))
  ∃ (h1 h2 : ℝ), 
    h1 > 0 ∧ h2 > 0 ∧
    (b + x) * h1 / ((a + x) * h2) = k / p ∧
    x > b ∧ x < a :=
by sorry

end trapezoid_division_common_side_l1118_111859


namespace choir_arrangement_max_l1118_111891

theorem choir_arrangement_max (n : ℕ) : 
  (∃ k : ℕ, n = k^2 + 11) ∧ 
  (∃ x : ℕ, n = x * (x + 5)) →
  n ≤ 126 :=
sorry

end choir_arrangement_max_l1118_111891


namespace max_value_x_plus_2y_l1118_111851

theorem max_value_x_plus_2y (x y : ℝ) 
  (h1 : 4 * x + 3 * y ≤ 12) 
  (h2 : 3 * x + 6 * y ≤ 9) : 
  ∃ (max : ℝ), max = 3 ∧ x + 2 * y ≤ max ∧ 
  ∀ (z : ℝ), (∃ (a b : ℝ), 4 * a + 3 * b ≤ 12 ∧ 3 * a + 6 * b ≤ 9 ∧ z = a + 2 * b) → z ≤ max :=
sorry

end max_value_x_plus_2y_l1118_111851
