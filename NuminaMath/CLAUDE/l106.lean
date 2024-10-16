import Mathlib

namespace NUMINAMATH_CALUDE_expression_equals_14_l106_10669

theorem expression_equals_14 (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h_sum : x + y + z = 0) (h_prod : x*y + x*z + y*z ≠ 0) :
  (x^7 + y^7 + z^7) / (x*y*z*(x^2 + y^2 + z^2)) = 14 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_14_l106_10669


namespace NUMINAMATH_CALUDE_g_zero_at_three_l106_10697

def g (x s : ℝ) : ℝ := 3 * x^4 + 2 * x^3 - x^2 - 4 * x + s

theorem g_zero_at_three (s : ℝ) : g 3 s = 0 ↔ s = -276 := by sorry

end NUMINAMATH_CALUDE_g_zero_at_three_l106_10697


namespace NUMINAMATH_CALUDE_daniel_candy_distribution_l106_10690

/-- Given that Daniel has 24 pieces of candy and 5 sisters, prove that the least number of
    candies he should take away to distribute the remaining candies equally is 4. -/
theorem daniel_candy_distribution (total_candy : ℕ) (num_sisters : ℕ) 
  (h1 : total_candy = 24) (h2 : num_sisters = 5) :
  let remaining_candy := total_candy - (total_candy / num_sisters) * num_sisters
  remaining_candy = 4 := by
  sorry

end NUMINAMATH_CALUDE_daniel_candy_distribution_l106_10690


namespace NUMINAMATH_CALUDE_inequality_implication_l106_10686

theorem inequality_implication (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : a - d > b - c := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l106_10686


namespace NUMINAMATH_CALUDE_solution_count_33_l106_10611

/-- The number of solutions to 3x + 2y + z = n in positive integers x, y, z -/
def solution_count (n : ℕ+) : ℕ := sorry

/-- The set of possible values for n -/
def possible_values : Set ℕ+ := {22, 24, 25}

/-- Theorem: If the equation 3x + 2y + z = n has exactly 33 solutions in positive integers x, y, and z,
    then n is in the set {22, 24, 25} -/
theorem solution_count_33 (n : ℕ+) : solution_count n = 33 → n ∈ possible_values := by sorry

end NUMINAMATH_CALUDE_solution_count_33_l106_10611


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l106_10694

-- Define set A
def A : Set ℝ := {x | 1 < x ∧ x < 4}

-- Define set B
def B : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

-- Theorem statement
theorem intersection_complement_equality : A ∩ (Set.univ \ B) = Set.Ioo 3 4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l106_10694


namespace NUMINAMATH_CALUDE_candidate_vote_percentage_l106_10610

/-- Theorem: Given a total of 6000 votes and a candidate losing by 1800 votes,
    the percentage of votes the candidate received is 35%. -/
theorem candidate_vote_percentage
  (total_votes : ℕ)
  (vote_difference : ℕ)
  (h_total : total_votes = 6000)
  (h_diff : vote_difference = 1800) :
  (total_votes - vote_difference) * 100 / (2 * total_votes) = 35 := by
  sorry

end NUMINAMATH_CALUDE_candidate_vote_percentage_l106_10610


namespace NUMINAMATH_CALUDE_age_sum_proof_l106_10625

theorem age_sum_proof (a b c : ℕ+) : 
  a * b * c = 72 → 
  a ≤ b ∧ a ≤ c → 
  a + b + c = 15 := by
sorry

end NUMINAMATH_CALUDE_age_sum_proof_l106_10625


namespace NUMINAMATH_CALUDE_problem_solution_l106_10691

theorem problem_solution (x y : ℝ) (h1 : x + y = 10) (h2 : x / y = 7 / 3) : x = 7 ∧ y = 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l106_10691


namespace NUMINAMATH_CALUDE_missing_fibonacci_term_l106_10653

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci n + fibonacci (n + 1)

theorem missing_fibonacci_term : ∃ x : ℕ, 
  fibonacci 0 = 1 ∧ 
  fibonacci 1 = 1 ∧ 
  fibonacci 2 = 2 ∧ 
  fibonacci 3 = 3 ∧ 
  fibonacci 4 = 5 ∧ 
  fibonacci 5 = x ∧ 
  fibonacci 6 = 13 ∧ 
  x = 8 := by
  sorry

end NUMINAMATH_CALUDE_missing_fibonacci_term_l106_10653


namespace NUMINAMATH_CALUDE_sum_of_triangle_perimeters_l106_10658

/-- Given an equilateral triangle with side length 45 cm, if we repeatedly form new equilateral
    triangles by joining the midpoints of the previous triangle's sides, the sum of the perimeters
    of all these triangles is 270 cm. -/
theorem sum_of_triangle_perimeters (s : ℝ) (h : s = 45) :
  let perimeter_sum := (3 * s) / (1 - (1/2 : ℝ))
  perimeter_sum = 270 := by sorry

end NUMINAMATH_CALUDE_sum_of_triangle_perimeters_l106_10658


namespace NUMINAMATH_CALUDE_probability_higher_first_lower_second_l106_10683

def card_set : Finset ℕ := Finset.range 7

def favorable_outcomes : Finset (ℕ × ℕ) :=
  Finset.filter (fun p => p.1 > p.2) (card_set.product card_set)

theorem probability_higher_first_lower_second :
  (favorable_outcomes.card : ℚ) / (card_set.card * card_set.card) = 3 / 7 := by
  sorry

end NUMINAMATH_CALUDE_probability_higher_first_lower_second_l106_10683


namespace NUMINAMATH_CALUDE_gianna_savings_period_l106_10607

/-- Proves that Gianna saved money for 365 days given the conditions -/
theorem gianna_savings_period (daily_savings : ℕ) (total_savings : ℕ) 
  (h1 : daily_savings = 39)
  (h2 : total_savings = 14235) :
  total_savings / daily_savings = 365 := by
  sorry

end NUMINAMATH_CALUDE_gianna_savings_period_l106_10607


namespace NUMINAMATH_CALUDE_product_scaling_l106_10664

theorem product_scaling (a b c : ℝ) (h : (268 : ℝ) * 74 = 19732) :
  2.68 * 0.74 = 1.9732 := by
  sorry

end NUMINAMATH_CALUDE_product_scaling_l106_10664


namespace NUMINAMATH_CALUDE_range_of_y₂_l106_10630

-- Define the ellipse C₁
def C₁ (x y : ℝ) : Prop := x^2 / 3 + y^2 / 2 = 1

-- Define the foci F₁ and F₂
def F₁ : ℝ × ℝ := (-1, 0)
def F₂ : ℝ × ℝ := (1, 0)

-- Define line l₁
def l₁ (x : ℝ) : Prop := x = -1

-- Define line l₂
def l₂ (y t : ℝ) : Prop := y = t

-- Define point P
def P (t : ℝ) : ℝ × ℝ := (-1, t)

-- Define curve C₂
def C₂ (x y : ℝ) : Prop := y^2 = 4 * x

-- Define points A, B, and C on C₂
def A : ℝ × ℝ := (1, 2)
def B (x₁ y₁ : ℝ) : Prop := C₂ x₁ y₁ ∧ (x₁, y₁) ≠ A
def C (x₂ y₂ : ℝ) : Prop := C₂ x₂ y₂ ∧ (x₂, y₂) ≠ A

-- AB perpendicular to BC
def perpendicular (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  (x₁ - 1) * (x₂ - x₁) + (y₁ - 2) * (y₂ - y₁) = 0

-- Theorem statement
theorem range_of_y₂ (x₁ y₁ x₂ y₂ : ℝ) :
  B x₁ y₁ → C x₂ y₂ → perpendicular x₁ y₁ x₂ y₂ →
  y₂ ∈ (Set.Iic (-6) \ {-6}) ∪ Set.Ici 10 :=
sorry

end NUMINAMATH_CALUDE_range_of_y₂_l106_10630


namespace NUMINAMATH_CALUDE_prob_higher_roll_and_sum_l106_10652

/-- The number of sides on a standard die -/
def die_sides : ℕ := 6

/-- The probability of rolling a higher number on one die compared to another -/
def prob_higher_roll : ℚ :=
  (die_sides * (die_sides - 1) / 2) / (die_sides^2 : ℚ)

/-- The sum of the numerator and denominator of the probability fraction in lowest terms -/
def sum_num_denom : ℕ := 17

theorem prob_higher_roll_and_sum :
  prob_higher_roll = 5/12 ∧ sum_num_denom = 17 := by sorry

end NUMINAMATH_CALUDE_prob_higher_roll_and_sum_l106_10652


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l106_10660

/-- Two numbers are inversely proportional if their product is constant -/
def InverselyProportional (x y : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ x * y = k

theorem inverse_proportion_problem (x y : ℝ) 
  (h1 : InverselyProportional x y)
  (h2 : ∃ x₀ y₀ : ℝ, x₀ + y₀ = 60 ∧ x₀ = 3 * y₀ ∧ InverselyProportional x₀ y₀) :
  x = 12 → y = 56.25 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l106_10660


namespace NUMINAMATH_CALUDE_number_ordering_l106_10651

theorem number_ordering : (2 : ℕ)^30 < (6 : ℕ)^10 ∧ (6 : ℕ)^10 < (3 : ℕ)^20 := by sorry

end NUMINAMATH_CALUDE_number_ordering_l106_10651


namespace NUMINAMATH_CALUDE_car_sale_profit_ratio_l106_10645

theorem car_sale_profit_ratio (c₁ c₂ : ℝ) (h : c₁ > 0 ∧ c₂ > 0) :
  (1.1 * c₁ + 0.9 * c₂ - (c₁ + c₂)) / (c₁ + c₂) = 0.01 →
  c₂ = (9 / 11) * c₁ := by
  sorry

end NUMINAMATH_CALUDE_car_sale_profit_ratio_l106_10645


namespace NUMINAMATH_CALUDE_walters_age_2009_l106_10609

theorem walters_age_2009 (walter_age_2004 : ℝ) (grandmother_age_2004 : ℝ) : 
  walter_age_2004 = grandmother_age_2004 / 3 →
  (2004 - walter_age_2004) + (2004 - grandmother_age_2004) = 4018 →
  walter_age_2004 + 5 = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_walters_age_2009_l106_10609


namespace NUMINAMATH_CALUDE_min_value_of_expression_l106_10699

theorem min_value_of_expression (a b c : ℝ) 
  (sum_condition : a + b + c = -1)
  (product_condition : a * b * c ≤ -3) :
  (a * b + 1) / (a + b) + (b * c + 1) / (b + c) + (c * a + 1) / (c + a) ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l106_10699


namespace NUMINAMATH_CALUDE_share_difference_for_given_distribution_l106_10693

/-- Represents the distribution of money among three people -/
structure Distribution where
  ratio1 : ℕ
  ratio2 : ℕ
  ratio3 : ℕ
  share2 : ℕ

/-- Calculates the difference between the first and third person's shares -/
def shareDifference (d : Distribution) : ℕ :=
  let part := d.share2 / d.ratio2
  let share1 := part * d.ratio1
  let share3 := part * d.ratio3
  share3 - share1

/-- Theorem stating the difference between shares for the given distribution -/
theorem share_difference_for_given_distribution :
  ∀ d : Distribution,
    d.ratio1 = 3 ∧ d.ratio2 = 5 ∧ d.ratio3 = 9 ∧ d.share2 = 1500 →
    shareDifference d = 1800 := by
  sorry

#check share_difference_for_given_distribution

end NUMINAMATH_CALUDE_share_difference_for_given_distribution_l106_10693


namespace NUMINAMATH_CALUDE_fraction_equivalence_l106_10624

theorem fraction_equivalence (x : ℝ) (h : x ≠ 5) :
  ¬(∀ x : ℝ, x ≠ 5 → (x + 3) / (x - 5) = 3 / (-5)) :=
by sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l106_10624


namespace NUMINAMATH_CALUDE_hyperbola_ellipse_relationship_range_of_m_l106_10621

def P (m : ℝ) : Prop := ∃ (x y : ℝ), x^2/(m-1) + y^2/(m-4) = 1 ∧ (m-1)*(m-4) < 0

def Q (m : ℝ) : Prop := ∃ (x y : ℝ), x^2/(m-2) + y^2/(4-m) = 1 ∧ m-2 > 0 ∧ 4-m > 0 ∧ m-2 ≠ 4-m

theorem hyperbola_ellipse_relationship (m : ℝ) :
  (P m → Q m) ∧ ¬(Q m → P m) :=
sorry

theorem range_of_m (m : ℝ) :
  (¬(P m ∧ Q m) ∧ (P m ∨ Q m)) → ((1 < m ∧ m ≤ 2) ∨ m = 3) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_ellipse_relationship_range_of_m_l106_10621


namespace NUMINAMATH_CALUDE_m_values_l106_10635

def A (m : ℝ) : Set ℝ := {1, 2, 3, m}
def B (m : ℝ) : Set ℝ := {m^2, 3}

theorem m_values (m : ℝ) :
  A m ∪ B m = A m →
  m = -1 ∨ m = Real.sqrt 2 ∨ m = -Real.sqrt 2 ∨ m = 0 := by
  sorry

end NUMINAMATH_CALUDE_m_values_l106_10635


namespace NUMINAMATH_CALUDE_yellow_balls_count_l106_10656

theorem yellow_balls_count (purple_count blue_count : ℕ) 
  (min_tries : ℕ) (yellow_count : ℕ) : 
  purple_count = 7 → 
  blue_count = 5 → 
  min_tries = 19 →
  yellow_count = min_tries - (purple_count + blue_count + 1) →
  yellow_count = 6 := by
  sorry

end NUMINAMATH_CALUDE_yellow_balls_count_l106_10656


namespace NUMINAMATH_CALUDE_cosine_product_inequality_l106_10631

theorem cosine_product_inequality (a b c x : ℝ) :
  -(Real.sin ((b - c) / 2))^2 ≤ Real.cos (a * x + b) * Real.cos (a * x + c) ∧
  Real.cos (a * x + b) * Real.cos (a * x + c) ≤ (Real.cos ((b - c) / 2))^2 := by
  sorry

end NUMINAMATH_CALUDE_cosine_product_inequality_l106_10631


namespace NUMINAMATH_CALUDE_sum_of_squares_geq_sum_of_products_inequality_of_square_roots_l106_10605

-- Statement 1
theorem sum_of_squares_geq_sum_of_products (a b c : ℝ) : 
  a^2 + b^2 + c^2 ≥ a*b + a*c + b*c := by sorry

-- Statement 2
theorem inequality_of_square_roots : 
  Real.sqrt 6 + Real.sqrt 7 > 2 * Real.sqrt 2 + Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_sum_of_squares_geq_sum_of_products_inequality_of_square_roots_l106_10605


namespace NUMINAMATH_CALUDE_product_of_primes_with_square_sum_l106_10667

theorem product_of_primes_with_square_sum (p₁ p₂ p₃ p₄ : ℕ) : 
  Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ →
  p₁^2 + p₂^2 + p₃^2 + p₄^2 = 476 →
  p₁ * p₂ * p₃ * p₄ = 1989 := by
sorry

end NUMINAMATH_CALUDE_product_of_primes_with_square_sum_l106_10667


namespace NUMINAMATH_CALUDE_rod_cutting_l106_10637

theorem rod_cutting (rod_length : ℝ) (piece_length : ℝ) (h1 : rod_length = 47.5) (h2 : piece_length = 0.40) :
  ⌊rod_length / piece_length⌋ = 118 := by
sorry

end NUMINAMATH_CALUDE_rod_cutting_l106_10637


namespace NUMINAMATH_CALUDE_opposite_of_one_third_l106_10650

theorem opposite_of_one_third : 
  (opposite : ℚ → ℚ) (1/3) = -(1/3) := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_one_third_l106_10650


namespace NUMINAMATH_CALUDE_books_sold_l106_10632

theorem books_sold (initial : ℕ) (added : ℕ) (final : ℕ) : 
  initial = 41 → added = 2 → final = 10 → initial - (initial - final + added) = 33 :=
by sorry

end NUMINAMATH_CALUDE_books_sold_l106_10632


namespace NUMINAMATH_CALUDE_cube_surface_area_increase_l106_10668

theorem cube_surface_area_increase (L : ℝ) (h : L > 0) :
  let original_surface_area := 6 * L^2
  let new_edge_length := 1.4 * L
  let new_surface_area := 6 * new_edge_length^2
  (new_surface_area - original_surface_area) / original_surface_area = 0.96 := by
sorry

end NUMINAMATH_CALUDE_cube_surface_area_increase_l106_10668


namespace NUMINAMATH_CALUDE_select_books_with_one_specific_l106_10692

/-- The number of ways to select k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of books -/
def total_books : ℕ := 8

/-- The number of books to be selected -/
def books_to_select : ℕ := 5

/-- The number of ways to select 5 books from 8 books, where one specific book must always be included -/
def ways_to_select : ℕ := choose (total_books - 1) (books_to_select - 1)

theorem select_books_with_one_specific :
  ways_to_select = 35 :=
sorry

end NUMINAMATH_CALUDE_select_books_with_one_specific_l106_10692


namespace NUMINAMATH_CALUDE_fish_tagging_theorem_l106_10647

/-- The number of fish in the pond -/
def total_fish : ℕ := 3200

/-- The number of fish caught in the second catch -/
def second_catch : ℕ := 80

/-- The number of tagged fish found in the second catch -/
def tagged_in_second : ℕ := 2

/-- The number of fish initially caught, tagged, and returned -/
def initially_tagged : ℕ := 80

theorem fish_tagging_theorem :
  (tagged_in_second : ℚ) / second_catch = initially_tagged / total_fish →
  initially_tagged = 80 := by
  sorry

end NUMINAMATH_CALUDE_fish_tagging_theorem_l106_10647


namespace NUMINAMATH_CALUDE_class_vision_median_l106_10639

/-- Represents the vision data for a class of students -/
structure VisionData where
  visions : List ℝ
  counts : List ℕ
  total_students : ℕ

/-- Calculates the median of a VisionData set -/
def median (data : VisionData) : ℝ :=
  sorry

/-- The specific vision data for the class -/
def class_vision_data : VisionData :=
  { visions := [4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0],
    counts := [1, 2, 6, 3, 3, 4, 1, 2, 5, 7, 5],
    total_students := 39 }

/-- Theorem stating that the median of the class vision data is 4.6 -/
theorem class_vision_median :
  median class_vision_data = 4.6 := by
  sorry

end NUMINAMATH_CALUDE_class_vision_median_l106_10639


namespace NUMINAMATH_CALUDE_rocket_launch_l106_10646

/-- Rocket launch problem -/
theorem rocket_launch (a : ℝ) (g : ℝ) (t : ℝ) (h_object : ℝ) : 
  a = 20 → g = 10 → t = 40 → h_object = 45000 →
  let v₀ : ℝ := a * t
  let h₀ : ℝ := (1/2) * a * t^2
  let t_max : ℝ := v₀ / g
  let h_max : ℝ := h₀ + v₀ * t_max - (1/2) * g * t_max^2
  h_max = 48000 ∧ h_max > h_object :=
by sorry

end NUMINAMATH_CALUDE_rocket_launch_l106_10646


namespace NUMINAMATH_CALUDE_diborane_combustion_heat_correct_l106_10622

/-- Represents the heat of vaporization of water in kJ/mol -/
def water_vaporization_heat : ℝ := 44

/-- Represents the amount of diborane in moles -/
def diborane_amount : ℝ := 0.3

/-- Represents the heat released during combustion in kJ -/
def heat_released : ℝ := 609.9

/-- Represents the heat of combustion of diborane in kJ/mol -/
def diborane_combustion_heat : ℝ := -2165

/-- Theorem stating that the given heat of combustion of diborane is correct -/
theorem diborane_combustion_heat_correct : 
  diborane_combustion_heat = -heat_released / diborane_amount - 3 * water_vaporization_heat :=
sorry

end NUMINAMATH_CALUDE_diborane_combustion_heat_correct_l106_10622


namespace NUMINAMATH_CALUDE_johns_shower_duration_l106_10676

/-- Proves that John's shower duration is 10 minutes given the conditions --/
theorem johns_shower_duration :
  let days_in_four_weeks : ℕ := 28
  let shower_frequency : ℕ := 2  -- every other day
  let water_usage_per_minute : ℚ := 2  -- gallons per minute
  let total_water_usage : ℚ := 280  -- gallons in 4 weeks
  
  let num_showers : ℕ := days_in_four_weeks / shower_frequency
  let water_per_shower : ℚ := total_water_usage / num_showers
  let shower_duration : ℚ := water_per_shower / water_usage_per_minute
  
  shower_duration = 10 := by sorry

end NUMINAMATH_CALUDE_johns_shower_duration_l106_10676


namespace NUMINAMATH_CALUDE_simplify_expression_l106_10659

theorem simplify_expression (y : ℝ) :
  3 * y + 9 * y^2 - 15 - (5 - 3 * y - 9 * y^2) = 18 * y^2 + 6 * y - 20 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l106_10659


namespace NUMINAMATH_CALUDE_megan_markers_count_l106_10608

/-- The total number of markers Megan has after receiving more from Robert -/
def total_markers (initial : ℕ) (received : ℕ) : ℕ :=
  initial + received

/-- Theorem stating that Megan's total markers is the sum of her initial markers and those received from Robert -/
theorem megan_markers_count (initial : ℕ) (received : ℕ) :
  total_markers initial received = initial + received :=
by
  sorry

end NUMINAMATH_CALUDE_megan_markers_count_l106_10608


namespace NUMINAMATH_CALUDE_mango_rice_flour_cost_l106_10689

/-- Given the cost relationships between mangos, rice, and flour, 
    prove that the total cost of 4 kg of mangos, 3 kg of rice, and 5 kg of flour is $1027.2 -/
theorem mango_rice_flour_cost 
  (mango_cost rice_cost flour_cost : ℝ) 
  (h1 : 10 * mango_cost = 24 * rice_cost) 
  (h2 : 6 * flour_cost = 2 * rice_cost) 
  (h3 : flour_cost = 24) : 
  4 * mango_cost + 3 * rice_cost + 5 * flour_cost = 1027.2 := by
sorry

end NUMINAMATH_CALUDE_mango_rice_flour_cost_l106_10689


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l106_10628

/-- A line passing through point (2,-1) and perpendicular to x+y-3=0 has the equation x-y-3=0 -/
theorem perpendicular_line_equation :
  let point : ℝ × ℝ := (2, -1)
  let perpendicular_to : ℝ → ℝ → ℝ := fun x y => x + y - 3
  let line_equation : ℝ → ℝ → ℝ := fun x y => x - y - 3
  (∀ x y, perpendicular_to x y = 0 → (line_equation x y = 0 ↔ 
    (x - point.1) * 1 = (y - point.2) * 1 ∧
    1 * (-1) = -1)) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l106_10628


namespace NUMINAMATH_CALUDE_inverse_proposition_l106_10687

theorem inverse_proposition (a b : ℝ) : 
  (a = 0 → a * b = 0) ↔ (a * b = 0 → a = 0) :=
sorry

end NUMINAMATH_CALUDE_inverse_proposition_l106_10687


namespace NUMINAMATH_CALUDE_brown_utility_bill_l106_10616

/-- The value of the other bills Mrs. Brown used to pay her utility bill -/
def other_bills_value (total_bill : ℕ) (ten_dollar_bills : ℕ) : ℕ :=
  total_bill - (ten_dollar_bills * 10)

/-- Proof that the value of the other bills Mrs. Brown used is $150 -/
theorem brown_utility_bill :
  other_bills_value 170 2 = 150 := by
  sorry

end NUMINAMATH_CALUDE_brown_utility_bill_l106_10616


namespace NUMINAMATH_CALUDE_jersey_profit_calculation_l106_10679

/-- The amount the shop makes off each jersey -/
def jersey_profit : ℝ := 185.85

/-- The amount the shop makes off each t-shirt -/
def tshirt_profit : ℝ := 240

/-- The number of t-shirts sold -/
def tshirts_sold : ℕ := 177

/-- The number of jerseys sold -/
def jerseys_sold : ℕ := 23

/-- The difference in cost between a t-shirt and a jersey -/
def tshirt_jersey_diff : ℝ := 30

theorem jersey_profit_calculation :
  jersey_profit = (tshirts_sold * tshirt_profit) / (tshirts_sold + jerseys_sold) - tshirt_jersey_diff :=
by sorry

end NUMINAMATH_CALUDE_jersey_profit_calculation_l106_10679


namespace NUMINAMATH_CALUDE_product_without_x2_x3_implies_p_plus_q_eq_neg_four_l106_10678

theorem product_without_x2_x3_implies_p_plus_q_eq_neg_four (p q : ℝ) :
  (∀ x : ℝ, (x^2 + p) * (x^2 - q*x + 4) = x^4 + (-p*q)*x + 4*p) →
  p + q = -4 := by
  sorry

end NUMINAMATH_CALUDE_product_without_x2_x3_implies_p_plus_q_eq_neg_four_l106_10678


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l106_10649

theorem sum_of_three_numbers : 6 + 8 + 11 = 25 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l106_10649


namespace NUMINAMATH_CALUDE_farmer_price_l106_10680

def potato_problem (x : ℝ) : Prop :=
  let andrey_revenue := 60 * (2 * x)
  let boris_revenue := 15 * (1.6 * x) + 45 * (2.24 * x)
  boris_revenue - andrey_revenue = 1200

theorem farmer_price : ∃ x : ℝ, potato_problem x ∧ x = 250 := by
  sorry

end NUMINAMATH_CALUDE_farmer_price_l106_10680


namespace NUMINAMATH_CALUDE_sally_cracker_sales_l106_10681

theorem sally_cracker_sales (saturday_sales : ℕ) (sunday_increase_percent : ℕ) : 
  saturday_sales = 60 → 
  sunday_increase_percent = 50 → 
  saturday_sales + (saturday_sales + sunday_increase_percent * saturday_sales / 100) = 150 := by
sorry

end NUMINAMATH_CALUDE_sally_cracker_sales_l106_10681


namespace NUMINAMATH_CALUDE_problem_solution_l106_10695

theorem problem_solution (x y : ℝ) 
  (h1 : x = 151)
  (h2 : x^3 * y - 4 * x^2 * y + 4 * x * y = 342200) : 
  y = 342200 / 3354151 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l106_10695


namespace NUMINAMATH_CALUDE_horner_method_multiplications_l106_10623

/-- Horner's method for polynomial evaluation -/
def horner_eval (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- Number of multiplications in Horner's method -/
def horner_multiplications (coeffs : List ℝ) : ℕ :=
  coeffs.length - 1

/-- The polynomial f(x) = 3x^4 + 3x^3 + 2x^2 + 6x + 1 -/
def f_coeffs : List ℝ := [3, 3, 2, 6, 1]

theorem horner_method_multiplications :
  horner_multiplications f_coeffs = 4 :=
by
  sorry

#eval horner_eval f_coeffs 0.5
#eval horner_multiplications f_coeffs

end NUMINAMATH_CALUDE_horner_method_multiplications_l106_10623


namespace NUMINAMATH_CALUDE_valid_division_exists_l106_10613

/-- Represents a grid cell that can contain a symbol -/
inductive Cell
  | Empty
  | Star
  | Cross

/-- Represents a 7x7 grid -/
def Grid := Fin 7 → Fin 7 → Cell

/-- Represents a matchstick placement -/
structure Matchstick where
  row : Fin 8
  col : Fin 8
  horizontal : Bool

/-- Counts the number of matchsticks in a list -/
def count_matchsticks (placements : List Matchstick) : Nat :=
  placements.length

/-- Checks if two parts of the grid are of equal size and shape -/
def equal_parts (g : Grid) (placements : List Matchstick) : Prop :=
  sorry

/-- Checks if the symbols (stars and crosses) are placed correctly -/
def correct_symbol_placement (g : Grid) : Prop :=
  sorry

/-- The main theorem stating that a valid division exists -/
theorem valid_division_exists : ∃ (g : Grid) (placements : List Matchstick),
  count_matchsticks placements = 26 ∧
  equal_parts g placements ∧
  correct_symbol_placement g :=
  sorry

end NUMINAMATH_CALUDE_valid_division_exists_l106_10613


namespace NUMINAMATH_CALUDE_production_time_is_13_hours_l106_10634

/-- The time needed to complete the remaining production task -/
def time_to_complete (total_parts : ℕ) (apprentice_rate : ℕ) (master_rate : ℕ) (parts_done : ℕ) : ℚ :=
  (total_parts - parts_done) / (apprentice_rate + master_rate)

/-- Proof that the time to complete the production task is 13 hours -/
theorem production_time_is_13_hours :
  let total_parts : ℕ := 500
  let apprentice_rate : ℕ := 15
  let master_rate : ℕ := 20
  let parts_done : ℕ := 45
  time_to_complete total_parts apprentice_rate master_rate parts_done = 13 := by
  sorry

#eval time_to_complete 500 15 20 45

end NUMINAMATH_CALUDE_production_time_is_13_hours_l106_10634


namespace NUMINAMATH_CALUDE_inequality_equivalence_l106_10606

theorem inequality_equivalence (x : ℝ) : x * (x^2 + 1) > (x + 1) * (x^2 - x + 1) ↔ x > 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l106_10606


namespace NUMINAMATH_CALUDE_no_odd_cube_ending_668_l106_10684

theorem no_odd_cube_ending_668 : ¬∃ (n : ℕ), 
  Odd n ∧ n > 0 ∧ n^3 % 1000 = 668 := by
  sorry

end NUMINAMATH_CALUDE_no_odd_cube_ending_668_l106_10684


namespace NUMINAMATH_CALUDE_function_equivalence_l106_10698

open Real

theorem function_equivalence (x : ℝ) :
  2 * (cos x)^2 - Real.sqrt 3 * sin (2 * x) = 2 * sin (2 * (x + 5 * π / 12)) + 1 := by
  sorry

end NUMINAMATH_CALUDE_function_equivalence_l106_10698


namespace NUMINAMATH_CALUDE_lara_overtakes_darla_l106_10617

/-- The length of the circular track in meters -/
def track_length : ℝ := 500

/-- The speed ratio of Lara to Darla -/
def speed_ratio : ℝ := 1.2

/-- The number of laps completed by Lara when she first overtakes Darla -/
def laps_completed : ℝ := 6

/-- Theorem stating that Lara completes 6 laps when she first overtakes Darla -/
theorem lara_overtakes_darla :
  ∃ (t : ℝ), t > 0 ∧ speed_ratio * t * track_length = t * track_length + track_length ∧
  laps_completed = speed_ratio * t * track_length / track_length :=
sorry

end NUMINAMATH_CALUDE_lara_overtakes_darla_l106_10617


namespace NUMINAMATH_CALUDE_milk_measurement_problem_l106_10682

/-- Represents a container for milk -/
structure Container :=
  (capacity : ℕ)
  (content : ℕ)

/-- Represents the state of all containers -/
structure State :=
  (can1 : Container)
  (can2 : Container)
  (jug5 : Container)
  (jug4 : Container)

/-- Represents a pouring operation -/
inductive Operation
  | CanToJug : Container → Container → Operation
  | JugToJug : Container → Container → Operation
  | JugToCan : Container → Container → Operation

/-- The result of applying an operation to a state -/
def applyOperation (s : State) (op : Operation) : State :=
  sorry

/-- The initial state with two full 80-liter cans and empty jugs -/
def initialState : State :=
  { can1 := ⟨80, 80⟩
  , can2 := ⟨80, 80⟩
  , jug5 := ⟨5, 0⟩
  , jug4 := ⟨4, 0⟩ }

/-- The goal state with exactly 2 liters in each jug -/
def goalState : State :=
  { can1 := ⟨80, 80⟩
  , can2 := ⟨80, 76⟩
  , jug5 := ⟨5, 2⟩
  , jug4 := ⟨4, 2⟩ }

/-- Theorem stating that the goal state can be reached in exactly 9 operations -/
theorem milk_measurement_problem :
  ∃ (ops : List Operation),
    ops.length = 9 ∧
    (ops.foldl applyOperation initialState) = goalState :=
  sorry

end NUMINAMATH_CALUDE_milk_measurement_problem_l106_10682


namespace NUMINAMATH_CALUDE_min_value_problem_l106_10600

theorem min_value_problem (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (h_sum : x + 2*y = 1) :
  ∃ (m : ℝ), m = 3/4 ∧ ∀ (x' y' : ℝ), x' ≥ 0 → y' ≥ 0 → x' + 2*y' = 1 → 2*x' + 3*y'^2 ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_problem_l106_10600


namespace NUMINAMATH_CALUDE_money_distribution_l106_10603

theorem money_distribution (a b c : ℕ) (total : ℕ) : 
  a + b + c = 9 → 
  b = 3 → 
  900 * b = 2700 → 
  900 * (a + b + c) = 2700 * 3 :=
by sorry

end NUMINAMATH_CALUDE_money_distribution_l106_10603


namespace NUMINAMATH_CALUDE_cos_x_plus_2y_equals_one_l106_10638

theorem cos_x_plus_2y_equals_one 
  (x y : ℝ) 
  (a : ℝ) 
  (h1 : x ∈ Set.Icc (-π/4) (π/4)) 
  (h2 : y ∈ Set.Icc (-π/4) (π/4)) 
  (h3 : x^3 + Real.sin x - 2*a = 0) 
  (h4 : 4*y^3 + Real.sin y * Real.cos y + a = 0) : 
  Real.cos (x + 2*y) = 1 := by
sorry

end NUMINAMATH_CALUDE_cos_x_plus_2y_equals_one_l106_10638


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l106_10665

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_third_term
  (a : ℕ → ℝ)
  (h_geom : is_geometric_sequence a)
  (h_roots : a 1 * a 5 = 9 ∧ a 1 + a 5 = 12) :
  a 3 = 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_third_term_l106_10665


namespace NUMINAMATH_CALUDE_marilyn_shared_bottle_caps_l106_10688

/-- The number of bottle caps Marilyn shared with Nancy -/
def shared_bottle_caps (initial : ℕ) (remaining : ℕ) : ℕ :=
  initial - remaining

theorem marilyn_shared_bottle_caps :
  shared_bottle_caps 51 15 = 36 :=
by sorry

end NUMINAMATH_CALUDE_marilyn_shared_bottle_caps_l106_10688


namespace NUMINAMATH_CALUDE_cosine_sum_theorem_l106_10675

theorem cosine_sum_theorem : 
  12 * (Real.cos (π / 8)) ^ 4 + 
  (Real.cos (3 * π / 8)) ^ 4 + 
  (Real.cos (5 * π / 8)) ^ 4 + 
  (Real.cos (7 * π / 8)) ^ 4 = 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_cosine_sum_theorem_l106_10675


namespace NUMINAMATH_CALUDE_not_all_greater_than_quarter_l106_10662

theorem not_all_greater_than_quarter (a b c : ℝ) 
  (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) (hc : 0 < c ∧ c < 1) : 
  ¬(((1 - a) * b > 1/4) ∧ ((1 - b) * c > 1/4) ∧ ((1 - c) * a > 1/4)) := by
  sorry

end NUMINAMATH_CALUDE_not_all_greater_than_quarter_l106_10662


namespace NUMINAMATH_CALUDE_sum_abc_equals_51_l106_10626

theorem sum_abc_equals_51 (a b c : ℕ+) 
  (h1 : a * b + c = 50)
  (h2 : a * c + b = 50)
  (h3 : b * c + a = 50) : 
  a + b + c = 51 := by
  sorry

end NUMINAMATH_CALUDE_sum_abc_equals_51_l106_10626


namespace NUMINAMATH_CALUDE_simplest_square_root_l106_10629

/-- Given real numbers a and b, with a ≠ 0, prove that √(a^2 + b^2) is the simplest form among:
    √(16a), √(a^2 + b^2), √(b/a), and √45 -/
theorem simplest_square_root (a b : ℝ) (ha : a ≠ 0) :
  ∃ (f : ℝ → ℝ), f (Real.sqrt (a^2 + b^2)) = Real.sqrt (a^2 + b^2) ∧
    (∀ g : ℝ → ℝ, g (Real.sqrt (16*a)) ≠ Real.sqrt (16*a) ∨
                   g (Real.sqrt (b/a)) ≠ Real.sqrt (b/a) ∨
                   g (Real.sqrt 45) ≠ Real.sqrt 45 ∨
                   g = f) :=
by sorry

end NUMINAMATH_CALUDE_simplest_square_root_l106_10629


namespace NUMINAMATH_CALUDE_slope_of_AA_l106_10641

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a triangle in 2D space
structure Triangle where
  A : Point2D
  B : Point2D
  C : Point2D

-- Define the transformation (shift right by 2 and reflect across y=x)
def transform (p : Point2D) : Point2D :=
  { x := p.y, y := p.x + 2 }

-- Theorem statement
theorem slope_of_AA'_is_one (t : Triangle)
  (h1 : t.A.x ≥ 0 ∧ t.A.y ≥ 0)  -- A is in first quadrant
  (h2 : t.B.x ≥ 0 ∧ t.B.y ≥ 0)  -- B is in first quadrant
  (h3 : t.C.x ≥ 0 ∧ t.C.y ≥ 0)  -- C is in first quadrant
  (h4 : t.A.x + 2 ≥ 0 ∧ t.A.y ≥ 0)  -- A+2 is in first quadrant
  (h5 : t.B.x + 2 ≥ 0 ∧ t.B.y ≥ 0)  -- B+2 is in first quadrant
  (h6 : t.C.x + 2 ≥ 0 ∧ t.C.y ≥ 0)  -- C+2 is in first quadrant
  (h7 : t.A.x ≠ t.A.y)  -- A not on y=x
  (h8 : t.B.x ≠ t.B.y)  -- B not on y=x
  (h9 : t.C.x ≠ t.C.y)  -- C not on y=x
  : (transform t.A).y - t.A.y = (transform t.A).x - t.A.x :=
by sorry

end NUMINAMATH_CALUDE_slope_of_AA_l106_10641


namespace NUMINAMATH_CALUDE_find_divisor_l106_10627

theorem find_divisor (N D : ℕ) (h1 : N = D * 8) (h2 : N % 5 = 4) : D = 3 := by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l106_10627


namespace NUMINAMATH_CALUDE_original_average_l106_10640

theorem original_average (n : ℕ) (A : ℝ) (h1 : n = 7) (h2 : (5 * n * A) / n = 100) : A = 20 := by
  sorry

end NUMINAMATH_CALUDE_original_average_l106_10640


namespace NUMINAMATH_CALUDE_complete_square_quadratic_l106_10602

theorem complete_square_quadratic (x : ℝ) : 
  ∃ (c d : ℝ), x^2 + 6*x - 4 = 0 ↔ (x + c)^2 = d ∧ d = 13 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_quadratic_l106_10602


namespace NUMINAMATH_CALUDE_total_basketballs_l106_10604

/-- Calculates the total number of basketballs for three basketball teams -/
theorem total_basketballs (spurs_players spurs_balls dynamos_players dynamos_balls lions_players lions_balls : ℕ) :
  spurs_players = 22 →
  spurs_balls = 11 →
  dynamos_players = 18 →
  dynamos_balls = 9 →
  lions_players = 26 →
  lions_balls = 7 →
  spurs_players * spurs_balls + dynamos_players * dynamos_balls + lions_players * lions_balls = 586 :=
by
  sorry

#check total_basketballs

end NUMINAMATH_CALUDE_total_basketballs_l106_10604


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l106_10643

theorem inequality_system_solution_set :
  let S := {x : ℝ | x + 8 < 4*x - 1 ∧ (1/2)*x ≥ 4 - (3/2)*x}
  S = {x : ℝ | x > 3} :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l106_10643


namespace NUMINAMATH_CALUDE_milk_container_problem_l106_10615

/-- The initial quantity of milk in container A -/
def initial_quantity : ℝ := 1248

/-- The quantity of milk transferred from C to B -/
def transfer_amount : ℝ := 156

/-- The fraction of container A's capacity that is in container B after pouring -/
def fraction_in_b : ℝ := 0.375

theorem milk_container_problem :
  -- Container A was filled to its brim
  -- All milk from A was poured into B and C
  -- Quantity in B is 62.5% less than capacity of A (which is equivalent to 37.5% of A)
  -- If 156 liters is transferred from C to B, both containers would have equal quantities
  initial_quantity * fraction_in_b + transfer_amount = 
  initial_quantity * (1 - fraction_in_b) - transfer_amount :=
by sorry

end NUMINAMATH_CALUDE_milk_container_problem_l106_10615


namespace NUMINAMATH_CALUDE_two_in_A_l106_10661

def A : Set ℝ := {x | x^2 - 4 = 0}

theorem two_in_A : 2 ∈ A := by sorry

end NUMINAMATH_CALUDE_two_in_A_l106_10661


namespace NUMINAMATH_CALUDE_gcf_factorial_seven_eight_l106_10619

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem gcf_factorial_seven_eight : 
  Nat.gcd (factorial 7) (factorial 8) = factorial 7 := by
  sorry

end NUMINAMATH_CALUDE_gcf_factorial_seven_eight_l106_10619


namespace NUMINAMATH_CALUDE_sum_of_exponents_product_divisors_360000_l106_10670

/-- The product of all positive integer divisors of a natural number n -/
def product_of_divisors (n : ℕ) : ℕ := sorry

/-- The sum of exponents in the prime factorization of a natural number n -/
def sum_of_exponents (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of exponents in the prime factorization of the product of all positive integer divisors of 360000 is 630 -/
theorem sum_of_exponents_product_divisors_360000 :
  sum_of_exponents (product_of_divisors 360000) = 630 := by sorry

end NUMINAMATH_CALUDE_sum_of_exponents_product_divisors_360000_l106_10670


namespace NUMINAMATH_CALUDE_savings_percentage_second_year_l106_10612

/-- Proves that under the given conditions, the savings percentage in the second year is 15% -/
theorem savings_percentage_second_year 
  (salary_first_year : ℝ) 
  (savings_rate_first_year : ℝ) 
  (salary_increase_rate : ℝ) 
  (savings_increase_rate : ℝ) : 
  savings_rate_first_year = 0.1 →
  salary_increase_rate = 0.1 →
  savings_increase_rate = 1.65 →
  (savings_increase_rate * savings_rate_first_year * salary_first_year) / 
  ((1 + salary_increase_rate) * salary_first_year) = 0.15 := by
sorry


end NUMINAMATH_CALUDE_savings_percentage_second_year_l106_10612


namespace NUMINAMATH_CALUDE_raft_problem_l106_10696

/-- The number of people who can fit on a raft under specific conditions -/
def raft_capacity (capacity_without_jackets : ℕ) (capacity_reduction : ℕ) (people_needing_jackets : ℕ) : ℕ :=
  let capacity_with_jackets := capacity_without_jackets - capacity_reduction
  min capacity_with_jackets (people_needing_jackets + (capacity_with_jackets - people_needing_jackets))

/-- Theorem stating that under the given conditions, 14 people can fit on the raft -/
theorem raft_problem : raft_capacity 21 7 8 = 14 := by
  sorry

end NUMINAMATH_CALUDE_raft_problem_l106_10696


namespace NUMINAMATH_CALUDE_oil_volume_in_liters_l106_10642

def bottle_volume : ℝ := 200
def num_bottles : ℕ := 20
def ml_per_liter : ℝ := 1000

theorem oil_volume_in_liters :
  (bottle_volume * num_bottles) / ml_per_liter = 4 := by
  sorry

end NUMINAMATH_CALUDE_oil_volume_in_liters_l106_10642


namespace NUMINAMATH_CALUDE_age_puzzle_solution_l106_10663

/-- Represents a person's age --/
structure Age :=
  (tens : Nat)
  (ones : Nat)
  (is_valid : tens ≤ 9 ∧ ones ≤ 9)

/-- The age after 10 years --/
def age_after_10_years (a : Age) : Nat :=
  10 * a.tens + a.ones + 10

/-- Helen's age is the reverse of Ellen's age --/
def is_reverse (helen : Age) (ellen : Age) : Prop :=
  helen.tens = ellen.ones ∧ helen.ones = ellen.tens

/-- In 10 years, Helen will be three times as old as Ellen --/
def future_age_relation (helen : Age) (ellen : Age) : Prop :=
  age_after_10_years helen = 3 * age_after_10_years ellen

/-- The current age difference --/
def age_difference (helen : Age) (ellen : Age) : Int :=
  (10 * helen.tens + helen.ones) - (10 * ellen.tens + ellen.ones)

theorem age_puzzle_solution :
  ∃ (helen ellen : Age),
    is_reverse helen ellen ∧
    future_age_relation helen ellen ∧
    age_difference helen ellen = 54 :=
  sorry

end NUMINAMATH_CALUDE_age_puzzle_solution_l106_10663


namespace NUMINAMATH_CALUDE_p_adic_valuation_properties_l106_10648

/-- The p-adic valuation of an integer -/
noncomputable def v_p (p : ℕ) (n : ℤ) : ℕ := sorry

/-- Properties of p-adic valuation for prime p and integers m, n -/
theorem p_adic_valuation_properties (p : ℕ) (m n : ℤ) (hp : Nat.Prime p) :
  (v_p p (m * n) = v_p p m + v_p p n) ∧
  (v_p p (m + n) ≥ min (v_p p m) (v_p p n)) ∧
  (v_p p (Int.gcd m n) = min (v_p p m) (v_p p n)) ∧
  (v_p p (Int.lcm m n) = max (v_p p m) (v_p p n)) :=
by sorry

end NUMINAMATH_CALUDE_p_adic_valuation_properties_l106_10648


namespace NUMINAMATH_CALUDE_angus_tokens_l106_10674

def token_value : ℕ := 4
def elsa_tokens : ℕ := 60
def token_difference : ℕ := 20

theorem angus_tokens : 
  ∃ (angus_tokens : ℕ), 
    angus_tokens * token_value = elsa_tokens * token_value - token_difference ∧ 
    angus_tokens = 55 :=
by sorry

end NUMINAMATH_CALUDE_angus_tokens_l106_10674


namespace NUMINAMATH_CALUDE_stream_speed_l106_10671

/-- Proves that given a boat trip where a man rows 72 km downstream and 30 km upstream,
    each taking 3 hours, the speed of the stream is 7 km/h. -/
theorem stream_speed (downstream_distance : ℝ) (upstream_distance : ℝ) (time : ℝ)
  (h1 : downstream_distance = 72)
  (h2 : upstream_distance = 30)
  (h3 : time = 3) :
  ∃ (boat_speed stream_speed : ℝ),
    downstream_distance = (boat_speed + stream_speed) * time ∧
    upstream_distance = (boat_speed - stream_speed) * time ∧
    stream_speed = 7 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_l106_10671


namespace NUMINAMATH_CALUDE_total_chocolate_bars_l106_10614

/-- The number of chocolate bars in a massive crate -/
def chocolateBarsInCrate (largeBozesPerCrate mediumBoxesPerLarge smallBoxesPerMedium barsPerSmall : ℕ) : ℕ :=
  largeBozesPerCrate * mediumBoxesPerLarge * smallBoxesPerMedium * barsPerSmall

/-- Theorem: The massive crate contains 153,900 chocolate bars -/
theorem total_chocolate_bars :
  chocolateBarsInCrate 10 19 27 30 = 153900 := by
  sorry

#eval chocolateBarsInCrate 10 19 27 30

end NUMINAMATH_CALUDE_total_chocolate_bars_l106_10614


namespace NUMINAMATH_CALUDE_right_triangle_integer_sides_l106_10673

theorem right_triangle_integer_sides (a b c : ℕ) : 
  a^2 + b^2 = c^2 → -- Pythagorean theorem (right-angled triangle)
  Nat.gcd a (Nat.gcd b c) = 1 → -- GCD of sides is 1
  ∃ m n : ℕ, 
    (a = 2*m*n ∧ b = m^2 - n^2 ∧ c = m^2 + n^2) ∨ 
    (b = 2*m*n ∧ a = m^2 - n^2 ∧ c = m^2 + n^2) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_integer_sides_l106_10673


namespace NUMINAMATH_CALUDE_fifth_friend_payment_l106_10657

def boat_purchase (a b c d e : ℝ) : Prop :=
  a + b + c + d + e = 120 ∧
  a = (1/3) * (b + c + d + e) ∧
  b = (1/4) * (a + c + d + e) ∧
  c = (1/5) * (a + b + d + e)

theorem fifth_friend_payment :
  ∃ a b c d : ℝ, boat_purchase a b c d 13 :=
sorry

end NUMINAMATH_CALUDE_fifth_friend_payment_l106_10657


namespace NUMINAMATH_CALUDE_min_mn_value_l106_10685

theorem min_mn_value (m : ℝ) (n : ℝ) (h_m : m > 0) 
  (h_ineq : ∀ x : ℝ, x > -m → x + m ≤ Real.exp ((2 * x / m) + n)) :
  ∃ (min_mn : ℝ), min_mn = -2 / Real.exp 2 ∧ 
    ∀ (m' n' : ℝ), (∀ x : ℝ, x > -m' → x + m' ≤ Real.exp ((2 * x / m') + n')) → 
      m' * n' ≥ min_mn :=
sorry

end NUMINAMATH_CALUDE_min_mn_value_l106_10685


namespace NUMINAMATH_CALUDE_winnie_lollipops_l106_10633

theorem winnie_lollipops (cherry wintergreen grape shrimp : ℕ) (friends : ℕ) : 
  cherry = 45 → wintergreen = 116 → grape = 4 → shrimp = 229 → friends = 11 →
  (cherry + wintergreen + grape + shrimp) % friends = 9 := by
  sorry

end NUMINAMATH_CALUDE_winnie_lollipops_l106_10633


namespace NUMINAMATH_CALUDE_probability_selection_l106_10655

def research_team (total : ℝ) : Prop :=
  let women := 0.75 * total
  let men := 0.25 * total
  let women_lawyers := 0.60 * women
  let women_engineers := 0.25 * women
  let women_doctors := 0.15 * women
  let men_lawyers := 0.40 * men
  let men_engineers := 0.35 * men
  let men_doctors := 0.25 * men
  (women + men = total) ∧
  (women_lawyers + women_engineers + women_doctors = women) ∧
  (men_lawyers + men_engineers + men_doctors = men)

theorem probability_selection (total : ℝ) (h : research_team total) :
  (0.75 * 0.60 * total + 0.75 * 0.25 * total + 0.25 * 0.25 * total) / total = 0.70 :=
by sorry

end NUMINAMATH_CALUDE_probability_selection_l106_10655


namespace NUMINAMATH_CALUDE_y₁_equals_y₂_at_half_y₁_greater_than_2y₂_by_5_at_negative_2_l106_10672

-- Define the functions y₁ and y₂
def y₁ (x : ℝ) : ℝ := -x + 3
def y₂ (x : ℝ) : ℝ := 2 + x

-- Theorem 1: y₁ = y₂ when x = 1/2
theorem y₁_equals_y₂_at_half : y₁ (1/2) = y₂ (1/2) := by sorry

-- Theorem 2: y₁ = 2y₂ + 5 when x = -2
theorem y₁_greater_than_2y₂_by_5_at_negative_2 : y₁ (-2) = 2 * y₂ (-2) + 5 := by sorry

end NUMINAMATH_CALUDE_y₁_equals_y₂_at_half_y₁_greater_than_2y₂_by_5_at_negative_2_l106_10672


namespace NUMINAMATH_CALUDE_only_proposition_3_true_l106_10636

theorem only_proposition_3_true : 
  (¬∀ (a b c : ℝ), a > b ∧ c ≠ 0 → a * c > b * c) ∧ 
  (¬∀ (a b c : ℝ), a > b → a * c^2 > b * c^2) ∧ 
  (∀ (a b c : ℝ), a * c^2 > b * c^2 → a > b) ∧ 
  (¬∀ (a b : ℝ), a > b → 1 / a < 1 / b) ∧ 
  (¬∀ (a b c d : ℝ), a > b ∧ b > 0 ∧ c > d → a * c > b * d) :=
by sorry

end NUMINAMATH_CALUDE_only_proposition_3_true_l106_10636


namespace NUMINAMATH_CALUDE_largest_inscribed_circle_radius_for_specific_quad_l106_10654

/-- A quadrilateral with given side lengths -/
structure Quadrilateral where
  ab : ℝ
  bc : ℝ
  cd : ℝ
  da : ℝ

/-- The radius of the largest inscribed circle in a quadrilateral -/
def largest_inscribed_circle_radius (q : Quadrilateral) : ℝ :=
  sorry

/-- Theorem stating the largest inscribed circle radius for a specific quadrilateral -/
theorem largest_inscribed_circle_radius_for_specific_quad :
  let q : Quadrilateral := ⟨15, 10, 8, 13⟩
  largest_inscribed_circle_radius q = 5.7 := by
  sorry

end NUMINAMATH_CALUDE_largest_inscribed_circle_radius_for_specific_quad_l106_10654


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_l106_10618

theorem necessary_not_sufficient (a b c : ℝ) :
  (∀ a b c : ℝ, a * c^2 > b * c^2 → a > b ∨ c = 0) ∧
  (∃ a b c : ℝ, a * c^2 > b * c^2 ∧ a ≤ b) :=
sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_l106_10618


namespace NUMINAMATH_CALUDE_rhombus_area_l106_10644

theorem rhombus_area (d₁ d₂ : ℝ) : 
  d₁^2 - 14*d₁ + 48 = 0 → 
  d₂^2 - 14*d₂ + 48 = 0 → 
  d₁ ≠ d₂ →
  (d₁ * d₂) / 2 = 24 := by
sorry

end NUMINAMATH_CALUDE_rhombus_area_l106_10644


namespace NUMINAMATH_CALUDE_course_length_l106_10666

/-- Represents the time taken by Team B to complete the course -/
def team_b_time : ℝ := 15

/-- Represents the speed of Team B in miles per hour -/
def team_b_speed : ℝ := 20

/-- Represents the difference in completion time between Team A and Team B -/
def time_difference : ℝ := 3

/-- Represents the difference in speed between Team A and Team B -/
def speed_difference : ℝ := 5

/-- Theorem stating that the course length is 300 miles -/
theorem course_length : 
  team_b_speed * team_b_time = 300 :=
sorry

end NUMINAMATH_CALUDE_course_length_l106_10666


namespace NUMINAMATH_CALUDE_boatman_journey_l106_10620

/-- Represents the boatman's journey on the river -/
structure RiverJourney where
  v : ℝ  -- Speed of the boat in still water
  v_T : ℝ  -- Speed of the current
  upstream_distance : ℝ  -- Distance traveled upstream
  total_time : ℝ  -- Total time for the round trip

/-- Theorem stating the conditions and results of the boatman's journey -/
theorem boatman_journey (j : RiverJourney) : 
  j.upstream_distance = 12.5 ∧ 
  (3 / (j.v - j.v_T) = 5 / (j.v + j.v_T)) ∧ 
  (j.upstream_distance / (j.v - j.v_T) + j.upstream_distance / (j.v + j.v_T) = j.total_time) ∧ 
  j.total_time = 8 → 
  j.v_T = 5/6 ∧ 
  j.upstream_distance / (j.v - j.v_T) = 5 := by
  sorry

end NUMINAMATH_CALUDE_boatman_journey_l106_10620


namespace NUMINAMATH_CALUDE_sqrt_expression_equality_l106_10601

theorem sqrt_expression_equality : 
  Real.sqrt 48 / Real.sqrt 3 - Real.sqrt (1/2) * Real.sqrt 12 + Real.sqrt 24 = 4 + Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equality_l106_10601


namespace NUMINAMATH_CALUDE_cubic_function_sum_l106_10677

-- Define the function f
def f (a b c x : ℤ) : ℝ := x^3 + a*x^2 + b*x + c

-- State the theorem
theorem cubic_function_sum (a b c : ℤ) : 
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧   -- a, b, c are non-zero
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧   -- a, b, c are distinct
  f a b c a = a^3 ∧         -- f(a) = a^3
  f a b c b = b^3           -- f(b) = b^3
  → a + b + c = 18 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_sum_l106_10677
