import Mathlib

namespace solution_value_l1518_151852

theorem solution_value (x : ℝ) (h : 1 - 6/x + 9/x^2 = 0) : 3/x = 1 := by
  sorry

end solution_value_l1518_151852


namespace lisa_scenery_photos_l1518_151895

-- Define the variables
def animal_photos : ℕ := 10
def flower_photos : ℕ := 3 * animal_photos
def scenery_photos : ℕ := flower_photos - 10
def total_photos : ℕ := 45

-- Theorem to prove
theorem lisa_scenery_photos :
  scenery_photos = 20 ∧
  animal_photos + flower_photos + scenery_photos = total_photos :=
by sorry

end lisa_scenery_photos_l1518_151895


namespace normal_distribution_probability_bagged_rice_probability_l1518_151887

/-- The cumulative distribution function of the standard normal distribution -/
noncomputable def Φ : ℝ → ℝ := sorry

/-- The probability density function of the normal distribution with mean μ and variance σ² -/
noncomputable def normalPDF (μ σ : ℝ) (x : ℝ) : ℝ := sorry

theorem normal_distribution_probability (μ σ x₁ x₂ : ℝ) (hσ : σ > 0) :
  (∫ x in x₁..x₂, normalPDF μ σ x) = Φ ((x₂ - μ) / σ) - Φ ((x₁ - μ) / σ) :=
sorry

/-- The probability that a value from N(10, 0.01) is between 9.8 and 10.2 -/
theorem bagged_rice_probability :
  (∫ x in 9.8..10.2, normalPDF 10 0.1 x) = 2 * Φ 2 - 1 :=
sorry

end normal_distribution_probability_bagged_rice_probability_l1518_151887


namespace sqrt_three_irrational_l1518_151892

theorem sqrt_three_irrational : Irrational (Real.sqrt 3) := by
  sorry

end sqrt_three_irrational_l1518_151892


namespace johns_remaining_money_l1518_151813

/-- Calculates the remaining money after John's pizza and drink purchase. -/
def remaining_money (q : ℝ) : ℝ :=
  let drink_cost := q
  let small_pizza_cost := q
  let large_pizza_cost := 4 * q
  let total_spent := 2 * drink_cost + 2 * small_pizza_cost + large_pizza_cost
  50 - total_spent

/-- Proves that John's remaining money is equal to 50 - 8q. -/
theorem johns_remaining_money (q : ℝ) : remaining_money q = 50 - 8 * q := by
  sorry

end johns_remaining_money_l1518_151813


namespace line_through_circle_center_l1518_151805

/-- The equation of a line passing through the center of a given circle with a specific slope angle -/
theorem line_through_circle_center (x y : ℝ) : 
  (∀ x y, (x + 1)^2 + (y - 2)^2 = 4) →  -- Circle equation
  (∃ m b : ℝ, y = m * x + b ∧ m = 1) →  -- Line with slope 1 (45° angle)
  (∃ x₀ y₀ : ℝ, (x₀ + 1)^2 + (y₀ - 2)^2 = 4 ∧ y₀ = x₀ + 3) →  -- Line passes through circle center
  x - y + 3 = 0  -- Resulting line equation
:= by sorry

end line_through_circle_center_l1518_151805


namespace basketball_games_l1518_151850

theorem basketball_games (c : ℕ) : 
  (3 * c / 4 : ℚ) = (7 * c / 10 : ℚ) - 5 ∧ 
  (c / 4 : ℚ) = (3 * c / 10 : ℚ) - 5 → 
  c = 100 := by
  sorry

end basketball_games_l1518_151850


namespace path_area_is_675_l1518_151876

/-- Calculates the area of a path surrounding a rectangular field. -/
def path_area (field_length field_width path_width : ℝ) : ℝ :=
  (field_length + 2 * path_width) * (field_width + 2 * path_width) - field_length * field_width

/-- Theorem: The area of the path surrounding the given rectangular field is 675 sq m. -/
theorem path_area_is_675 (field_length field_width path_width cost_per_sqm total_cost : ℝ) :
  field_length = 75 →
  field_width = 55 →
  path_width = 2.5 →
  cost_per_sqm = 10 →
  total_cost = 6750 →
  path_area field_length field_width path_width = 675 :=
by
  sorry

#eval path_area 75 55 2.5

end path_area_is_675_l1518_151876


namespace cubic_roots_sum_squares_l1518_151847

theorem cubic_roots_sum_squares (p q r : ℝ) : 
  p^3 - 15*p^2 + 25*p - 12 = 0 →
  q^3 - 15*q^2 + 25*q - 12 = 0 →
  r^3 - 15*r^2 + 25*r - 12 = 0 →
  (p+q)^2 + (q+r)^2 + (r+p)^2 = 400 :=
by sorry

end cubic_roots_sum_squares_l1518_151847


namespace thursday_productivity_l1518_151821

/-- Represents the relationship between cups of coffee and lines of code written --/
structure CoffeeProductivity where
  k : ℝ  -- Proportionality constant
  coffee_to_code : ℝ → ℝ  -- Function that converts cups of coffee to lines of code

/-- Given the conditions from the problem, prove that the programmer wrote 250 lines of code on Thursday --/
theorem thursday_productivity (cp : CoffeeProductivity) 
  (h1 : cp.coffee_to_code 3 = 150)  -- Wednesday's data
  (h2 : ∀ c, cp.coffee_to_code c = cp.k * c)  -- Direct proportionality
  : cp.coffee_to_code 5 = 250 := by
  sorry

#check thursday_productivity

end thursday_productivity_l1518_151821


namespace even_sum_probability_l1518_151849

/-- The set of the first twenty prime numbers -/
def first_twenty_primes : Finset ℕ := sorry

/-- The number of ways to select 6 numbers from a set of 20 -/
def total_selections : ℕ := Nat.choose 20 6

/-- The number of ways to select 6 odd numbers from the set of odd primes in first_twenty_primes -/
def odd_selections : ℕ := Nat.choose 19 6

/-- The probability of selecting six prime numbers from first_twenty_primes such that their sum is even -/
def prob_even_sum : ℚ := odd_selections / total_selections

theorem even_sum_probability : prob_even_sum = 354 / 505 := by sorry

end even_sum_probability_l1518_151849


namespace path_area_calculation_l1518_151802

/-- Calculates the area of a path surrounding a rectangular field -/
def path_area (field_length field_width path_width : ℝ) : ℝ :=
  (field_length + 2 * path_width) * (field_width + 2 * path_width) - field_length * field_width

theorem path_area_calculation (field_length field_width path_width : ℝ)
  (h1 : field_length = 75)
  (h2 : field_width = 55)
  (h3 : path_width = 3.5) :
  path_area field_length field_width path_width = 959 := by
  sorry

#eval path_area 75 55 3.5

end path_area_calculation_l1518_151802


namespace sum_of_prime_factors_2310_l1518_151856

theorem sum_of_prime_factors_2310 : 
  (Finset.sum (Finset.filter Nat.Prime (Finset.range (2310 + 1))) id) = 28 := by
  sorry

end sum_of_prime_factors_2310_l1518_151856


namespace remaining_files_count_l1518_151820

def initial_music_files : ℕ := 4
def initial_video_files : ℕ := 21
def initial_document_files : ℕ := 12
def initial_photo_files : ℕ := 30
def initial_app_files : ℕ := 7

def deleted_video_files : ℕ := 15
def deleted_document_files : ℕ := 10
def deleted_photo_files : ℕ := 18
def deleted_app_files : ℕ := 3

theorem remaining_files_count :
  initial_music_files +
  (initial_video_files - deleted_video_files) +
  (initial_document_files - deleted_document_files) +
  (initial_photo_files - deleted_photo_files) +
  (initial_app_files - deleted_app_files) = 28 := by
  sorry

end remaining_files_count_l1518_151820


namespace selling_price_for_loss_is_40_l1518_151893

/-- The selling price that yields the same loss as the profit for an article -/
def selling_price_for_loss (cost_price : ℕ) (profit_selling_price : ℕ) : ℕ :=
  cost_price - (profit_selling_price - cost_price)

/-- Proof that the selling price for loss is 40 given the conditions -/
theorem selling_price_for_loss_is_40 :
  selling_price_for_loss 47 54 = 40 := by
  sorry

#eval selling_price_for_loss 47 54

end selling_price_for_loss_is_40_l1518_151893


namespace necessary_condition_for_false_proposition_l1518_151824

theorem necessary_condition_for_false_proposition (a : ℝ) :
  (¬ ∃ x₀ : ℝ, x₀^2 - a*x₀ + 1 ≤ 0) → (-2 ≤ a ∧ a ≤ 2) :=
by sorry

end necessary_condition_for_false_proposition_l1518_151824


namespace cube_root_negative_l1518_151871

theorem cube_root_negative (a : ℝ) (k : ℝ) (h : k^3 = a) : 
  ((-a : ℝ)^(1/3 : ℝ) : ℝ) = -k := by sorry

end cube_root_negative_l1518_151871


namespace number_value_proof_l1518_151870

theorem number_value_proof (x : ℝ) : (8^3 * x^3) / 679 = 549.7025036818851 ↔ x = 9 := by
  sorry

end number_value_proof_l1518_151870


namespace smallest_multiple_with_remainder_three_l1518_151872

theorem smallest_multiple_with_remainder_three : 
  (∀ n : ℕ, n > 1 ∧ n < 843 → 
    ¬(n % 4 = 3 ∧ n % 5 = 3 ∧ n % 6 = 3 ∧ n % 7 = 3 ∧ n % 8 = 3)) ∧ 
  (843 % 4 = 3 ∧ 843 % 5 = 3 ∧ 843 % 6 = 3 ∧ 843 % 7 = 3 ∧ 843 % 8 = 3) :=
by sorry

end smallest_multiple_with_remainder_three_l1518_151872


namespace radical_simplification_l1518_151844

theorem radical_simplification (p : ℝ) (hp : p > 0) :
  Real.sqrt (15 * p^3) * Real.sqrt (3 * p^5) * Real.sqrt (4 * p^2) / Real.sqrt (2 * p) = 6 * p^(9/2) * Real.sqrt (5/2) :=
by sorry

end radical_simplification_l1518_151844


namespace geometric_sequence_sum_l1518_151819

/-- A geometric sequence with negative terms -/
def NegativeGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n ∧ a n < 0

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  NegativeGeometricSequence a →
  a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 36 →
  a 3 + a 5 = -6 := by
  sorry

end geometric_sequence_sum_l1518_151819


namespace scalene_to_right_triangle_l1518_151810

theorem scalene_to_right_triangle 
  (a b c : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hc : 0 < c) 
  (hab : a ≠ b) 
  (hbc : b ≠ c) 
  (hac : a ≠ c) :
  ∃ x : ℝ, (a + x)^2 + (b + x)^2 = (c + x)^2 :=
sorry

end scalene_to_right_triangle_l1518_151810


namespace no_valid_coloring_l1518_151833

theorem no_valid_coloring : ¬∃ (f : ℕ+ → Bool), 
  (∀ n : ℕ+, f n ≠ f (n + 5)) ∧ 
  (∀ n : ℕ+, f n ≠ f (2 * n)) := by
  sorry

end no_valid_coloring_l1518_151833


namespace executive_board_selection_l1518_151868

theorem executive_board_selection (n : ℕ) (k : ℕ) (h1 : n = 12) (h2 : k = 5) :
  Nat.choose n k = 792 := by
  sorry

end executive_board_selection_l1518_151868


namespace rhombus_area_l1518_151857

/-- The area of a rhombus with vertices at (0, 3.5), (11, 0), (0, -3.5), and (-11, 0) is 77 square units. -/
theorem rhombus_area : 
  let vertices : List (ℝ × ℝ) := [(0, 3.5), (11, 0), (0, -3.5), (-11, 0)]
  let vertical_diagonal : ℝ := |3.5 - (-3.5)|
  let horizontal_diagonal : ℝ := |11 - (-11)|
  let area : ℝ := (vertical_diagonal * horizontal_diagonal) / 2
  area = 77 := by sorry

end rhombus_area_l1518_151857


namespace probability_all_black_is_correct_l1518_151823

def urn_black_balls : ℕ := 10
def urn_white_balls : ℕ := 5
def total_balls : ℕ := urn_black_balls + urn_white_balls
def drawn_balls : ℕ := 2

def probability_all_black : ℚ := (urn_black_balls.choose drawn_balls) / (total_balls.choose drawn_balls)

theorem probability_all_black_is_correct :
  probability_all_black = 3 / 7 :=
sorry

end probability_all_black_is_correct_l1518_151823


namespace complex_magnitude_one_l1518_151803

theorem complex_magnitude_one (z : ℂ) (r : ℝ) 
  (h1 : |r| < 2) 
  (h2 : z + z⁻¹ = r) : 
  Complex.abs z = 1 := by
  sorry

end complex_magnitude_one_l1518_151803


namespace cost_price_per_meter_correct_cost_price_fabric_C_is_120_l1518_151891

/-- Calculates the cost price per meter of fabric given the selling price, number of meters, and profit per meter. -/
def costPricePerMeter (sellingPrice : ℚ) (meters : ℚ) (profitPerMeter : ℚ) : ℚ :=
  (sellingPrice - meters * profitPerMeter) / meters

/-- Represents the fabric types and their properties -/
structure FabricType where
  name : String
  sellingPrice : ℚ
  meters : ℚ
  profitPerMeter : ℚ

/-- Theorem stating that the cost price per meter calculation is correct for all fabric types -/
theorem cost_price_per_meter_correct (fabric : FabricType) :
  costPricePerMeter fabric.sellingPrice fabric.meters fabric.profitPerMeter =
  (fabric.sellingPrice - fabric.meters * fabric.profitPerMeter) / fabric.meters :=
by
  sorry

/-- The three fabric types given in the problem -/
def fabricA : FabricType := ⟨"A", 6000, 45, 12⟩
def fabricB : FabricType := ⟨"B", 10800, 60, 15⟩
def fabricC : FabricType := ⟨"C", 3900, 30, 10⟩

/-- Theorem stating that the cost price per meter for fabric C is 120 -/
theorem cost_price_fabric_C_is_120 :
  costPricePerMeter fabricC.sellingPrice fabricC.meters fabricC.profitPerMeter = 120 :=
by
  sorry

end cost_price_per_meter_correct_cost_price_fabric_C_is_120_l1518_151891


namespace blue_pens_to_pencils_ratio_l1518_151859

theorem blue_pens_to_pencils_ratio 
  (blue_pens black_pens red_pens pencils : ℕ) : 
  black_pens = blue_pens + 10 →
  pencils = 8 →
  red_pens = pencils - 2 →
  blue_pens + black_pens + red_pens = 48 →
  blue_pens = 2 * pencils :=
by sorry

end blue_pens_to_pencils_ratio_l1518_151859


namespace floor_sqrt_80_l1518_151880

theorem floor_sqrt_80 : ⌊Real.sqrt 80⌋ = 8 := by sorry

end floor_sqrt_80_l1518_151880


namespace x_equals_one_sufficient_not_necessary_l1518_151822

theorem x_equals_one_sufficient_not_necessary (x : ℝ) :
  (x = 1 → x * (x - 1) = 0) ∧ (∃ y : ℝ, y ≠ 1 ∧ y * (y - 1) = 0) := by
  sorry

end x_equals_one_sufficient_not_necessary_l1518_151822


namespace product_of_solutions_l1518_151886

theorem product_of_solutions (x : ℝ) : 
  (|5 * x| + 7 = 47) → (∃ y : ℝ, |5 * y| + 7 = 47 ∧ x * y = -64) :=
by sorry

end product_of_solutions_l1518_151886


namespace difference_of_squares_factorization_l1518_151832

theorem difference_of_squares_factorization (y : ℝ) :
  49 - 16 * y^2 = (7 - 4*y) * (7 + 4*y) := by
  sorry

end difference_of_squares_factorization_l1518_151832


namespace min_value_reciprocal_sum_l1518_151862

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x * y = 4) :
  ∀ a b : ℝ, a > 0 → b > 0 → a * b = 4 → (1 / x + 1 / y) ≤ (1 / a + 1 / b) ∧
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x * y = 4 ∧ 1 / x + 1 / y = 2 :=
sorry

end min_value_reciprocal_sum_l1518_151862


namespace sum_of_coordinates_l1518_151816

/-- Given a function g where g(4) = 8, and h defined as h(x) = (g(x))^2 + 1,
    prove that the sum of coordinates of the point (4, h(4)) is 69. -/
theorem sum_of_coordinates (g : ℝ → ℝ) (h : ℝ → ℝ) : 
  g 4 = 8 → 
  (∀ x, h x = (g x)^2 + 1) → 
  4 + h 4 = 69 := by
sorry

end sum_of_coordinates_l1518_151816


namespace salt_solution_problem_l1518_151853

theorem salt_solution_problem (x : ℝ) : 
  x > 0 →  -- Ensure x is positive
  let initial_salt := 0.2 * x
  let after_evaporation := 0.75 * x
  let final_volume := after_evaporation + 7 + 14
  let final_salt := initial_salt + 14
  (final_salt / final_volume = 1/3) →
  x = 140 :=
by sorry

end salt_solution_problem_l1518_151853


namespace rainfall_second_week_l1518_151842

/-- Proves that given a total rainfall of 40 inches over two weeks, 
    where the second week's rainfall is 1.5 times the first week's, 
    the rainfall in the second week is 24 inches. -/
theorem rainfall_second_week (total_rainfall : ℝ) (ratio : ℝ) : 
  total_rainfall = 40 ∧ ratio = 1.5 → 
  ∃ (first_week : ℝ), 
    first_week + ratio * first_week = total_rainfall ∧ 
    ratio * first_week = 24 := by
  sorry

#check rainfall_second_week

end rainfall_second_week_l1518_151842


namespace trig_expression_value_l1518_151897

theorem trig_expression_value (x : Real) (h : Real.tan x = -1/2) :
  Real.sin x ^ 2 + 3 * Real.sin x * Real.cos x - 1 = -2 := by
  sorry

end trig_expression_value_l1518_151897


namespace max_chocolates_ben_l1518_151867

theorem max_chocolates_ben (total : ℕ) (ben carol : ℕ) (k : ℕ) : 
  total = 30 →
  ben + carol = total →
  carol = k * ben →
  k > 0 →
  ben ≤ 15 :=
by sorry

end max_chocolates_ben_l1518_151867


namespace mean_median_difference_l1518_151883

/-- Represents the score distribution in a class -/
structure ScoreDistribution where
  total_students : ℕ
  score_75_percent : ℚ
  score_82_percent : ℚ
  score_87_percent : ℚ
  score_90_percent : ℚ
  score_98_percent : ℚ

/-- Calculates the mean score given a score distribution -/
def mean_score (sd : ScoreDistribution) : ℚ :=
  (75 * sd.score_75_percent + 82 * sd.score_82_percent + 87 * sd.score_87_percent +
   90 * sd.score_90_percent + 98 * sd.score_98_percent) / 100

/-- Calculates the median score given a score distribution -/
def median_score (sd : ScoreDistribution) : ℚ := 87

/-- The main theorem stating the difference between mean and median scores -/
theorem mean_median_difference (sd : ScoreDistribution) 
  (h1 : sd.total_students = 10)
  (h2 : sd.score_75_percent = 15)
  (h3 : sd.score_82_percent = 10)
  (h4 : sd.score_87_percent = 40)
  (h5 : sd.score_90_percent = 20)
  (h6 : sd.score_98_percent = 15) :
  |mean_score sd - median_score sd| = 9 := by
  sorry

end mean_median_difference_l1518_151883


namespace triangle_area_l1518_151808

-- Define the curve
def f (x : ℝ) : ℝ := (x - 4)^2 * (x + 3)

-- Define the x-intercepts
def x_intercept_1 : ℝ := -3
def x_intercept_2 : ℝ := 4

-- Define the y-intercept
def y_intercept : ℝ := f 0

-- Theorem statement
theorem triangle_area : 
  let base := x_intercept_2 - x_intercept_1
  let height := y_intercept
  (1 / 2 : ℝ) * base * height = 168 := by sorry

end triangle_area_l1518_151808


namespace like_terms_imply_sum_l1518_151855

-- Define the concept of "like terms" for our specific case
def are_like_terms (m n : ℤ) : Prop :=
  m + 3 = 4 ∧ n + 3 = 1

-- State the theorem
theorem like_terms_imply_sum (m n : ℤ) :
  are_like_terms m n → m + n = -1 := by
  sorry

end like_terms_imply_sum_l1518_151855


namespace buses_per_week_is_165_l1518_151866

/-- Calculates the number of buses leaving a station in a week -/
def total_buses_per_week (
  weekday_interval : ℕ
  ) (weekday_hours : ℕ
  ) (weekday_count : ℕ
  ) (weekend_interval : ℕ
  ) (weekend_hours : ℕ
  ) (weekend_count : ℕ
  ) : ℕ :=
  let weekday_buses := weekday_count * (weekday_hours * 60 / weekday_interval)
  let weekend_buses := weekend_count * (weekend_hours * 60 / weekend_interval)
  weekday_buses + weekend_buses

/-- Theorem stating that the total number of buses leaving the station in a week is 165 -/
theorem buses_per_week_is_165 :
  total_buses_per_week 40 14 5 20 10 2 = 165 := by
  sorry


end buses_per_week_is_165_l1518_151866


namespace harveys_steak_sales_l1518_151888

/-- Given the initial number of steaks, the number left after the first sale,
    and the number of additional steaks sold, calculate the total number of steaks sold. -/
def total_steaks_sold (initial : ℕ) (left_after_first_sale : ℕ) (additional_sold : ℕ) : ℕ :=
  (initial - left_after_first_sale) + additional_sold

/-- Theorem stating that for Harvey's specific case, the total number of steaks sold is 17. -/
theorem harveys_steak_sales : total_steaks_sold 25 12 4 = 17 := by
  sorry

end harveys_steak_sales_l1518_151888


namespace child_ticket_cost_l1518_151839

/-- Proves that the cost of a child ticket is $3.50 given the specified conditions -/
theorem child_ticket_cost (adult_price : ℝ) (total_tickets : ℕ) (total_cost : ℝ) (adult_tickets : ℕ) : ℝ :=
  let child_tickets := total_tickets - adult_tickets
  let child_price := (total_cost - (adult_price * adult_tickets)) / child_tickets
  by
    -- Assuming:
    have h1 : adult_price = 5.50 := by sorry
    have h2 : total_tickets = 21 := by sorry
    have h3 : total_cost = 83.50 := by sorry
    have h4 : adult_tickets = 5 := by sorry

    -- Proof goes here
    sorry

    -- Conclusion
    -- child_price = 3.50

end child_ticket_cost_l1518_151839


namespace garden_length_l1518_151814

/-- Proves that the length of the larger garden is 90 yards given the conditions -/
theorem garden_length (w : ℝ) (l : ℝ) : 
  l = 3 * w →  -- larger garden length is three times its width
  360 = 2 * l + 2 * w + 2 * (w / 2) + 2 * (l / 2) →  -- total fencing equals 360 yards
  l = 90 := by
  sorry

#check garden_length

end garden_length_l1518_151814


namespace circle_radii_order_l1518_151837

theorem circle_radii_order (r_A r_B r_C : ℝ) : 
  r_A = 2 →
  2 * Real.pi * r_B = 10 * Real.pi →
  Real.pi * r_C^2 = 16 * Real.pi →
  r_A < r_C ∧ r_C < r_B := by
sorry

end circle_radii_order_l1518_151837


namespace function_properties_l1518_151818

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := 3 * Real.sin (2 * x + φ)

theorem function_properties (φ : ℝ) 
  (h1 : φ ∈ Set.Ioo (-π) 0)
  (h2 : ∀ x, f x φ = f (π/4 - x) φ) :
  φ = -3*π/4 ∧
  (∀ k : ℤ, ∀ x : ℝ, 5*π/8 + k*π ≤ x ∧ x ≤ 9*π/8 + k*π → 
    ∀ y : ℝ, x < y → f y φ < f x φ) ∧
  Set.range (fun x => f x φ) = Set.Icc (-3) (3*Real.sqrt 2/2) :=
by sorry

end function_properties_l1518_151818


namespace tree_space_calculation_l1518_151851

/-- Given a road of length 151 feet where 11 trees are planted with 14 feet between each tree,
    prove that each tree occupies 1 square foot of sidewalk space. -/
theorem tree_space_calculation (road_length : ℕ) (num_trees : ℕ) (gap_between_trees : ℕ) :
  road_length = 151 →
  num_trees = 11 →
  gap_between_trees = 14 →
  (road_length - (num_trees - 1) * gap_between_trees) / num_trees = 1 := by
  sorry

end tree_space_calculation_l1518_151851


namespace martin_family_ice_cream_l1518_151879

/-- The cost of ice cream for the Martin family at the mall --/
def ice_cream_cost (double_scoop_price : ℕ) : Prop :=
  let kiddie_scoop_price : ℕ := 3
  let regular_scoop_price : ℕ := 4
  let num_regular_scoops : ℕ := 2  -- Mr. and Mrs. Martin
  let num_kiddie_scoops : ℕ := 2   -- Two children
  let num_double_scoops : ℕ := 3   -- Three teenage children
  let total_cost : ℕ := 32
  (num_regular_scoops * regular_scoop_price +
   num_kiddie_scoops * kiddie_scoop_price +
   num_double_scoops * double_scoop_price) = total_cost

theorem martin_family_ice_cream : ice_cream_cost 6 := by
  sorry

end martin_family_ice_cream_l1518_151879


namespace quadratic_inequality_solution_range_l1518_151884

theorem quadratic_inequality_solution_range (c : ℝ) : 
  (c > 0) → 
  (∃ x : ℝ, x^2 - 8*x + c < 0) ↔ 
  (c > 0 ∧ c < 16) :=
by sorry

end quadratic_inequality_solution_range_l1518_151884


namespace expression_simplification_and_evaluation_l1518_151800

theorem expression_simplification_and_evaluation :
  let x := Real.tan (45 * π / 180) + Real.cos (30 * π / 180)
  (x / (x^2 - 1)) * ((x - 1) / x - 2) = -2 * Real.sqrt 3 / 3 := by
  sorry

end expression_simplification_and_evaluation_l1518_151800


namespace range_of_f_l1518_151841

def f (x : ℝ) : ℝ := |x + 8| - |x - 3|

theorem range_of_f :
  Set.range f = Set.Icc (-5) 17 := by
  sorry

end range_of_f_l1518_151841


namespace min_value_x_plus_y_l1518_151815

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : (x + 1) * (y + 1) = 9) :
  ∀ a b : ℝ, a > 0 → b > 0 → (a + 1) * (b + 1) = 9 → x + y ≤ a + b ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ (x + 1) * (y + 1) = 9 ∧ x + y = 4 :=
sorry

end min_value_x_plus_y_l1518_151815


namespace shaded_area_four_circles_l1518_151864

/-- The area of the shaded region formed by the intersection of four circles -/
theorem shaded_area_four_circles (r : ℝ) (h : r = 5) : 
  let circle_area := π * r^2
  let quarter_circle_area := circle_area / 4
  let triangle_area := r^2 / 2
  let shaded_segment := quarter_circle_area - triangle_area
  4 * shaded_segment = 25 * π - 50 := by sorry

end shaded_area_four_circles_l1518_151864


namespace intersection_complement_theorem_l1518_151861

open Set

-- Define the universal set U as the real numbers
def U : Set ℝ := univ

-- Define set M
def M : Set ℝ := {x | 0 < x ∧ x < 2}

-- Define set N
def N : Set ℝ := {x | x ≥ 1}

-- State the theorem
theorem intersection_complement_theorem :
  M ∩ (U \ N) = {x : ℝ | 0 < x ∧ x < 1} := by sorry

end intersection_complement_theorem_l1518_151861


namespace average_speed_theorem_l1518_151877

def speed_1 : ℝ := 100
def speed_2 : ℝ := 80
def speed_3_4 : ℝ := 90
def speed_5 : ℝ := 60
def speed_6 : ℝ := 70

def duration_1 : ℝ := 1
def duration_2 : ℝ := 1
def duration_3_4 : ℝ := 2
def duration_5 : ℝ := 1
def duration_6 : ℝ := 1

def total_distance : ℝ := 
  speed_1 * duration_1 + 
  speed_2 * duration_2 + 
  speed_3_4 * duration_3_4 + 
  speed_5 * duration_5 + 
  speed_6 * duration_6

def total_time : ℝ := 
  duration_1 + duration_2 + duration_3_4 + duration_5 + duration_6

theorem average_speed_theorem : 
  total_distance / total_time = 490 / 6 := by
  sorry

end average_speed_theorem_l1518_151877


namespace cards_left_l1518_151865

/-- Given that Nell had 242 cards initially and gave away 136 cards,
    prove that she has 106 cards left. -/
theorem cards_left (initial_cards given_away_cards : ℕ) 
  (h1 : initial_cards = 242)
  (h2 : given_away_cards = 136) :
  initial_cards - given_away_cards = 106 := by
  sorry

end cards_left_l1518_151865


namespace candy_store_sampling_theorem_l1518_151899

/-- The percentage of customers who sample candy but are not caught -/
def uncaught_samplers (total_samplers caught_samplers : ℝ) : ℝ :=
  total_samplers - caught_samplers

theorem candy_store_sampling_theorem 
  (total_samplers : ℝ) 
  (caught_samplers : ℝ) 
  (h1 : caught_samplers = 22)
  (h2 : total_samplers = 23.913043478260867) :
  uncaught_samplers total_samplers caught_samplers = 1.913043478260867 := by
  sorry

end candy_store_sampling_theorem_l1518_151899


namespace basketball_percentage_l1518_151896

theorem basketball_percentage (total_students : ℕ) (chess_percent : ℚ) (chess_or_basketball : ℕ) : 
  total_students = 250 →
  chess_percent = 1/10 →
  chess_or_basketball = 125 →
  ∃ (basketball_percent : ℚ), 
    basketball_percent = 2/5 ∧ 
    (basketball_percent + chess_percent) * total_students = chess_or_basketball :=
by
  sorry

end basketball_percentage_l1518_151896


namespace smallest_integer_fraction_eleven_satisfies_smallest_integer_is_eleven_l1518_151828

theorem smallest_integer_fraction (y : ℤ) : (5 : ℚ) / 8 < (y : ℚ) / 17 → y ≥ 11 :=
by sorry

theorem eleven_satisfies (y : ℤ) : (5 : ℚ) / 8 < (11 : ℚ) / 17 :=
by sorry

theorem smallest_integer_is_eleven : 
  ∃ y : ℤ, ((5 : ℚ) / 8 < (y : ℚ) / 17) ∧ (∀ z : ℤ, (5 : ℚ) / 8 < (z : ℚ) / 17 → z ≥ y) ∧ y = 11 :=
by sorry

end smallest_integer_fraction_eleven_satisfies_smallest_integer_is_eleven_l1518_151828


namespace extended_altitude_triangle_l1518_151838

/-- Given a triangle ABC with sides a, b, c, angles α, β, γ, and area t,
    we extend its altitudes beyond the sides by their own lengths to form
    a new triangle A'B'C' with sides a', b', c' and area t'. -/
theorem extended_altitude_triangle
  (a b c a' b' c' : ℝ)
  (α β γ : ℝ)
  (t t' : ℝ)
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ α > 0 ∧ β > 0 ∧ γ > 0)
  (h_angles : α + β + γ = Real.pi)
  (h_area : t = (1/2) * a * b * Real.sin γ)
  (h_extended : a' > a ∧ b' > b ∧ c' > c) :
  (a'^2 + b'^2 + c'^2 - (a^2 + b^2 + c^2) = 32 * t * Real.sin α * Real.sin β * Real.sin γ) ∧
  (t' = t * (3 + 8 * Real.cos α * Real.cos β * Real.cos γ)) := by
  sorry

end extended_altitude_triangle_l1518_151838


namespace constant_function_proof_l1518_151801

theorem constant_function_proof (f : ℝ → ℝ) 
  (h1 : ∀ x, f (2 + x) = 2 - f x) 
  (h2 : ∀ x, f (x + 3) ≥ f x) : 
  ∀ x, f x = 1 := by
  sorry

end constant_function_proof_l1518_151801


namespace max_triangle_area_is_sqrt3_div_2_l1518_151831

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a line in 2D space -/
structure Line where
  k : ℝ
  m : ℝ

/-- The maximum area of triangle AOB for the given ellipse and line conditions -/
def max_triangle_area (e : Ellipse) (l : Line) : ℝ :=
  sorry

/-- Main theorem: The maximum area of triangle AOB is √3/2 under the given conditions -/
theorem max_triangle_area_is_sqrt3_div_2 
  (e : Ellipse) 
  (h_vertex : e.b = 1)
  (h_eccentricity : Real.sqrt (e.a^2 - e.b^2) / e.a = Real.sqrt 6 / 3)
  (l : Line)
  (h_distance : |l.m| / Real.sqrt (1 + l.k^2) = Real.sqrt 3 / 2) :
  max_triangle_area e l = Real.sqrt 3 / 2 :=
sorry

end max_triangle_area_is_sqrt3_div_2_l1518_151831


namespace max_min_f_on_I_l1518_151834

-- Define the function
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 5

-- Define the interval
def I : Set ℝ := Set.Icc 0 3

-- State the theorem
theorem max_min_f_on_I :
  ∃ (a b : ℝ), a ∈ I ∧ b ∈ I ∧
  (∀ x ∈ I, f x ≤ f a) ∧
  (∀ x ∈ I, f x ≥ f b) ∧
  f a = 5 ∧ f b = -15 := by
sorry

end max_min_f_on_I_l1518_151834


namespace minimum_pass_rate_four_subjects_l1518_151875

theorem minimum_pass_rate_four_subjects 
  (math_pass : Real) (chinese_pass : Real) (english_pass : Real) (chemistry_pass : Real)
  (h_math : math_pass = 0.99)
  (h_chinese : chinese_pass = 0.98)
  (h_english : english_pass = 0.96)
  (h_chemistry : chemistry_pass = 0.92) :
  1 - (1 - math_pass + 1 - chinese_pass + 1 - english_pass + 1 - chemistry_pass) = 0.85 := by
  sorry

end minimum_pass_rate_four_subjects_l1518_151875


namespace circle_symmetry_min_value_l1518_151846

/-- The minimum value of 1/a + 3/b for a circle symmetric to a line --/
theorem circle_symmetry_min_value (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_symmetry : ∃ (x y : ℝ), x^2 + y^2 + 2*x - 6*y + 1 = 0 ∧ a*x - b*y + 3 = 0) :
  (∀ (a' b' : ℝ), a' > 0 → b' > 0 → 
    (∃ (x y : ℝ), x^2 + y^2 + 2*x - 6*y + 1 = 0 ∧ a'*x - b'*y + 3 = 0) → 
    1/a + 3/b ≤ 1/a' + 3/b') ∧
  (1/a + 3/b = 16/3) := by sorry

end circle_symmetry_min_value_l1518_151846


namespace last_second_occurrence_is_two_l1518_151858

-- Define the Fibonacci sequence modulo 10
def fib_mod_10 : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => (fib_mod_10 n + fib_mod_10 (n + 1)) % 10

-- Define a function to check if a digit has appeared at least twice up to a given index
def appears_twice (d : ℕ) (n : ℕ) : Prop :=
  ∃ i j, i < j ∧ j ≤ n ∧ fib_mod_10 i = d ∧ fib_mod_10 j = d

-- State the theorem
theorem last_second_occurrence_is_two :
  ∀ d, d ≠ 2 → ∃ n, appears_twice d n ∧ ¬appears_twice 2 n :=
sorry

end last_second_occurrence_is_two_l1518_151858


namespace expression_simplification_l1518_151885

theorem expression_simplification (a b x y : ℝ) :
  (2*a - (4*a + 5*b) + 2*(3*a - 4*b) = 4*a - 13*b) ∧
  (5*x^2 - 2*(3*y^2 - 5*x^2) + (-4*y^2 + 7*x*y) = 15*x^2 - 10*y^2 + 7*x*y) :=
by sorry

end expression_simplification_l1518_151885


namespace min_workers_theorem_l1518_151873

-- Define the problem parameters
def total_days : ℕ := 40
def days_worked : ℕ := 10
def initial_workers : ℕ := 10
def work_completed : ℚ := 1/4

-- Define the function to calculate the minimum number of workers
def min_workers_needed (total_days : ℕ) (days_worked : ℕ) (initial_workers : ℕ) (work_completed : ℚ) : ℕ :=
  -- Implementation details are not provided in the statement
  sorry

-- Theorem statement
theorem min_workers_theorem :
  min_workers_needed total_days days_worked initial_workers work_completed = 10 :=
by sorry

end min_workers_theorem_l1518_151873


namespace dusty_single_layer_purchase_l1518_151812

/-- Represents the cost and quantity of cake slices purchased by Dusty -/
structure CakePurchase where
  single_layer_price : ℕ
  double_layer_price : ℕ
  double_layer_quantity : ℕ
  payment : ℕ
  change : ℕ

/-- Calculates the number of single layer cake slices purchased -/
def single_layer_quantity (purchase : CakePurchase) : ℕ :=
  (purchase.payment - purchase.change - purchase.double_layer_price * purchase.double_layer_quantity) / purchase.single_layer_price

/-- Theorem stating that Dusty bought 7 single layer cake slices -/
theorem dusty_single_layer_purchase :
  let purchase := CakePurchase.mk 4 7 5 100 37
  single_layer_quantity purchase = 7 := by
  sorry

end dusty_single_layer_purchase_l1518_151812


namespace sin_1050_degrees_l1518_151845

theorem sin_1050_degrees : Real.sin (1050 * π / 180) = -1/2 := by
  sorry

end sin_1050_degrees_l1518_151845


namespace fraction_simplification_l1518_151860

theorem fraction_simplification (a b : ℝ) (h1 : a ≠ b) (h2 : b ≠ 0) :
  let x := a^2 / b^2
  (a^2 + b^2) / (a^2 - b^2) = (x + 1) / (x - 1) := by
  sorry

end fraction_simplification_l1518_151860


namespace trapezoid_perimeter_l1518_151898

/-- The perimeter of a trapezoid JKLM with given coordinates is 36 units -/
theorem trapezoid_perimeter : 
  let J : ℝ × ℝ := (-2, -4)
  let K : ℝ × ℝ := (-2, 2)
  let L : ℝ × ℝ := (6, 8)
  let M : ℝ × ℝ := (6, -4)
  let dist (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  dist J K + dist K L + dist L M + dist M J = 36 := by
  sorry

end trapezoid_perimeter_l1518_151898


namespace total_original_cost_of_cars_l1518_151825

/-- Calculates the original price of a car before depreciation -/
def originalPrice (soldPrice : ℚ) (depreciationRate : ℚ) : ℚ :=
  soldPrice / (1 - depreciationRate)

/-- Proves that the total original cost of two cars is $3058.82 -/
theorem total_original_cost_of_cars 
  (oldCarSoldPrice : ℚ) 
  (secondOldestCarSoldPrice : ℚ) 
  (oldCarDepreciationRate : ℚ) 
  (secondOldestCarDepreciationRate : ℚ)
  (h1 : oldCarSoldPrice = 1800)
  (h2 : secondOldestCarSoldPrice = 900)
  (h3 : oldCarDepreciationRate = 1/10)
  (h4 : secondOldestCarDepreciationRate = 3/20) :
  originalPrice oldCarSoldPrice oldCarDepreciationRate + 
  originalPrice secondOldestCarSoldPrice secondOldestCarDepreciationRate = 3058.82 := by
  sorry

#eval originalPrice 1800 (1/10) + originalPrice 900 (3/20)

end total_original_cost_of_cars_l1518_151825


namespace exponential_equation_solution_l1518_151863

theorem exponential_equation_solution :
  ∃ m : ℤ, (3 : ℝ)^m * 9^m = 81^(m - 24) ∧ m = 96 :=
by
  sorry

end exponential_equation_solution_l1518_151863


namespace truck_fuel_relationship_l1518_151890

/-- Represents the fuel consumption model of a truck -/
structure TruckFuelModel where
  tankCapacity : ℝ
  fuelConsumptionRate : ℝ

/-- Calculates the remaining fuel in the tank after a given time -/
def remainingFuel (model : TruckFuelModel) (time : ℝ) : ℝ :=
  model.tankCapacity - model.fuelConsumptionRate * time

/-- Theorem: The relationship between remaining fuel and traveling time for the given truck -/
theorem truck_fuel_relationship (model : TruckFuelModel) 
  (h1 : model.tankCapacity = 60)
  (h2 : model.fuelConsumptionRate = 8) :
  ∀ t : ℝ, remainingFuel model t = 60 - 8 * t :=
by sorry

end truck_fuel_relationship_l1518_151890


namespace apex_distance_theorem_l1518_151889

/-- Represents a right octagonal pyramid with two parallel cross sections -/
structure RightOctagonalPyramid where
  small_area : ℝ
  large_area : ℝ
  plane_distance : ℝ

/-- The distance from the apex to the plane of the larger cross section -/
def apex_to_large_section (p : RightOctagonalPyramid) : ℝ :=
  36 -- We define this as 36 based on the problem statement

/-- Theorem stating the relationship between the pyramid's properties and the apex distance -/
theorem apex_distance_theorem (p : RightOctagonalPyramid) 
  (h1 : p.small_area = 256 * Real.sqrt 2)
  (h2 : p.large_area = 576 * Real.sqrt 2)
  (h3 : p.plane_distance = 12) :
  apex_to_large_section p = 36 := by
  sorry

#check apex_distance_theorem

end apex_distance_theorem_l1518_151889


namespace second_discount_percentage_l1518_151809

theorem second_discount_percentage
  (list_price : ℝ)
  (final_price : ℝ)
  (first_discount : ℝ)
  (h1 : list_price = 150)
  (h2 : final_price = 105)
  (h3 : first_discount = 19.954259576901087)
  : ∃ (second_discount : ℝ), 
    abs (second_discount - 12.552) < 0.001 ∧
    final_price = list_price * (1 - first_discount / 100) * (1 - second_discount / 100) :=
by sorry

end second_discount_percentage_l1518_151809


namespace angle_double_quadrant_l1518_151829

/-- Given that α is an angle in the second quadrant, prove that 2α is an angle in the third or fourth quadrant. -/
theorem angle_double_quadrant (α : Real) (h : π/2 < α ∧ α < π) :
  π < 2*α ∧ 2*α < 2*π :=
by sorry

end angle_double_quadrant_l1518_151829


namespace scientific_notation_of_384000_l1518_151836

/-- Given a number 384000, prove that its scientific notation representation is 3.84 × 10^5 -/
theorem scientific_notation_of_384000 :
  (384000 : ℝ) = 3.84 * (10 : ℝ)^5 := by
  sorry

end scientific_notation_of_384000_l1518_151836


namespace inequality_and_equality_condition_l1518_151806

theorem inequality_and_equality_condition (a b : ℝ) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : a + b < 2) :
  (1 / (1 + a^2) + 1 / (1 + b^2) ≤ 2 / (1 + a*b)) ∧
  ((1 / (1 + a^2) + 1 / (1 + b^2) = 2 / (1 + a*b)) ↔ a = b) := by
  sorry

end inequality_and_equality_condition_l1518_151806


namespace min_value_when_a_is_one_range_of_a_l1518_151830

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + 1| + |x - 4| - a

-- Theorem for the minimum value when a = 1
theorem min_value_when_a_is_one :
  ∀ x : ℝ, f 1 x ≥ 4 ∧ ∃ y : ℝ, f 1 y = 4 :=
sorry

-- Theorem for the range of a
theorem range_of_a :
  ∀ a : ℝ, (∀ x : ℝ, f a x ≥ 4/a + 1) ↔ (a < 0 ∨ a = 2) :=
sorry

end min_value_when_a_is_one_range_of_a_l1518_151830


namespace slope_of_line_from_equation_l1518_151840

theorem slope_of_line_from_equation (x₁ x₂ y₁ y₂ : ℝ) 
  (h₁ : x₁ ≠ x₂)
  (h₂ : (3 : ℝ) / x₁ + (4 : ℝ) / y₁ = 0)
  (h₃ : (3 : ℝ) / x₂ + (4 : ℝ) / y₂ = 0) :
  (y₂ - y₁) / (x₂ - x₁) = -(4 : ℝ) / 3 := by
sorry

end slope_of_line_from_equation_l1518_151840


namespace polynomial_division_remainder_l1518_151894

theorem polynomial_division_remainder : ∃ q : Polynomial ℚ, 
  3 * X^2 - 22 * X + 63 = (X - 3) * q + 24 := by
  sorry

end polynomial_division_remainder_l1518_151894


namespace artichoke_dip_theorem_l1518_151807

/-- The amount of money Hakeem has to spend on artichokes -/
def budget : ℚ := 15

/-- The cost of one artichoke -/
def artichoke_cost : ℚ := 5/4

/-- The number of artichokes needed to make a batch of dip -/
def artichokes_per_batch : ℕ := 3

/-- The amount of dip (in ounces) produced from one batch -/
def dip_per_batch : ℚ := 5

/-- The maximum amount of dip (in ounces) that can be made with the given budget -/
def max_dip : ℚ := 20

theorem artichoke_dip_theorem :
  (budget / artichoke_cost).floor * (dip_per_batch / artichokes_per_batch) = max_dip :=
sorry

end artichoke_dip_theorem_l1518_151807


namespace exists_even_non_zero_from_step_two_l1518_151835

/-- Represents the state of the sequence at a given step -/
def SequenceState := ℤ → ℤ

/-- The initial state of the sequence -/
def initial_state : SequenceState :=
  fun i => if i = 0 then 1 else 0

/-- Updates the sequence for one step -/
def update_sequence (s : SequenceState) : SequenceState :=
  fun i => s (i - 1) + s i + s (i + 1)

/-- Checks if a number is even and non-zero -/
def is_even_non_zero (n : ℤ) : Prop :=
  n ≠ 0 ∧ n % 2 = 0

/-- The sequence after n steps -/
def sequence_at_step (n : ℕ) : SequenceState :=
  match n with
  | 0 => initial_state
  | n + 1 => update_sequence (sequence_at_step n)

/-- The main theorem to be proved -/
theorem exists_even_non_zero_from_step_two (n : ℕ) (h : n ≥ 2) :
  ∃ i : ℤ, is_even_non_zero ((sequence_at_step n) i) :=
sorry

end exists_even_non_zero_from_step_two_l1518_151835


namespace expression_value_l1518_151804

theorem expression_value (a b c : ℤ) (ha : a = 12) (hb : b = 8) (hc : c = 3) :
  ((a - b + c) - (a - (b + c))) = 6 := by
  sorry

end expression_value_l1518_151804


namespace expression_simplification_l1518_151826

theorem expression_simplification :
  let x : ℝ := Real.sqrt 2 + 1
  let expr := ((2 * x - 1) / (x + 1) - x + 1) / ((x - 2) / (x^2 + 2*x + 1))
  expr = -12 * Real.sqrt 2 - 20 := by
  sorry

end expression_simplification_l1518_151826


namespace smallest_n_for_interval_multiple_l1518_151881

theorem smallest_n_for_interval_multiple : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), 1 ≤ m ∧ m ≤ 1992 → 
    ∃ (k : ℕ), (m : ℚ) / 1993 < (k : ℚ) / n ∧ (k : ℚ) / n < ((m + 1) : ℚ) / 1994) ∧
  (∀ (n' : ℕ), 0 < n' ∧ n' < n → 
    ∃ (m : ℕ), 1 ≤ m ∧ m ≤ 1992 ∧
      ∀ (k : ℕ), ¬((m : ℚ) / 1993 < (k : ℚ) / n' ∧ (k : ℚ) / n' < ((m + 1) : ℚ) / 1994)) ∧
  n = 3987 :=
by sorry

end smallest_n_for_interval_multiple_l1518_151881


namespace sum_a_b_is_negative_two_l1518_151827

theorem sum_a_b_is_negative_two (a b : ℝ) (h : |a - 1| + (b + 3)^2 = 0) : a + b = -2 := by
  sorry

end sum_a_b_is_negative_two_l1518_151827


namespace eraser_buyers_difference_l1518_151811

theorem eraser_buyers_difference : ∀ (price : ℕ) (fifth_graders fourth_graders : ℕ),
  price > 0 →
  fifth_graders * price = 325 →
  fourth_graders * price = 460 →
  fourth_graders = 40 →
  fourth_graders - fifth_graders = 27 := by
sorry

end eraser_buyers_difference_l1518_151811


namespace number_equality_l1518_151878

theorem number_equality : ∃ x : ℝ, x * 120 = 173 * 240 ∧ x = 346 := by
  sorry

end number_equality_l1518_151878


namespace zoo_visitors_l1518_151874

theorem zoo_visitors (sandwiches_per_person : ℝ) (total_sandwiches : ℕ) :
  sandwiches_per_person = 3.0 →
  total_sandwiches = 657 →
  ↑total_sandwiches / sandwiches_per_person = 219 := by
sorry

end zoo_visitors_l1518_151874


namespace difference_between_fractions_l1518_151848

theorem difference_between_fractions (n : ℝ) (h : n = 140) : (4/5 * n) - (65/100 * n) = 21 := by
  sorry

end difference_between_fractions_l1518_151848


namespace arithmetic_geometric_relation_l1518_151854

/-- An arithmetic sequence with non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  h_d : d ≠ 0
  h_arith : ∀ n, a (n + 1) = a n + d

/-- A geometric sequence -/
structure GeometricSequence where
  b : ℕ → ℝ
  r : ℝ
  h_geom : ∀ n, b (n + 1) = r * b n

/-- The theorem statement -/
theorem arithmetic_geometric_relation (seq : ArithmeticSequence)
    (h_geom : ∃ (g : GeometricSequence), 
      g.b 1 = seq.a 2 ∧ g.b 2 = seq.a 3 ∧ g.b 3 = seq.a 7) :
    (∃ (g : GeometricSequence), 
      g.b 1 = seq.a 2 ∧ g.b 2 = seq.a 3 ∧ g.b 3 = seq.a 7 ∧ g.r = 4) := by
  sorry

end arithmetic_geometric_relation_l1518_151854


namespace polynomial_expansion_l1518_151869

theorem polynomial_expansion (x : ℝ) : 
  (7 * x + 3) * (5 * x^2 + 2 * x + 4) = 35 * x^3 + 29 * x^2 + 34 * x + 12 := by
  sorry

end polynomial_expansion_l1518_151869


namespace parkingLotSpaces_l1518_151882

/-- Represents a car parking lot with three sections. -/
structure ParkingLot where
  section1 : ℕ
  section2 : ℕ
  section3 : ℕ

/-- Calculates the total number of spaces in the parking lot. -/
def totalSpaces (lot : ParkingLot) : ℕ :=
  lot.section1 + lot.section2 + lot.section3

/-- Theorem stating the total number of spaces in the parking lot. -/
theorem parkingLotSpaces : ∃ (lot : ParkingLot), 
  lot.section1 = 320 ∧ 
  lot.section2 = 440 ∧ 
  lot.section2 = lot.section3 + 200 ∧
  totalSpaces lot = 1000 := by
  sorry

end parkingLotSpaces_l1518_151882


namespace third_group_students_l1518_151843

/-- The number of tissues in each mini tissue box -/
def tissues_per_box : ℕ := 40

/-- The number of students in the first kindergartner group -/
def group1_students : ℕ := 9

/-- The number of students in the second kindergartner group -/
def group2_students : ℕ := 10

/-- The total number of tissues brought by all groups -/
def total_tissues : ℕ := 1200

/-- Theorem stating that the number of students in the third kindergartner group is 11 -/
theorem third_group_students :
  ∃ (x : ℕ), x = 11 ∧ 
  tissues_per_box * (group1_students + group2_students + x) = total_tissues :=
sorry

end third_group_students_l1518_151843


namespace monic_quadratic_root_l1518_151817

theorem monic_quadratic_root (x : ℂ) : x^2 + 4*x + 9 = 0 ↔ x = -2 - Complex.I * Real.sqrt 5 ∨ x = -2 + Complex.I * Real.sqrt 5 :=
sorry

end monic_quadratic_root_l1518_151817
