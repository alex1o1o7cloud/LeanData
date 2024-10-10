import Mathlib

namespace large_circle_radius_l3195_319599

theorem large_circle_radius (C₁ C₂ C₃ C₄ O : ℝ × ℝ) (r : ℝ) :
  -- Four unit circles externally tangent in square formation
  r = 1 ∧
  dist C₁ C₂ = 2 ∧ dist C₂ C₃ = 2 ∧ dist C₃ C₄ = 2 ∧ dist C₄ C₁ = 2 ∧
  -- Large circle internally tangent to the four unit circles
  dist O C₁ = dist O C₂ ∧ dist O C₂ = dist O C₃ ∧ dist O C₃ = dist O C₄ ∧
  dist O C₁ = dist C₁ C₃ / 2 + r →
  -- Radius of the large circle
  dist O C₁ + r = Real.sqrt 2 + 2 := by
sorry


end large_circle_radius_l3195_319599


namespace jessica_test_score_l3195_319525

-- Define the given conditions
def initial_students : ℕ := 20
def initial_average : ℚ := 75
def new_students : ℕ := 21
def new_average : ℚ := 76

-- Define Jessica's score as a variable
def jessica_score : ℚ := sorry

-- Theorem to prove
theorem jessica_test_score : 
  (initial_students * initial_average + jessica_score) / new_students = new_average := by
  sorry

end jessica_test_score_l3195_319525


namespace expression_value_at_eight_l3195_319513

theorem expression_value_at_eight :
  let x : ℝ := 8
  (x^6 - 64*x^3 + 1024) / (x^3 - 16) = 480 := by
  sorry

end expression_value_at_eight_l3195_319513


namespace sunflower_germination_rate_l3195_319596

theorem sunflower_germination_rate 
  (daisy_seeds : ℕ) 
  (sunflower_seeds : ℕ) 
  (daisy_germination_rate : ℚ) 
  (flower_production_rate : ℚ) 
  (total_flowering_plants : ℕ) :
  daisy_seeds = 25 →
  sunflower_seeds = 25 →
  daisy_germination_rate = 3/5 →
  flower_production_rate = 4/5 →
  total_flowering_plants = 28 →
  (daisy_seeds : ℚ) * daisy_germination_rate * flower_production_rate +
  (sunflower_seeds : ℚ) * (4/5) * flower_production_rate = total_flowering_plants →
  (4/5) = 20 / sunflower_seeds :=
by sorry

end sunflower_germination_rate_l3195_319596


namespace tims_age_l3195_319590

theorem tims_age (james_age john_age tim_age : ℕ) : 
  james_age = 23 → 
  john_age = 35 → 
  tim_age = 2 * john_age - 5 → 
  tim_age = 65 := by
sorry

end tims_age_l3195_319590


namespace max_value_inequality_l3195_319567

theorem max_value_inequality (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (h : |1/a| + |1/b| + |1/c| ≤ 3) : 
  (a^2 + 4*(b^2 + c^2)) * (b^2 + 4*(a^2 + c^2)) * (c^2 + 4*(a^2 + b^2)) ≥ 729 ∧ 
  ∀ m > 729, ∃ a' b' c' : ℝ, a' ≠ 0 ∧ b' ≠ 0 ∧ c' ≠ 0 ∧ 
    |1/a'| + |1/b'| + |1/c'| ≤ 3 ∧
    (a'^2 + 4*(b'^2 + c'^2)) * (b'^2 + 4*(a'^2 + c'^2)) * (c'^2 + 4*(a'^2 + b'^2)) < m :=
by sorry

end max_value_inequality_l3195_319567


namespace smallest_divisible_by_four_and_five_l3195_319582

/-- A function that checks if a number contains the digits 1, 2, 3, 4, and 5 exactly once -/
def containsDigitsOnce (n : ℕ) : Prop := sorry

/-- A function that returns the set of all five-digit numbers containing 1, 2, 3, 4, and 5 exactly once -/
def fiveDigitSet : Set ℕ := {n : ℕ | 10000 ≤ n ∧ n < 100000 ∧ containsDigitsOnce n}

theorem smallest_divisible_by_four_and_five :
  ∃ (n : ℕ), n ∈ fiveDigitSet ∧ n % 4 = 0 ∧ n % 5 = 0 ∧
  ∀ (m : ℕ), m ∈ fiveDigitSet → m % 4 = 0 → m % 5 = 0 → n ≤ m ∧
  n = 14532 := by
  sorry

end smallest_divisible_by_four_and_five_l3195_319582


namespace jack_sugar_final_amount_l3195_319575

/-- Calculates the final amount of sugar Jack has after a series of transactions -/
def final_sugar_amount (initial : ℤ) (use_day2 borrow_day2 buy_day3 buy_day4 use_day5 return_day5 : ℤ) : ℤ :=
  initial - use_day2 - borrow_day2 + buy_day3 + buy_day4 - use_day5 + return_day5

/-- Theorem stating that Jack's final sugar amount is 85 pounds -/
theorem jack_sugar_final_amount :
  final_sugar_amount 65 18 5 30 20 10 3 = 85 := by
  sorry

end jack_sugar_final_amount_l3195_319575


namespace unique_quadratic_pair_l3195_319531

/-- A function that checks if a quadratic equation has exactly one real solution -/
def hasExactlyOneRealSolution (a b c : ℤ) : Prop :=
  b * b = 4 * a * c

/-- The theorem stating that there exists exactly one ordered pair (b,c) satisfying the conditions -/
theorem unique_quadratic_pair :
  ∃! (b c : ℕ), 
    0 < b ∧ b ≤ 6 ∧
    0 < c ∧ c ≤ 6 ∧
    hasExactlyOneRealSolution 1 b c ∧
    hasExactlyOneRealSolution 1 c b :=
sorry

end unique_quadratic_pair_l3195_319531


namespace solve_equation_l3195_319507

theorem solve_equation : ∃ y : ℚ, (2 / 7) * (1 / 8) * y = 12 ∧ y = 336 := by sorry

end solve_equation_l3195_319507


namespace curve_symmetric_line_k_l3195_319539

/-- The curve equation --/
def curve (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 6*y + 1 = 0

/-- The line equation --/
def line (k x y : ℝ) : Prop :=
  k*x + 2*y - 4 = 0

/-- Two points are symmetric with respect to a line --/
def symmetric (P Q : ℝ × ℝ) (k : ℝ) : Prop :=
  ∃ (x y : ℝ), line k x y ∧ 
    (P.1 + Q.1 = 2*x) ∧ (P.2 + Q.2 = 2*y)

theorem curve_symmetric_line_k (P Q : ℝ × ℝ) (k : ℝ) :
  P ≠ Q →
  curve P.1 P.2 →
  curve Q.1 Q.2 →
  symmetric P Q k →
  k = 2 := by
  sorry

end curve_symmetric_line_k_l3195_319539


namespace alarm_system_probability_l3195_319541

theorem alarm_system_probability (p : ℝ) (h1 : p = 0.4) :
  let prob_at_least_one := 1 - (1 - p) * (1 - p)
  prob_at_least_one = 0.64 := by
sorry

end alarm_system_probability_l3195_319541


namespace replacement_cost_theorem_l3195_319569

/-- The cost to replace all cardio machines in a chain of gyms -/
def total_replacement_cost (num_gyms : ℕ) (bikes_per_gym treadmills_per_gym ellipticals_per_gym : ℕ)
  (bike_cost : ℝ) : ℝ :=
  let treadmill_cost := 1.5 * bike_cost
  let elliptical_cost := 2 * treadmill_cost
  let total_bikes := num_gyms * bikes_per_gym
  let total_treadmills := num_gyms * treadmills_per_gym
  let total_ellipticals := num_gyms * ellipticals_per_gym
  total_bikes * bike_cost + total_treadmills * treadmill_cost + total_ellipticals * elliptical_cost

/-- Theorem stating the total cost to replace all cardio machines -/
theorem replacement_cost_theorem :
  total_replacement_cost 20 10 5 5 700 = 455000 := by
  sorry


end replacement_cost_theorem_l3195_319569


namespace diophantine_approximation_l3195_319574

theorem diophantine_approximation (x : ℝ) (h_irr : Irrational x) (h_pos : x > 0) :
  ∀ n : ℕ, ∃ p q : ℤ, q > n ∧ q > 0 ∧ |x - (p : ℝ) / q| ≤ 1 / q^2 := by
  sorry

end diophantine_approximation_l3195_319574


namespace trailing_zeroes_sum_factorials_l3195_319505

/-- Calculate the number of trailing zeroes in n! -/
def trailingZeroes (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125) + (n / 625)

/-- The number of trailing zeroes in 500! + 200! is 124 -/
theorem trailing_zeroes_sum_factorials :
  max (trailingZeroes 500) (trailingZeroes 200) = 124 := by sorry

end trailing_zeroes_sum_factorials_l3195_319505


namespace non_obtuse_triangle_perimeter_gt_four_circumradius_l3195_319598

/-- A triangle with vertices A, B, and C in the real plane. -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The perimeter of a triangle. -/
def perimeter (t : Triangle) : ℝ := sorry

/-- The radius of the circumcircle of a triangle. -/
def circumradius (t : Triangle) : ℝ := sorry

/-- Predicate to check if a triangle is non-obtuse. -/
def is_non_obtuse (t : Triangle) : Prop := sorry

/-- Theorem: For any non-obtuse triangle, its perimeter is greater than
    four times the radius of its circumcircle. -/
theorem non_obtuse_triangle_perimeter_gt_four_circumradius (t : Triangle) :
  is_non_obtuse t → perimeter t > 4 * circumradius t := by sorry

end non_obtuse_triangle_perimeter_gt_four_circumradius_l3195_319598


namespace trigonometric_identity_l3195_319572

theorem trigonometric_identity (α : Real) (h : 3 * Real.sin α + Real.cos α = 0) :
  1 / (Real.cos α ^ 2 + Real.sin (2 * α)) = 10 / 3 := by
  sorry

end trigonometric_identity_l3195_319572


namespace at_least_100_odd_population_days_l3195_319560

/-- Represents the state of the Martian population on a given day -/
structure PopulationState :=
  (day : ℕ)
  (births : ℕ)
  (population : ℕ)

/-- A function that calculates the population state for each day -/
def populationEvolution : ℕ → PopulationState → PopulationState :=
  sorry

/-- The total number of Martians born throughout history -/
def totalBirths : ℕ := sorry

/-- Theorem stating that there are at least 100 days with odd population -/
theorem at_least_100_odd_population_days
  (h_odd_births : Odd totalBirths)
  (h_lifespan : ∀ (m : ℕ), m < totalBirths → ∃ (b d : ℕ), d - b = 100 ∧ PopulationState.population (populationEvolution d (PopulationState.mk b 1 1)) = PopulationState.population (populationEvolution (d + 1) (PopulationState.mk b 1 1)) - 1) :
  ∃ (S : Finset ℕ), S.card ≥ 100 ∧ ∀ (d : ℕ), d ∈ S → Odd (PopulationState.population (populationEvolution d (PopulationState.mk 0 0 0))) :=
sorry

end at_least_100_odd_population_days_l3195_319560


namespace sphere_radius_calculation_l3195_319536

/-- Given a sphere on a horizontal plane, if a vertical stick casts a shadow and the sphere's shadow extends from its base, then we can calculate the radius of the sphere. -/
theorem sphere_radius_calculation (stick_height stick_shadow sphere_shadow : ℝ) 
  (stick_height_pos : stick_height > 0)
  (stick_shadow_pos : stick_shadow > 0)
  (sphere_shadow_pos : sphere_shadow > 0)
  (h_stick : stick_height = 1.5)
  (h_stick_shadow : stick_shadow = 1)
  (h_sphere_shadow : sphere_shadow = 8) :
  ∃ r : ℝ, r > 0 ∧ r / (sphere_shadow - r) = stick_height / stick_shadow ∧ r = 4.8 := by
sorry

end sphere_radius_calculation_l3195_319536


namespace socks_in_washing_machine_l3195_319537

/-- The number of players in a soccer match -/
def num_players : ℕ := 11

/-- The number of socks each player wears -/
def socks_per_player : ℕ := 2

/-- The total number of socks in the washing machine -/
def total_socks : ℕ := num_players * socks_per_player

theorem socks_in_washing_machine : total_socks = 22 := by
  sorry

end socks_in_washing_machine_l3195_319537


namespace restaurant_bill_theorem_l3195_319503

/-- Represents the cost structure and group composition at a restaurant --/
structure RestaurantBill where
  adult_meal_costs : Fin 3 → ℕ
  adult_beverage_cost : ℕ
  kid_beverage_cost : ℕ
  total_people : ℕ
  kids_count : ℕ
  adult_meal_counts : Fin 3 → ℕ
  total_beverages : ℕ

/-- Calculates the total bill for a group at the restaurant --/
def calculate_total_bill (bill : RestaurantBill) : ℕ :=
  let adult_meals_cost := (bill.adult_meal_costs 0 * bill.adult_meal_counts 0) +
                          (bill.adult_meal_costs 1 * bill.adult_meal_counts 1) +
                          (bill.adult_meal_costs 2 * bill.adult_meal_counts 2)
  let adult_beverages_cost := min (bill.total_people - bill.kids_count) bill.total_beverages * bill.adult_beverage_cost
  let kid_beverages_cost := (bill.total_beverages - min (bill.total_people - bill.kids_count) bill.total_beverages) * bill.kid_beverage_cost
  adult_meals_cost + adult_beverages_cost + kid_beverages_cost

/-- Theorem stating that the total bill for the given group is $59 --/
theorem restaurant_bill_theorem (bill : RestaurantBill)
  (h1 : bill.adult_meal_costs = ![5, 7, 9])
  (h2 : bill.adult_beverage_cost = 2)
  (h3 : bill.kid_beverage_cost = 1)
  (h4 : bill.total_people = 14)
  (h5 : bill.kids_count = 7)
  (h6 : bill.adult_meal_counts = ![4, 2, 1])
  (h7 : bill.total_beverages = 9) :
  calculate_total_bill bill = 59 := by
  sorry

end restaurant_bill_theorem_l3195_319503


namespace farthest_line_from_origin_l3195_319551

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The origin point (0,0) -/
def origin : Point := ⟨0, 0⟩

/-- The point A(1,2) -/
def pointA : Point := ⟨1, 2⟩

/-- Check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Calculate the distance from a point to a line -/
noncomputable def distancePointToLine (p : Point) (l : Line) : ℝ :=
  (abs (l.a * p.x + l.b * p.y + l.c)) / Real.sqrt (l.a^2 + l.b^2)

/-- The line x + 2y - 5 = 0 -/
def targetLine : Line := ⟨1, 2, -5⟩

theorem farthest_line_from_origin : 
  (pointOnLine pointA targetLine) ∧ 
  (∀ l : Line, pointOnLine pointA l → distancePointToLine origin targetLine ≥ distancePointToLine origin l) :=
sorry

end farthest_line_from_origin_l3195_319551


namespace weight_of_replaced_person_l3195_319512

/-- Proves that the weight of the replaced person is 45 kg given the conditions -/
theorem weight_of_replaced_person
  (n : ℕ)
  (original_average : ℝ)
  (weight_increase : ℝ)
  (new_person_weight : ℝ)
  (h1 : n = 8)
  (h2 : weight_increase = 2.5)
  (h3 : new_person_weight = 65)
  : ∃ (replaced_weight : ℝ),
    n * (original_average + weight_increase) - n * original_average
    = new_person_weight - replaced_weight
    ∧ replaced_weight = 45 := by
  sorry

end weight_of_replaced_person_l3195_319512


namespace intersection_of_A_and_B_l3195_319556

def A : Set ℝ := {-1, 1, 3, 5}
def B : Set ℝ := {x | x^2 - 4 < 0}

theorem intersection_of_A_and_B : A ∩ B = {-1, 1} := by sorry

end intersection_of_A_and_B_l3195_319556


namespace unique_prime_solution_l3195_319535

theorem unique_prime_solution : 
  ∃! (p m : ℕ), 
    Prime p ∧ 
    m > 0 ∧ 
    p^3 + m*(p + 2) = m^2 + p + 1 ∧ 
    p = 2 ∧ 
    m = 5 := by
  sorry

end unique_prime_solution_l3195_319535


namespace tan_beta_value_l3195_319520

theorem tan_beta_value (α β : Real) 
  (h1 : Real.tan α = -2) 
  (h2 : Real.tan (α + β) = 1) : 
  Real.tan β = -3 := by sorry

end tan_beta_value_l3195_319520


namespace difference_divisible_by_19_l3195_319516

theorem difference_divisible_by_19 (n : ℕ) : 26^n ≡ 7^n [ZMOD 19] := by
  sorry

end difference_divisible_by_19_l3195_319516


namespace garden_center_discount_l3195_319559

/-- Represents the purchase and payment details at a garden center --/
structure GardenPurchase where
  pansy_count : ℕ
  pansy_price : ℚ
  hydrangea_count : ℕ
  hydrangea_price : ℚ
  petunia_count : ℕ
  petunia_price : ℚ
  paid_amount : ℚ
  change_received : ℚ

/-- Calculates the discount offered by the garden center --/
def calculate_discount (purchase : GardenPurchase) : ℚ :=
  let total_cost := purchase.pansy_count * purchase.pansy_price +
                    purchase.hydrangea_count * purchase.hydrangea_price +
                    purchase.petunia_count * purchase.petunia_price
  let amount_paid := purchase.paid_amount - purchase.change_received
  total_cost - amount_paid

/-- Theorem stating that the discount for the given purchase is $3.00 --/
theorem garden_center_discount :
  let purchase := GardenPurchase.mk 5 2.5 1 12.5 5 1 50 23
  calculate_discount purchase = 3 := by sorry

end garden_center_discount_l3195_319559


namespace divisible_by_four_or_seven_l3195_319578

theorem divisible_by_four_or_seven : ∃ (S : Finset ℕ), 
  (∀ n ∈ S, 1 ≤ n ∧ n ≤ 60 ∧ (4 ∣ n ∨ 7 ∣ n)) ∧ 
  (∀ n, 1 ≤ n ∧ n ≤ 60 ∧ (4 ∣ n ∨ 7 ∣ n) → n ∈ S) ∧
  Finset.card S = 21 := by
  sorry

end divisible_by_four_or_seven_l3195_319578


namespace cos_103pi_4_l3195_319530

theorem cos_103pi_4 : Real.cos (103 * Real.pi / 4) = Real.sqrt 2 / 2 := by
  sorry

end cos_103pi_4_l3195_319530


namespace paint_cost_per_kg_l3195_319519

/-- The cost of paint per kg for a cube with given conditions -/
theorem paint_cost_per_kg (coverage : ℝ) (total_cost : ℝ) (side_length : ℝ) :
  coverage = 16 →
  total_cost = 876 →
  side_length = 8 →
  (total_cost / (6 * side_length^2 / coverage)) = 36.5 :=
by sorry

end paint_cost_per_kg_l3195_319519


namespace teacher_age_survey_is_comprehensive_l3195_319586

-- Define the survey types
inductive SurveyType
  | TelevisionLifespan
  | CityIncome
  | StudentMyopia
  | TeacherAge

-- Define a function to determine if a survey is suitable for comprehensive method
def isSuitableForComprehensiveSurvey (survey : SurveyType) : Prop :=
  match survey with
  | .TelevisionLifespan => false  -- Involves destructiveness, must be sampled
  | .CityIncome => false          -- Large number of people, suitable for sampling
  | .StudentMyopia => false       -- Large number of people, suitable for sampling
  | .TeacherAge => true           -- Small number of people, easy to survey comprehensively

-- Theorem statement
theorem teacher_age_survey_is_comprehensive :
  isSuitableForComprehensiveSurvey SurveyType.TeacherAge = true := by
  sorry

end teacher_age_survey_is_comprehensive_l3195_319586


namespace infinitely_many_close_fractions_l3195_319554

theorem infinitely_many_close_fractions (x : ℝ) (hx_pos : x > 0) (hx_irrational : ¬ ∃ (a b : ℤ), x = a / b) :
  ∀ n : ℕ, ∃ p q : ℤ, q > n ∧ q > 0 ∧ |x - (p : ℝ) / q| ≤ 1 / q^2 :=
sorry

end infinitely_many_close_fractions_l3195_319554


namespace floor_plus_x_eq_seventeen_fourths_l3195_319563

theorem floor_plus_x_eq_seventeen_fourths :
  ∃ x : ℚ, (⌊x⌋ : ℚ) + x = 17 / 4 ∧ x = 9 / 4 := by
  sorry

end floor_plus_x_eq_seventeen_fourths_l3195_319563


namespace functional_equation_iff_forms_l3195_319584

/-- The functional equation that f and g must satisfy for all real x and y -/
def functional_equation (f g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, Real.sin x + Real.cos y = f x + f y + g x - g y

/-- The proposed form of function f -/
def f_form (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = (Real.sin x + Real.cos x) / 2

/-- The proposed form of function g, with an arbitrary constant C -/
def g_form (g : ℝ → ℝ) : Prop :=
  ∃ C : ℝ, ∀ x : ℝ, g x = (Real.sin x - Real.cos x) / 2 + C

/-- The main theorem stating the equivalence between the functional equation and the proposed forms of f and g -/
theorem functional_equation_iff_forms (f g : ℝ → ℝ) :
  functional_equation f g ↔ (f_form f ∧ g_form g) :=
sorry

end functional_equation_iff_forms_l3195_319584


namespace max_value_of_2x_plus_y_l3195_319587

theorem max_value_of_2x_plus_y (x y : ℝ) (h : 4 * x^2 + y^2 + x * y = 5) :
  ∃ (M : ℝ), M = 2 * Real.sqrt 2 ∧ ∀ (z : ℝ), 2 * x + y ≤ z → z ≤ M :=
by sorry

end max_value_of_2x_plus_y_l3195_319587


namespace roses_in_garden_l3195_319524

/-- Proves that the number of roses in the garden before cutting is equal to
    the final number of roses in the vase minus the initial number of roses in the vase. -/
theorem roses_in_garden (initial_vase : ℕ) (cut_from_garden : ℕ) (final_vase : ℕ)
  (h1 : initial_vase = 7)
  (h2 : cut_from_garden = 13)
  (h3 : final_vase = 20)
  (h4 : final_vase = initial_vase + cut_from_garden) :
  cut_from_garden = final_vase - initial_vase :=
by sorry

end roses_in_garden_l3195_319524


namespace pens_bought_theorem_l3195_319540

/-- The number of pens bought at the cost price -/
def num_pens_bought : ℕ := 17

/-- The number of pens sold to equal the cost price of the bought pens -/
def num_pens_sold : ℕ := 12

/-- The gain percentage -/
def gain_percentage : ℚ := 40/100

theorem pens_bought_theorem :
  ∀ (cost_price selling_price : ℚ),
  cost_price > 0 →
  selling_price > 0 →
  (num_pens_bought : ℚ) * cost_price = (num_pens_sold : ℚ) * selling_price →
  (selling_price - cost_price) / cost_price = gain_percentage →
  num_pens_bought = 17 :=
by
  sorry

end pens_bought_theorem_l3195_319540


namespace specific_tetrahedron_volume_l3195_319571

/-- A tetrahedron ABCD with specific properties -/
structure Tetrahedron where
  /-- The dihedral angle between faces ABC and BCD in radians -/
  dihedral_angle : ℝ
  /-- The area of triangle ABC -/
  area_ABC : ℝ
  /-- The area of triangle BCD -/
  area_BCD : ℝ
  /-- The length of edge BC -/
  length_BC : ℝ

/-- The volume of the tetrahedron -/
def volume (t : Tetrahedron) : ℝ :=
  sorry

/-- Theorem stating the volume of the specific tetrahedron -/
theorem specific_tetrahedron_volume :
  ∃ t : Tetrahedron,
    t.dihedral_angle = 30 * (π / 180) ∧
    t.area_ABC = 120 ∧
    t.area_BCD = 80 ∧
    t.length_BC = 10 ∧
    volume t = 320 :=
  sorry

end specific_tetrahedron_volume_l3195_319571


namespace fractional_square_gt_floor_square_l3195_319549

theorem fractional_square_gt_floor_square (x : ℝ) (hx : x > 0) :
  (x ^ 2 - ⌊x ^ 2⌋) > (⌊x⌋ ^ 2) ↔ ∃ n : ℤ, Real.sqrt (n ^ 2 + 1) ≤ x ∧ x < n + 1 := by
  sorry

end fractional_square_gt_floor_square_l3195_319549


namespace carlas_classroom_desks_full_l3195_319527

/-- Represents the classroom setup and attendance for Carla's sixth-grade class -/
structure Classroom where
  total_students : ℕ
  restroom_students : ℕ
  rows : ℕ
  desks_per_row : ℕ

/-- Calculates the fraction of desks that are full in the classroom -/
def fraction_of_desks_full (c : Classroom) : ℚ :=
  let absent_students := 3 * c.restroom_students - 1
  let students_in_classroom := c.total_students - absent_students - c.restroom_students
  let total_desks := c.rows * c.desks_per_row
  (students_in_classroom : ℚ) / (total_desks : ℚ)

/-- Theorem stating that the fraction of desks full in Carla's classroom is 2/3 -/
theorem carlas_classroom_desks_full :
  ∃ (c : Classroom), c.total_students = 23 ∧ c.restroom_students = 2 ∧ c.rows = 4 ∧ c.desks_per_row = 6 ∧
  fraction_of_desks_full c = 2 / 3 :=
by
  sorry

end carlas_classroom_desks_full_l3195_319527


namespace cubic_function_properties_l3195_319566

def f (x b c d : ℝ) : ℝ := x^3 + b*x^2 + c*x + d

theorem cubic_function_properties :
  ∀ b c d : ℝ,
  f 0 b c d = 2 →
  (∀ y : ℝ, 6*(-1) - y + 7 = 0 ↔ y = f (-1) b c d) →
  (∀ x : ℝ, f x b c d = x^3 - 3*x^2 - 3*x + 2) ∧
  (∀ x : ℝ, x < 1 - Real.sqrt 2 ∨ x > 1 + Real.sqrt 2 → 
    ∀ h : ℝ, h > 0 → f (x + h) b c d > f x b c d) ∧
  (∀ x : ℝ, 1 - Real.sqrt 2 < x ∧ x < 1 + Real.sqrt 2 → 
    ∀ h : ℝ, h > 0 → f (x + h) b c d < f x b c d) :=
by sorry

end cubic_function_properties_l3195_319566


namespace balloon_count_l3195_319591

theorem balloon_count (colors : Nat) (yellow_taken : Nat) : 
  colors = 4 → yellow_taken = 84 → colors * yellow_taken * 2 = 672 := by
  sorry

end balloon_count_l3195_319591


namespace negation_of_proposition_l3195_319514

theorem negation_of_proposition (p : ℝ → Prop) :
  (¬ ∀ x : ℝ, x^2 + x + 1 > 0) ↔ (∃ x : ℝ, x^2 + x + 1 ≤ 0) :=
by sorry

end negation_of_proposition_l3195_319514


namespace second_feeding_maggots_l3195_319542

/-- Given the total number of maggots served and the number of maggots in the first feeding,
    calculate the number of maggots in the second feeding. -/
def maggots_in_second_feeding (total_maggots : ℕ) (first_feeding : ℕ) : ℕ :=
  total_maggots - first_feeding

/-- Theorem stating that given 20 total maggots and 10 maggots in the first feeding,
    the number of maggots in the second feeding is 10. -/
theorem second_feeding_maggots :
  maggots_in_second_feeding 20 10 = 10 := by
  sorry

end second_feeding_maggots_l3195_319542


namespace diamond_calculation_l3195_319577

def diamond (a b : ℚ) : ℚ := a - 1 / b

theorem diamond_calculation :
  let x := diamond (diamond 1 3) 2
  let y := diamond 1 (diamond 3 2)
  x - y = -13/30 := by sorry

end diamond_calculation_l3195_319577


namespace intersection_and_union_when_a_is_neg_one_intersection_equals_B_iff_l3195_319594

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 4*x - 5 ≥ 0}
def B (a : ℝ) : Set ℝ := {x | 2*a ≤ x ∧ x ≤ a + 2}

-- Theorem for part 1
theorem intersection_and_union_when_a_is_neg_one :
  (A ∩ B (-1)) = {x | -2 ≤ x ∧ x ≤ -1} ∧
  (A ∪ B (-1)) = {x | x ≤ 1 ∨ x ≥ 5} := by sorry

-- Theorem for part 2
theorem intersection_equals_B_iff :
  ∀ a : ℝ, (A ∩ B a = B a) ↔ (a > 2 ∨ a ≤ -3) := by sorry

end intersection_and_union_when_a_is_neg_one_intersection_equals_B_iff_l3195_319594


namespace ratio_x_to_y_l3195_319557

theorem ratio_x_to_y (x y : ℚ) (h : (7 * x - 4 * y) / (20 * x - 3 * y) = 4 / 9) :
  x / y = -24 / 17 := by sorry

end ratio_x_to_y_l3195_319557


namespace jons_textbooks_weight_l3195_319534

theorem jons_textbooks_weight (brandon_weight : ℝ) (jon_weight : ℝ) : 
  brandon_weight = 8 → jon_weight = 3 * brandon_weight → jon_weight = 24 := by
  sorry

end jons_textbooks_weight_l3195_319534


namespace alpha_values_l3195_319558

theorem alpha_values (α : ℂ) (h1 : α ≠ 1) 
  (h2 : Complex.abs (α^2 - 1) = 2 * Complex.abs (α - 1))
  (h3 : Complex.abs (α^4 - 1) = 4 * Complex.abs (α - 1)) :
  α = Complex.I * Real.sqrt 3 ∨ α = -Complex.I * Real.sqrt 3 :=
sorry

end alpha_values_l3195_319558


namespace sqrt_equation_solution_l3195_319585

theorem sqrt_equation_solution (t : ℝ) : 
  Real.sqrt (3 * Real.sqrt (2 * t - 1)) = (12 - 2 * t) ^ (1/4) → t = 21/20 :=
by sorry

end sqrt_equation_solution_l3195_319585


namespace angle_A_measure_l3195_319597

/-- Given a complex geometric figure with the following properties:
    - Angle B is 120°
    - Angle B forms a linear pair with another angle
    - A triangle adjacent to this setup contains an angle of 50°
    - A small triangle connected to one vertex of the larger triangle has an angle of 45°
    - This small triangle shares a vertex with angle A
    Prove that the measure of angle A is 65° -/
theorem angle_A_measure (B : Real) (adjacent_angle : Real) (large_triangle_angle : Real) (small_triangle_angle : Real) (A : Real) :
  B = 120 →
  B + adjacent_angle = 180 →
  large_triangle_angle = 50 →
  small_triangle_angle = 45 →
  A + small_triangle_angle + (180 - B - large_triangle_angle) = 180 →
  A = 65 := by
  sorry


end angle_A_measure_l3195_319597


namespace greatest_prime_factor_of_4_power_minus_2_power_29_l3195_319518

theorem greatest_prime_factor_of_4_power_minus_2_power_29 (n : ℕ) : 
  (∃ (p : ℕ), Nat.Prime p ∧ p ∣ (4^n - 2^29) ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ (4^n - 2^29) → q ≤ p) ∧
  (∀ (q : ℕ), Nat.Prime q → q ∣ (4^n - 2^29) → q ≤ 31) ∧
  (31 ∣ (4^n - 2^29)) →
  n = 17 :=
by sorry


end greatest_prime_factor_of_4_power_minus_2_power_29_l3195_319518


namespace range_of_a_l3195_319583

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, 4^x - (a+3)*2^x + 1 = 0) → a ≥ -1 := by
sorry

end range_of_a_l3195_319583


namespace x_value_proof_l3195_319580

theorem x_value_proof (x : ℝ) (h : 9 / (x^2) = x / 81) : x = 9 := by
  sorry

end x_value_proof_l3195_319580


namespace two_point_distribution_max_value_l3195_319538

/-- A random variable following a two-point distribution -/
structure TwoPointDistribution where
  p : ℝ
  hp : 0 < p ∧ p < 1

/-- The expected value of a two-point distribution -/
def expectedValue (ξ : TwoPointDistribution) : ℝ := ξ.p

/-- The variance of a two-point distribution -/
def variance (ξ : TwoPointDistribution) : ℝ := ξ.p * (1 - ξ.p)

/-- The theorem stating the maximum value of (2D(ξ)-1)/E(ξ) for a two-point distribution -/
theorem two_point_distribution_max_value (ξ : TwoPointDistribution) :
  (∃ (c : ℝ), ∀ (η : TwoPointDistribution), (2 * variance η - 1) / expectedValue η ≤ c) ∧
  (∃ (ξ_max : TwoPointDistribution), (2 * variance ξ_max - 1) / expectedValue ξ_max = 2 - 2 * Real.sqrt 2) :=
sorry

end two_point_distribution_max_value_l3195_319538


namespace statement_b_incorrect_l3195_319579

/-- A predicate representing the conditions for a point to be on a locus -/
def LocusCondition (α : Type*) := α → Prop

/-- A predicate representing the geometric locus itself -/
def GeometricLocus (α : Type*) := α → Prop

/-- Statement B: If a point is on the locus, then it satisfies the conditions;
    however, there may be points not on the locus that also satisfy these conditions. -/
def StatementB (α : Type*) (locus : GeometricLocus α) (condition : LocusCondition α) : Prop :=
  (∀ x : α, locus x → condition x) ∧
  ∃ y : α, condition y ∧ ¬locus y

/-- Theorem stating that Statement B is an incorrect method for defining a geometric locus -/
theorem statement_b_incorrect (α : Type*) :
  ¬∀ (locus : GeometricLocus α) (condition : LocusCondition α),
    StatementB α locus condition ↔ (∀ x : α, locus x ↔ condition x) :=
sorry

end statement_b_incorrect_l3195_319579


namespace nine_chapters_problem_l3195_319544

theorem nine_chapters_problem (x y : ℕ) :
  y = 2*x + 9 ∧ y = 3*(x - 2) ↔ 
  (∃ (filled_cars : ℕ), 
    x = filled_cars + 2 ∧ 
    y = 3 * filled_cars) :=
sorry

end nine_chapters_problem_l3195_319544


namespace range_of_a_l3195_319562

-- Define the propositions p and q
def p (x a : ℝ) : Prop := -4 < x - a ∧ x - a < 4
def q (x : ℝ) : Prop := (x - 2) * (3 - x) > 0

-- Define the theorem
theorem range_of_a (a : ℝ) :
  (∀ x, q x → p x a) →  -- q is a sufficient condition for p
  -1 ≤ a ∧ a ≤ 6 :=
by sorry

end range_of_a_l3195_319562


namespace original_number_proof_l3195_319546

theorem original_number_proof : ∃ x : ℝ, x * 0.74 = 1.9832 ∧ x = 2.68 := by
  sorry

end original_number_proof_l3195_319546


namespace inequality_proof_l3195_319570

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^2 / b + b^2 / c + c^2 / a ≥ a + b + c + 4 * (a - b)^2 / (a + b + c) := by
  sorry

end inequality_proof_l3195_319570


namespace exp_ge_x_plus_one_l3195_319506

theorem exp_ge_x_plus_one : ∀ x : ℝ, Real.exp x ≥ x + 1 := by
  sorry

end exp_ge_x_plus_one_l3195_319506


namespace spherical_distance_for_pi_over_six_l3195_319593

/-- The spherical distance between two points on a sphere's surface -/
def spherical_distance (R : ℝ) (angle : ℝ) : ℝ := R * angle

/-- Theorem: The spherical distance between two points A and B on a sphere with radius R,
    where the angle AOB is π/6, is equal to (π/6)R -/
theorem spherical_distance_for_pi_over_six (R : ℝ) (h : R > 0) :
  spherical_distance R (π/6) = (π/6) * R := by sorry

end spherical_distance_for_pi_over_six_l3195_319593


namespace hyperbola_focal_length_l3195_319511

/-- The focal length of a hyperbola with equation x²/a² - y² = 1,
    where one of its asymptotes is perpendicular to the line 3x + y + 1 = 0,
    is equal to 2√10. -/
theorem hyperbola_focal_length (a : ℝ) :
  (∃ (x y : ℝ), x^2 / a^2 - y^2 = 1) →
  (∃ (m : ℝ), m * (-1/3) = -1 ∧ y = m * x) →
  2 * Real.sqrt (1 + a^2) = 2 * Real.sqrt 10 :=
by sorry

end hyperbola_focal_length_l3195_319511


namespace reciprocal_of_negative_one_fifth_l3195_319581

theorem reciprocal_of_negative_one_fifth : 
  ∀ x : ℚ, x = -1/5 → (∃ y : ℚ, y * x = 1 ∧ y = -5) :=
by sorry

end reciprocal_of_negative_one_fifth_l3195_319581


namespace meshed_gears_angular_velocity_ratio_l3195_319515

structure Gear where
  teeth : ℕ
  angularVelocity : ℝ

/-- The ratio of angular velocities for three meshed gears is proportional to the product of the other two gears' teeth counts. -/
theorem meshed_gears_angular_velocity_ratio 
  (A B C : Gear) 
  (h_mesh : A.angularVelocity * A.teeth = B.angularVelocity * B.teeth ∧ 
            B.angularVelocity * B.teeth = C.angularVelocity * C.teeth) :
  A.angularVelocity / (B.teeth * C.teeth) = 
  B.angularVelocity / (A.teeth * C.teeth) ∧
  B.angularVelocity / (A.teeth * C.teeth) = 
  C.angularVelocity / (A.teeth * B.teeth) :=
by sorry

end meshed_gears_angular_velocity_ratio_l3195_319515


namespace percentage_spent_l3195_319543

theorem percentage_spent (initial_amount remaining_amount : ℝ) 
  (h1 : initial_amount = 1200)
  (h2 : remaining_amount = 840) :
  (initial_amount - remaining_amount) / initial_amount * 100 = 30 := by
  sorry

end percentage_spent_l3195_319543


namespace P_not_in_second_quadrant_l3195_319548

/-- A point is in the second quadrant if its x-coordinate is negative and its y-coordinate is positive -/
def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- The coordinates of point P as a function of m -/
def P (m : ℝ) : ℝ × ℝ := (m^2 + m, m - 1)

/-- Theorem stating that P(m) cannot be in the second quadrant for any real m -/
theorem P_not_in_second_quadrant (m : ℝ) : ¬ second_quadrant (P m).1 (P m).2 := by
  sorry

end P_not_in_second_quadrant_l3195_319548


namespace tangent_circle_radius_l3195_319523

/-- The radius of a circle tangent to four semicircles in a square -/
theorem tangent_circle_radius (s : ℝ) (h : s = 4) : 
  let r := 2 * (Real.sqrt 2 - 1)
  let semicircle_radius := s / 2
  let square_diagonal := s * Real.sqrt 2
  r = square_diagonal / 2 - semicircle_radius :=
by sorry

end tangent_circle_radius_l3195_319523


namespace min_draw_for_20_balls_l3195_319576

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  red : Nat
  green : Nat
  white : Nat
  blue : Nat

/-- The minimum number of balls to draw to ensure at least 20 of a single color -/
def minDrawToEnsure20 (counts : BallCounts) : Nat :=
  sorry

/-- The theorem stating the minimum number of balls to draw -/
theorem min_draw_for_20_balls (counts : BallCounts) 
  (h1 : counts.red = 23)
  (h2 : counts.green = 24)
  (h3 : counts.white = 12)
  (h4 : counts.blue = 21) :
  minDrawToEnsure20 counts = 70 :=
sorry

end min_draw_for_20_balls_l3195_319576


namespace largest_angle_in_isosceles_triangle_l3195_319533

-- Define an isosceles triangle with one angle of 50°
def IsoscelesTriangle (a b c : ℝ) : Prop :=
  a + b + c = 180 ∧ a = b ∧ a = 50

-- Theorem statement
theorem largest_angle_in_isosceles_triangle 
  {a b c : ℝ} (h : IsoscelesTriangle a b c) : 
  max a (max b c) = 80 := by
  sorry

end largest_angle_in_isosceles_triangle_l3195_319533


namespace arithmetic_sequence_middle_term_l3195_319504

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In an arithmetic sequence {aₙ}, if a₁ + a₉ = 10, then a₅ = 5 -/
theorem arithmetic_sequence_middle_term 
  (a : ℕ → ℝ) 
  (h_arith : ArithmeticSequence a) 
  (h_sum : a 1 + a 9 = 10) : 
  a 5 = 5 := by
  sorry

end arithmetic_sequence_middle_term_l3195_319504


namespace first_ring_hexagons_fiftieth_ring_hexagons_nth_ring_hexagons_l3195_319528

/-- The number of hexagons in the nth ring around a central hexagon in a hexagonal tiling -/
def hexagons_in_nth_ring (n : ℕ) : ℕ := 6 * n

/-- The first ring contains 6 hexagons -/
theorem first_ring_hexagons : hexagons_in_nth_ring 1 = 6 := by sorry

/-- The 50th ring contains 300 hexagons -/
theorem fiftieth_ring_hexagons : hexagons_in_nth_ring 50 = 300 := by sorry

/-- For any natural number n, the nth ring contains 6n hexagons -/
theorem nth_ring_hexagons (n : ℕ) : hexagons_in_nth_ring n = 6 * n := by sorry

end first_ring_hexagons_fiftieth_ring_hexagons_nth_ring_hexagons_l3195_319528


namespace large_jar_capacity_l3195_319595

/-- Given a shelf of jars with the following properties:
  * There are 100 total jars
  * Small jars hold 3 liters each
  * The total capacity of all jars is 376 liters
  * There are 62 small jars
  This theorem proves that each large jar holds 5 liters. -/
theorem large_jar_capacity (total_jars : ℕ) (small_jar_capacity : ℕ) (total_capacity : ℕ) (small_jars : ℕ)
  (h1 : total_jars = 100)
  (h2 : small_jar_capacity = 3)
  (h3 : total_capacity = 376)
  (h4 : small_jars = 62) :
  (total_capacity - small_jars * small_jar_capacity) / (total_jars - small_jars) = 5 := by
  sorry

end large_jar_capacity_l3195_319595


namespace bananas_multiple_of_three_l3195_319509

/-- Represents the number of fruit baskets that can be made -/
def num_baskets : ℕ := 3

/-- Represents the number of oranges Peter has -/
def oranges : ℕ := 18

/-- Represents the number of pears Peter has -/
def pears : ℕ := 27

/-- Represents the number of bananas Peter has -/
def bananas : ℕ := sorry

/-- Theorem stating that the number of bananas must be a multiple of 3 -/
theorem bananas_multiple_of_three :
  ∃ k : ℕ, bananas = 3 * k ∧
  oranges % num_baskets = 0 ∧
  pears % num_baskets = 0 ∧
  bananas % num_baskets = 0 :=
sorry

end bananas_multiple_of_three_l3195_319509


namespace divisibility_by_eleven_l3195_319501

theorem divisibility_by_eleven (n : ℕ) (a b c d e : ℕ) 
  (h1 : n = a * 10000 + b * 1000 + c * 100 + d * 10 + e)
  (h2 : a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10) : 
  n ≡ (a + c + e) - (b + d) [MOD 11] := by
  sorry

end divisibility_by_eleven_l3195_319501


namespace modulus_of_complex_fraction_l3195_319547

theorem modulus_of_complex_fraction (z : ℂ) : z = (1 + I) / (1 - I) → Complex.abs z = 1 := by
  sorry

end modulus_of_complex_fraction_l3195_319547


namespace abs_plus_square_zero_implies_sum_l3195_319565

theorem abs_plus_square_zero_implies_sum (x y : ℝ) :
  |x + 3| + (2*y - 5)^2 = 0 → x + 2*y = 2 := by
  sorry

end abs_plus_square_zero_implies_sum_l3195_319565


namespace last_digit_of_one_third_to_tenth_l3195_319508

theorem last_digit_of_one_third_to_tenth (n : ℕ) : 
  (1 : ℚ) / 3^10 * 10^n % 10 = 5 :=
sorry

end last_digit_of_one_third_to_tenth_l3195_319508


namespace prime_divisor_implies_equal_l3195_319553

theorem prime_divisor_implies_equal (m n : ℕ) : 
  Prime (m + n + 1) → 
  (m + n + 1) ∣ (2 * (m^2 + n^2) - 1) → 
  m = n :=
by sorry

end prime_divisor_implies_equal_l3195_319553


namespace billy_coins_l3195_319500

/-- Given the number of piles of quarters and dimes, and the number of coins per pile,
    calculate the total number of coins. -/
def total_coins (quarter_piles dime_piles coins_per_pile : ℕ) : ℕ :=
  (quarter_piles + dime_piles) * coins_per_pile

/-- Theorem stating that with 2 piles of quarters, 3 piles of dimes, and 4 coins per pile,
    the total number of coins is 20. -/
theorem billy_coins : total_coins 2 3 4 = 20 := by
  sorry

end billy_coins_l3195_319500


namespace function_max_min_sum_l3195_319550

theorem function_max_min_sum (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  let f := fun x : ℝ => (5 * a^x + 1) / (a^x - 1) + Real.log (Real.sqrt (1 + x^2) - x)
  ∃ (M N : ℝ), (∀ x, f x ≤ M) ∧ (∀ x, N ≤ f x) ∧ M + N = 4 := by
  sorry

end function_max_min_sum_l3195_319550


namespace apple_sales_leftover_l3195_319521

/-- The number of apples left over after selling all possible baskets -/
def leftover_apples (oliver patricia quentin basket_size : ℕ) : ℕ :=
  (oliver + patricia + quentin) % basket_size

theorem apple_sales_leftover :
  leftover_apples 58 36 15 12 = 1 := by
  sorry

end apple_sales_leftover_l3195_319521


namespace line_relationship_exclusive_line_relationship_unique_l3195_319592

-- Define a line in 3D space
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

-- Define the relationship between two lines
inductive LineRelationship
  | Parallel
  | Skew
  | Intersecting

-- Define a function to determine the relationship between two lines
def determineRelationship (l1 l2 : Line3D) : LineRelationship :=
  sorry

-- Theorem: Two lines must have exactly one of the three relationships
theorem line_relationship_exclusive (l1 l2 : Line3D) :
  (determineRelationship l1 l2 = LineRelationship.Parallel) ∨
  (determineRelationship l1 l2 = LineRelationship.Skew) ∨
  (determineRelationship l1 l2 = LineRelationship.Intersecting) :=
  sorry

-- Theorem: The relationship between two lines is unique
theorem line_relationship_unique (l1 l2 : Line3D) :
  ¬((determineRelationship l1 l2 = LineRelationship.Parallel) ∧
    (determineRelationship l1 l2 = LineRelationship.Skew)) ∧
  ¬((determineRelationship l1 l2 = LineRelationship.Parallel) ∧
    (determineRelationship l1 l2 = LineRelationship.Intersecting)) ∧
  ¬((determineRelationship l1 l2 = LineRelationship.Skew) ∧
    (determineRelationship l1 l2 = LineRelationship.Intersecting)) :=
  sorry

end line_relationship_exclusive_line_relationship_unique_l3195_319592


namespace more_girls_than_boys_l3195_319588

theorem more_girls_than_boys (total_students : ℕ) (boys : ℕ) 
  (h1 : total_students = 466)
  (h2 : boys = 127)
  (h3 : boys < total_students - boys) :
  total_students - boys - boys = 212 := by
  sorry

end more_girls_than_boys_l3195_319588


namespace sum_of_coordinates_B_l3195_319545

/-- Given points A(0, 0) and B(x, 3) where the slope of AB is 3/4, 
    prove that the sum of B's coordinates is 7. -/
theorem sum_of_coordinates_B (x : ℝ) : 
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (x, 3)
  (3 - 0) / (x - 0) = 3 / 4 →
  x + 3 = 7 := by
sorry

end sum_of_coordinates_B_l3195_319545


namespace ed_length_l3195_319552

/-- Given five points in a plane with specific distances between them, prove that ED = 74 -/
theorem ed_length (A B C D E : EuclideanSpace ℝ (Fin 2)) 
  (h_AB : dist A B = 12)
  (h_BC : dist B C = 50)
  (h_CD : dist C D = 38)
  (h_AD : dist A D = 100)
  (h_BE : dist B E = 30)
  (h_CE : dist C E = 40) :
  dist E D = 74 := by
  sorry

end ed_length_l3195_319552


namespace line_equation_proof_l3195_319510

-- Define the line l
def line_l : Set (ℝ × ℝ) := {(x, y) | 4*x - 3*y - 1 = 0}

-- Define the given line
def given_line : Set (ℝ × ℝ) := {(x, y) | 3*x + 4*y - 3 = 0}

-- Define the point A
def point_A : ℝ × ℝ := (-2, -3)

theorem line_equation_proof :
  -- Line l passes through point A
  point_A ∈ line_l ∧
  -- Line l is perpendicular to the given line
  (∀ (p q : ℝ × ℝ), p ∈ line_l → q ∈ line_l → p ≠ q →
    ∀ (r s : ℝ × ℝ), r ∈ given_line → s ∈ given_line → r ≠ s →
      ((p.1 - q.1) * (r.1 - s.1) + (p.2 - q.2) * (r.2 - s.2) = 0)) :=
by sorry

end line_equation_proof_l3195_319510


namespace fraction_to_decimal_l3195_319568

theorem fraction_to_decimal : (3 : ℚ) / 60 = (5 : ℚ) / 100 := by sorry

end fraction_to_decimal_l3195_319568


namespace largest_argument_l3195_319529

-- Define the complex number z
variable (z : ℂ)

-- Define the condition |z - 10i| = 5√2
def satisfies_condition (z : ℂ) : Prop :=
  Complex.abs (z - Complex.I * 10) = 5 * Real.sqrt 2

-- Define the theorem
theorem largest_argument :
  ∃ (z : ℂ), satisfies_condition z ∧
  ∀ (w : ℂ), satisfies_condition w → Complex.arg w ≤ Complex.arg z ∧
  z = -5 + 5 * Complex.I :=
sorry

end largest_argument_l3195_319529


namespace skew_lines_cannot_both_project_to_points_l3195_319564

/-- Two lines in 3D space are skew -/
def are_skew (l1 l2 : Line3) : Prop := sorry

/-- A line in 3D space -/
def Line3 : Type := sorry

/-- A plane in 3D space -/
def Plane3 : Type := sorry

/-- The projection of a line onto a plane -/
def project_line_to_plane (l : Line3) (p : Plane3) : Set Point := sorry

/-- A line is perpendicular to a plane -/
def perpendicular_to_plane (l : Line3) (p : Plane3) : Prop := sorry

theorem skew_lines_cannot_both_project_to_points (a b : Line3) (α : Plane3) 
  (h_skew : are_skew a b) :
  ¬(∃ (pa pb : Point), project_line_to_plane a α = {pa} ∧ project_line_to_plane b α = {pb}) :=
sorry

end skew_lines_cannot_both_project_to_points_l3195_319564


namespace ali_bookshelf_problem_l3195_319502

theorem ali_bookshelf_problem (x : ℕ) : 
  (x / 2 : ℕ) + (x / 3 : ℕ) + 3 + 7 = x → (x / 2 : ℕ) = 30 := by
  sorry

end ali_bookshelf_problem_l3195_319502


namespace instrumental_measurements_insufficient_l3195_319517

-- Define the concept of instrumental measurements
def InstrumentalMeasurement : Type := Unit

-- Define the concept of general geometric statements
def GeneralGeometricStatement : Type := Unit

-- Define the property of being approximate
def is_approximate (m : InstrumentalMeasurement) : Prop := sorry

-- Define the property of applying to infinite configurations
def applies_to_infinite_configurations (s : GeneralGeometricStatement) : Prop := sorry

-- Define the property of being performed on a finite number of instances
def performed_on_finite_instances (m : InstrumentalMeasurement) : Prop := sorry

-- Theorem stating that instrumental measurements are insufficient to justify general geometric statements
theorem instrumental_measurements_insufficient 
  (m : InstrumentalMeasurement) 
  (s : GeneralGeometricStatement) : 
  is_approximate m → 
  applies_to_infinite_configurations s → 
  performed_on_finite_instances m → 
  ¬(∃ (justification : Unit), True) := by sorry

end instrumental_measurements_insufficient_l3195_319517


namespace circle_radius_with_area_four_l3195_319573

theorem circle_radius_with_area_four (r : ℝ) :
  r > 0 → π * r^2 = 4 → r = 2 / Real.sqrt π := by sorry

end circle_radius_with_area_four_l3195_319573


namespace gcd_210_378_l3195_319561

theorem gcd_210_378 : Nat.gcd 210 378 = 42 := by
  sorry

end gcd_210_378_l3195_319561


namespace total_cost_calculation_l3195_319526

/-- Calculates the total cost of production given fixed cost, marginal cost, and number of products. -/
def totalCost (fixedCost marginalCost : ℕ) (numProducts : ℕ) : ℕ :=
  fixedCost + marginalCost * numProducts

/-- Proves that the total cost of producing 20 products is $16,000, given a fixed cost of $12,000 and a marginal cost of $200 per product. -/
theorem total_cost_calculation :
  totalCost 12000 200 20 = 16000 := by
  sorry

end total_cost_calculation_l3195_319526


namespace equation_solution_l3195_319522

theorem equation_solution :
  ∀ x : ℝ, x ≠ 1 →
  (3 / (x - 1) = 5 + 3 * x / (1 - x)) ↔ x = 4 := by
sorry

end equation_solution_l3195_319522


namespace equation_solution_range_l3195_319555

theorem equation_solution_range (x m : ℝ) : 9^x + 4 * 3^x - m = 0 → m > 0 := by
  sorry

end equation_solution_range_l3195_319555


namespace square_sum_given_conditions_l3195_319532

theorem square_sum_given_conditions (x y : ℝ) 
  (h1 : (x + y)^2 = 4) 
  (h2 : x * y = -1) : 
  x^2 + y^2 = 6 := by
sorry

end square_sum_given_conditions_l3195_319532


namespace division_result_l3195_319589

theorem division_result : ∃ (q : ℕ), 1254 = 6 * q → q = 209 := by
  sorry

end division_result_l3195_319589
