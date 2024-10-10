import Mathlib

namespace coefficient_a2_value_l2324_232490

/-- Given a complex number z and a polynomial expansion of (x-z)^4,
    prove that the coefficient of x^2 is -3 + 3√3i. -/
theorem coefficient_a2_value (z : ℂ) (a₀ a₁ a₂ a₃ a₄ : ℂ) :
  z = (1/2 : ℂ) + (Complex.I * Real.sqrt 3) / 2 →
  (fun x : ℂ ↦ (x - z)^4) = (fun x : ℂ ↦ a₀*x^4 + a₁*x^3 + a₂*x^2 + a₃*x + a₄) →
  a₂ = -3 + Complex.I * (3 * Real.sqrt 3) := by
sorry

end coefficient_a2_value_l2324_232490


namespace triangle_area_with_median_l2324_232426

/-- Given a triangle with two sides of length 1 and √15, and a median of length 2 to the third side,
    the area of the triangle is √15/2. -/
theorem triangle_area_with_median (a b c : ℝ) (m : ℝ) (h1 : a = 1) (h2 : b = Real.sqrt 15) (h3 : m = 2)
    (hm : m^2 = (2*a^2 + 2*b^2 - c^2) / 4) : (a * b) / 2 = Real.sqrt 15 / 2 := by
  sorry

end triangle_area_with_median_l2324_232426


namespace inequality_preservation_l2324_232403

theorem inequality_preservation (a b c : ℝ) (h : a > b) :
  a / (c^2 + 1) > b / (c^2 + 1) := by
  sorry

end inequality_preservation_l2324_232403


namespace max_value_of_shui_l2324_232472

/-- Represents the digits assigned to each Chinese character -/
structure ChineseDigits where
  jin : Fin 8
  xin : Fin 8
  li : Fin 8
  ke : Fin 8
  ba : Fin 8
  shan : Fin 8
  qiong : Fin 8
  shui : Fin 8

/-- All digits are unique -/
def all_unique (d : ChineseDigits) : Prop :=
  d.jin ≠ d.xin ∧ d.jin ≠ d.li ∧ d.jin ≠ d.ke ∧ d.jin ≠ d.ba ∧ d.jin ≠ d.shan ∧ d.jin ≠ d.qiong ∧ d.jin ≠ d.shui ∧
  d.xin ≠ d.li ∧ d.xin ≠ d.ke ∧ d.xin ≠ d.ba ∧ d.xin ≠ d.shan ∧ d.xin ≠ d.qiong ∧ d.xin ≠ d.shui ∧
  d.li ≠ d.ke ∧ d.li ≠ d.ba ∧ d.li ≠ d.shan ∧ d.li ≠ d.qiong ∧ d.li ≠ d.shui ∧
  d.ke ≠ d.ba ∧ d.ke ≠ d.shan ∧ d.ke ≠ d.qiong ∧ d.ke ≠ d.shui ∧
  d.ba ≠ d.shan ∧ d.ba ≠ d.qiong ∧ d.ba ≠ d.shui ∧
  d.shan ≠ d.qiong ∧ d.shan ≠ d.shui ∧
  d.qiong ≠ d.shui

/-- The sum of digits in each phrase is 19 -/
def sum_is_19 (d : ChineseDigits) : Prop :=
  d.jin.val + d.jin.val + d.xin.val + d.li.val = 19 ∧
  d.ke.val + d.ba.val + d.shan.val = 19 ∧
  d.shan.val + d.qiong.val + d.shui.val + d.jin.val = 19

/-- The ordering constraint: 尽 > 山 > 力 -/
def ordering_constraint (d : ChineseDigits) : Prop :=
  d.jin > d.shan ∧ d.shan > d.li

theorem max_value_of_shui (d : ChineseDigits) 
  (h1 : all_unique d) 
  (h2 : sum_is_19 d) 
  (h3 : ordering_constraint d) : 
  d.shui.val ≤ 7 :=
sorry

end max_value_of_shui_l2324_232472


namespace intersection_of_A_and_B_l2324_232406

-- Define the sets A and B
def A : Set ℝ := {x | x > 0}
def B : Set ℝ := {y | ∃ x, y = 1 - x^2}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x | 0 < x ∧ x ≤ 1} := by
  sorry

end intersection_of_A_and_B_l2324_232406


namespace distance_borya_vasya_l2324_232439

/-- Represents the positions of houses along a road -/
structure HousePositions where
  andrey : ℝ
  borya : ℝ
  vasya : ℝ
  gena : ℝ

/-- The race setup along the road -/
def RaceSetup (h : HousePositions) : Prop :=
  h.gena - h.andrey = 2450 ∧
  h.vasya - h.andrey = h.gena - h.borya ∧
  (h.borya + h.gena) / 2 - (h.andrey + h.vasya) / 2 = 1000

theorem distance_borya_vasya (h : HousePositions) (race : RaceSetup h) :
  h.vasya - h.borya = 450 := by
  sorry

end distance_borya_vasya_l2324_232439


namespace perpendicular_bisector_value_l2324_232410

/-- The perpendicular bisector of a line segment passing through two points. -/
structure PerpendicularBisector where
  -- The equation of the line: x + y = b
  b : ℝ
  -- The two points defining the line segment
  p1 : ℝ × ℝ := (2, 4)
  p2 : ℝ × ℝ := (6, 8)
  -- The condition that the line is a perpendicular bisector
  is_perp_bisector : b = p1.1 + p1.2 + p2.1 + p2.2

/-- The value of b for the perpendicular bisector of the line segment from (2,4) to (6,8) is 10. -/
theorem perpendicular_bisector_value : 
  ∀ (pb : PerpendicularBisector), pb.b = 10 := by
  sorry

end perpendicular_bisector_value_l2324_232410


namespace t_has_six_values_l2324_232428

/-- A type representing single-digit positive integers -/
def SingleDigit := {n : ℕ // 1 ≤ n ∧ n ≤ 9}

/-- The theorem stating that t can have 6 distinct values -/
theorem t_has_six_values 
  (p q r s t : SingleDigit) 
  (h1 : p.val - q.val = r.val)
  (h2 : r.val - s.val = t.val)
  (h3 : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ 
        q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ 
        r ≠ s ∧ r ≠ t ∧ 
        s ≠ t) :
  ∃ (values : Finset ℕ), values.card = 6 ∧ t.val ∈ values ∧ 
  ∀ x, x ∈ values → ∃ (p' q' r' s' t' : SingleDigit), 
    p'.val - q'.val = r'.val ∧ 
    r'.val - s'.val = t'.val ∧ 
    t'.val = x ∧
    p' ≠ q' ∧ p' ≠ r' ∧ p' ≠ s' ∧ p' ≠ t' ∧ 
    q' ≠ r' ∧ q' ≠ s' ∧ q' ≠ t' ∧ 
    r' ≠ s' ∧ r' ≠ t' ∧ 
    s' ≠ t' := by
  sorry

end t_has_six_values_l2324_232428


namespace shared_friends_l2324_232427

theorem shared_friends (james_friends : ℕ) (john_friends : ℕ) (combined_list : ℕ) :
  james_friends = 75 →
  john_friends = 3 * james_friends →
  combined_list = 275 →
  james_friends + john_friends - combined_list = 25 := by
sorry

end shared_friends_l2324_232427


namespace marias_average_balance_l2324_232476

/-- Given Maria's savings account balances for four months, prove that the average monthly balance is $300. -/
theorem marias_average_balance (jan feb mar apr : ℕ) 
  (h_jan : jan = 150)
  (h_feb : feb = 300)
  (h_mar : mar = 450)
  (h_apr : apr = 300) :
  (jan + feb + mar + apr) / 4 = 300 := by
  sorry

end marias_average_balance_l2324_232476


namespace remaining_child_meal_capacity_l2324_232487

/-- Represents the meal capacity and consumption for a trekking group -/
structure TrekkingMeal where
  total_adults : ℕ
  adults_fed : ℕ
  adult_meal_capacity : ℕ
  child_meal_capacity : ℕ
  remaining_child_capacity : ℕ

/-- Theorem stating that given the conditions of the trekking meal,
    the number of children that can be catered with the remaining food is 36 -/
theorem remaining_child_meal_capacity
  (meal : TrekkingMeal)
  (h1 : meal.total_adults = 55)
  (h2 : meal.adult_meal_capacity = 70)
  (h3 : meal.child_meal_capacity = 90)
  (h4 : meal.adults_fed = 42)
  (h5 : meal.remaining_child_capacity = 36) :
  meal.remaining_child_capacity = 36 := by
  sorry


end remaining_child_meal_capacity_l2324_232487


namespace hair_dye_cost_salon_hair_dye_cost_l2324_232411

/-- Calculates the cost of a box of hair dye based on salon revenue and expenses --/
theorem hair_dye_cost (haircut_price perm_price dye_job_price : ℕ)
  (haircuts perms dye_jobs : ℕ) (tips final_amount : ℕ) : ℕ :=
  let total_revenue := haircut_price * haircuts + perm_price * perms + dye_job_price * dye_jobs + tips
  let dye_cost := total_revenue - final_amount
  dye_cost / dye_jobs

/-- Proves that the cost of a box of hair dye is $10 given the problem conditions --/
theorem salon_hair_dye_cost : hair_dye_cost 30 40 60 4 1 2 50 310 = 10 := by
  sorry

end hair_dye_cost_salon_hair_dye_cost_l2324_232411


namespace smallest_m_is_170_l2324_232466

/-- The quadratic equation 10x^2 - mx + 660 = 0 has integral solutions -/
def has_integral_solutions (m : ℤ) : Prop :=
  ∃ x : ℤ, 10 * x^2 - m * x + 660 = 0

/-- 170 is a value of m for which the equation has integral solutions -/
axiom solution_exists : has_integral_solutions 170

/-- For any positive integer less than 170, the equation does not have integral solutions -/
axiom no_smaller_solution : ∀ k : ℤ, 0 < k → k < 170 → ¬(has_integral_solutions k)

theorem smallest_m_is_170 : 
  (∃ m : ℤ, 0 < m ∧ has_integral_solutions m) ∧ 
  (∀ k : ℤ, 0 < k ∧ has_integral_solutions k → 170 ≤ k) :=
sorry

end smallest_m_is_170_l2324_232466


namespace cylinder_band_length_l2324_232491

theorem cylinder_band_length (m k n : ℕ) : 
  (m > 0) → (k > 0) → (n > 0) → 
  (∀ (p : ℕ), Prime p → ¬(p^2 ∣ k)) →
  (2 * (24 * Real.sqrt 3 + 28 * Real.pi) = m * Real.sqrt k + n * Real.pi) →
  m + k + n = 107 := by
  sorry

end cylinder_band_length_l2324_232491


namespace students_taking_one_subject_l2324_232420

theorem students_taking_one_subject (both : ℕ) (geometry : ℕ) (history_only : ℕ)
  (h1 : both = 15)
  (h2 : geometry = 30)
  (h3 : history_only = 18) :
  geometry - both + history_only = 33 := by
sorry

end students_taking_one_subject_l2324_232420


namespace original_price_from_decreased_price_l2324_232418

/-- Proves that if an article's price after a 24% decrease is 684, then its original price was 900. -/
theorem original_price_from_decreased_price (decreased_price : ℝ) (decrease_percentage : ℝ) :
  decreased_price = 684 ∧ decrease_percentage = 24 →
  (1 - decrease_percentage / 100) * 900 = decreased_price := by
  sorry

#check original_price_from_decreased_price

end original_price_from_decreased_price_l2324_232418


namespace sweater_fraction_is_one_fourth_l2324_232412

/-- The amount Leila spent on the sweater -/
def sweater_cost : ℕ := 40

/-- The amount Leila had left after buying jewelry -/
def remaining_money : ℕ := 20

/-- The additional amount Leila spent on jewelry compared to the sweater -/
def jewelry_additional_cost : ℕ := 60

/-- Leila's total initial money -/
def total_money : ℕ := sweater_cost + remaining_money + sweater_cost + jewelry_additional_cost

/-- The fraction of total money spent on the sweater -/
def sweater_fraction : ℚ := sweater_cost / total_money

theorem sweater_fraction_is_one_fourth : sweater_fraction = 1/4 := by
  sorry

end sweater_fraction_is_one_fourth_l2324_232412


namespace ellipse_max_sum_l2324_232478

/-- The maximum value of x + y for points on the ellipse x^2/16 + y^2/9 = 1 is 5 -/
theorem ellipse_max_sum (x y : ℝ) : 
  x^2/16 + y^2/9 = 1 → x + y ≤ 5 ∧ ∃ (x₀ y₀ : ℝ), x₀^2/16 + y₀^2/9 = 1 ∧ x₀ + y₀ = 5 := by
  sorry

#check ellipse_max_sum

end ellipse_max_sum_l2324_232478


namespace water_consumption_proof_l2324_232498

/-- Calculates the total water consumption for horses over a given period. -/
def total_water_consumption (initial_horses : ℕ) (added_horses : ℕ) (drinking_water : ℕ) (bathing_water : ℕ) (days : ℕ) : ℕ :=
  let total_horses := initial_horses + added_horses
  let daily_consumption_per_horse := drinking_water + bathing_water
  let daily_consumption := total_horses * daily_consumption_per_horse
  daily_consumption * days

/-- Proves that the total water consumption for the given conditions is 1568 liters. -/
theorem water_consumption_proof :
  total_water_consumption 3 5 5 2 28 = 1568 := by
  sorry

end water_consumption_proof_l2324_232498


namespace right_triangle_with_inscribed_circle_l2324_232402

theorem right_triangle_with_inscribed_circle (d : ℝ) (h : d > 0) :
  ∃ (a b c : ℝ),
    b = a + d ∧
    c = b + d ∧
    a^2 + b^2 = c^2 ∧
    (a + b - c) / 2 = d :=
by
  sorry

#check right_triangle_with_inscribed_circle

end right_triangle_with_inscribed_circle_l2324_232402


namespace only_postcard_win_is_systematic_l2324_232461

-- Define the type for sampling methods
inductive SamplingMethod
| EmployeeRep
| MarketResearch
| LotteryDraw
| PostcardWin
| ExamAnalysis

-- Define what constitutes systematic sampling
def is_systematic_sampling (method : SamplingMethod) : Prop :=
  match method with
  | SamplingMethod.PostcardWin => True
  | _ => False

-- Theorem stating that only PostcardWin is systematic sampling
theorem only_postcard_win_is_systematic :
  ∀ (method : SamplingMethod),
    is_systematic_sampling method ↔ method = SamplingMethod.PostcardWin :=
by sorry

#check only_postcard_win_is_systematic

end only_postcard_win_is_systematic_l2324_232461


namespace circle_area_through_point_l2324_232441

/-- The area of a circle with center R(5, -2) passing through the point S(-4, 7) is 162π. -/
theorem circle_area_through_point : 
  let R : ℝ × ℝ := (5, -2)
  let S : ℝ × ℝ := (-4, 7)
  let radius := Real.sqrt ((R.1 - S.1)^2 + (R.2 - S.2)^2)
  π * radius^2 = 162 * π := by sorry

end circle_area_through_point_l2324_232441


namespace no_linear_term_implies_m_value_l2324_232482

theorem no_linear_term_implies_m_value (m : ℝ) : 
  (∀ x : ℝ, (x^2 - x + m) * (x - 8) = x^3 - 9*x^2 + 0*x + (-8*m)) → m = -8 := by
  sorry

end no_linear_term_implies_m_value_l2324_232482


namespace alpha_beta_composition_l2324_232451

theorem alpha_beta_composition (α β : ℝ → ℝ) (h_α : ∀ x, α x = 4 * x + 9) (h_β : ∀ x, β x = 7 * x + 6) :
  (∃ x, (α ∘ β) x = 4) ↔ (∃ x, x = -29/28) :=
by sorry

end alpha_beta_composition_l2324_232451


namespace pythagorean_theorem_construct_incommensurable_segments_l2324_232462

-- Define a type for geometric constructions
def GeometricConstruction : Type := Unit

-- Define a function to represent the construction of a segment
def constructSegment (length : ℝ) : GeometricConstruction := sorry

-- Define the Pythagorean theorem
theorem pythagorean_theorem (a b c : ℝ) : 
  a^2 + b^2 = c^2 ↔ ∃ (triangle : GeometricConstruction), true := sorry

-- Theorem stating that √2, √3, and √5 can be geometrically constructed
theorem construct_incommensurable_segments : 
  ∃ (construct_sqrt2 construct_sqrt3 construct_sqrt5 : GeometricConstruction),
    (∃ (a : ℝ), a^2 = 2 ∧ constructSegment a = construct_sqrt2) ∧
    (∃ (b : ℝ), b^2 = 3 ∧ constructSegment b = construct_sqrt3) ∧
    (∃ (c : ℝ), c^2 = 5 ∧ constructSegment c = construct_sqrt5) :=
sorry

end pythagorean_theorem_construct_incommensurable_segments_l2324_232462


namespace total_fruit_count_l2324_232442

theorem total_fruit_count (orange_crates : Nat) (oranges_per_crate : Nat)
                          (nectarine_boxes : Nat) (nectarines_per_box : Nat) :
  orange_crates = 12 →
  oranges_per_crate = 150 →
  nectarine_boxes = 16 →
  nectarines_per_box = 30 →
  orange_crates * oranges_per_crate + nectarine_boxes * nectarines_per_box = 2280 := by
  sorry

end total_fruit_count_l2324_232442


namespace ash_cloud_radius_l2324_232469

/-- Calculates the radius of an ash cloud from a volcano eruption -/
theorem ash_cloud_radius 
  (angle : Real) 
  (vertical_distance : Real) 
  (diameter_factor : Real) 
  (h1 : angle = 60) 
  (h2 : vertical_distance = 300) 
  (h3 : diameter_factor = 18) : 
  ∃ (radius : Real), abs (radius - 10228.74) < 0.01 := by
  sorry

end ash_cloud_radius_l2324_232469


namespace gcd_of_24_and_36_l2324_232458

theorem gcd_of_24_and_36 : Nat.gcd 24 36 = 12 := by
  sorry

end gcd_of_24_and_36_l2324_232458


namespace first_three_digit_in_square_sum_row_l2324_232431

/-- Represents a position in Pascal's triangle -/
structure Position :=
  (row : Nat)
  (col : Nat)

/-- Returns the value at a given position in Pascal's triangle -/
def pascalValue (pos : Position) : Nat :=
  sorry

/-- Returns the sum of a row in Pascal's triangle -/
def rowSum (n : Nat) : Nat :=
  2^n

/-- Checks if a number is a three-digit number -/
def isThreeDigit (n : Nat) : Bool :=
  100 ≤ n ∧ n ≤ 999

/-- The theorem to be proved -/
theorem first_three_digit_in_square_sum_row :
  let pos := Position.mk 16 1
  (isThreeDigit (pascalValue pos)) ∧
  (∃ k : Nat, rowSum 16 = k * k) ∧
  (∀ n < 16, ∀ i ≤ n, ¬(isThreeDigit (pascalValue (Position.mk n i)) ∧ ∃ k : Nat, rowSum n = k * k)) :=
by sorry

end first_three_digit_in_square_sum_row_l2324_232431


namespace minimal_intercept_line_l2324_232496

-- Define a line by its intercepts
structure Line where
  a : ℝ
  b : ℝ
  h1 : a > 0
  h2 : b > 0

-- Define the condition that the line passes through (1, 4)
def passesThrough (l : Line) : Prop :=
  1 / l.a + 4 / l.b = 1

-- Define the sum of intercepts
def sumOfIntercepts (l : Line) : ℝ :=
  l.a + l.b

-- State the theorem
theorem minimal_intercept_line :
  ∃ (l : Line),
    passesThrough l ∧
    ∀ (l' : Line), passesThrough l' → sumOfIntercepts l ≤ sumOfIntercepts l' ∧
    2 * (1 : ℝ) + 4 - 6 = 0 := by
  sorry

end minimal_intercept_line_l2324_232496


namespace train_crossing_time_l2324_232436

/-- Proves that a train 100 meters long, traveling at 36 km/hr, takes 10 seconds to cross an electric pole -/
theorem train_crossing_time : 
  let train_length : ℝ := 100  -- Length of the train in meters
  let train_speed_kmh : ℝ := 36  -- Speed of the train in km/hr
  let train_speed_ms : ℝ := train_speed_kmh * (1000 / 3600)  -- Speed in m/s
  let crossing_time : ℝ := train_length / train_speed_ms  -- Time to cross in seconds
  crossing_time = 10 := by sorry

end train_crossing_time_l2324_232436


namespace count_multiples_of_24_l2324_232497

def smallest_square_multiple_of_24 : ℕ := 144
def smallest_fourth_power_multiple_of_24 : ℕ := 1296

theorem count_multiples_of_24 :
  (Finset.range (smallest_fourth_power_multiple_of_24 / 24 + 1) ∩ 
   Finset.filter (λ n => n ≥ smallest_square_multiple_of_24 / 24) 
                 (Finset.range (smallest_fourth_power_multiple_of_24 / 24 + 1))).card = 49 :=
by sorry

end count_multiples_of_24_l2324_232497


namespace age_equals_birth_year_digit_sum_l2324_232486

theorem age_equals_birth_year_digit_sum :
  ∃! A : ℕ, 0 ≤ A ∧ A ≤ 99 ∧
  (∃ x y : ℕ, 0 ≤ x ∧ x ≤ 9 ∧ 0 ≤ y ∧ y ≤ 9 ∧
    A = 1893 - (1800 + 10 * x + y) ∧
    A = 1 + 8 + x + y) ∧
  A = 24 :=
sorry

end age_equals_birth_year_digit_sum_l2324_232486


namespace euler_disproof_l2324_232419

theorem euler_disproof : 133^4 + 110^4 + 56^4 = 143^4 := by
  sorry

end euler_disproof_l2324_232419


namespace expand_and_simplify_polynomial_l2324_232453

theorem expand_and_simplify_polynomial (a : ℝ) : 
  (a + 1) * (a + 3) * (a + 4) * (a + 5) * (a + 6) = 
  a^5 + 19*a^4 + 137*a^3 + 461*a^2 + 702*a + 360 := by
  sorry

end expand_and_simplify_polynomial_l2324_232453


namespace yoongi_has_more_points_l2324_232414

theorem yoongi_has_more_points : ∀ (yoongi_points jungkook_points : ℕ),
  yoongi_points = 4 →
  jungkook_points = 6 - 3 →
  yoongi_points > jungkook_points :=
by
  sorry

end yoongi_has_more_points_l2324_232414


namespace second_watermelon_weight_l2324_232481

theorem second_watermelon_weight (total_weight first_weight : ℝ) 
  (h1 : total_weight = 14.02)
  (h2 : first_weight = 9.91) : 
  total_weight - first_weight = 4.11 := by
sorry

end second_watermelon_weight_l2324_232481


namespace sqrt_5_times_sqrt_6_minus_1_over_sqrt_5_range_l2324_232489

theorem sqrt_5_times_sqrt_6_minus_1_over_sqrt_5_range : 
  4 < Real.sqrt 5 * (Real.sqrt 6 - 1 / Real.sqrt 5) ∧ 
  Real.sqrt 5 * (Real.sqrt 6 - 1 / Real.sqrt 5) < 5 := by
  sorry

end sqrt_5_times_sqrt_6_minus_1_over_sqrt_5_range_l2324_232489


namespace complex_reciprocal_sum_l2324_232401

theorem complex_reciprocal_sum (z w : ℂ) 
  (hz : Complex.abs z = 2)
  (hw : Complex.abs w = 4)
  (hzw : Complex.abs (z + w) = 3) :
  Complex.abs (1 / z + 1 / w) = 3 / 8 := by
sorry

end complex_reciprocal_sum_l2324_232401


namespace figure_colorings_l2324_232459

/-- Represents the number of ways to color a single equilateral triangle --/
def triangle_colorings : ℕ := 6

/-- Represents the number of ways to color each subsequent triangle --/
def subsequent_triangle_colorings : ℕ := 3

/-- Represents the number of ways to color the additional dot --/
def additional_dot_colorings : ℕ := 2

/-- The total number of dots in the figure --/
def total_dots : ℕ := 10

/-- The number of triangles in the figure --/
def num_triangles : ℕ := 3

theorem figure_colorings :
  triangle_colorings * subsequent_triangle_colorings ^ (num_triangles - 1) * additional_dot_colorings = 108 := by
  sorry

end figure_colorings_l2324_232459


namespace quilt_shaded_fraction_l2324_232452

/-- Represents a square quilt block -/
structure QuiltBlock where
  total_squares : ℕ
  divided_squares : ℕ
  divided_rectangles : ℕ
  shaded_column : ℕ

/-- The fraction of the quilt block that is shaded -/
def shaded_fraction (q : QuiltBlock) : ℚ :=
  q.shaded_column / q.total_squares

/-- Theorem stating that the shaded fraction is 1/3 for the given quilt block configuration -/
theorem quilt_shaded_fraction :
  ∀ q : QuiltBlock,
    q.total_squares = 9 ∧
    q.divided_squares = 3 ∧
    q.divided_rectangles = 3 ∧
    q.shaded_column = 1 →
    shaded_fraction q = 1/3 := by
  sorry


end quilt_shaded_fraction_l2324_232452


namespace strategy_D_lowest_price_l2324_232494

/-- Represents a pricing strategy with an increase followed by a decrease -/
structure PricingStrategy where
  increase : ℝ
  decrease : ℝ

/-- Calculates the final price factor for a given pricing strategy -/
def finalPriceFactor (strategy : PricingStrategy) : ℝ :=
  (1 + strategy.increase) * (1 - strategy.decrease)

/-- The four pricing strategies -/
def strategyA : PricingStrategy := ⟨0.1, 0.1⟩
def strategyB : PricingStrategy := ⟨-0.1, -0.1⟩
def strategyC : PricingStrategy := ⟨0.2, 0.2⟩
def strategyD : PricingStrategy := ⟨0.3, 0.3⟩

theorem strategy_D_lowest_price :
  finalPriceFactor strategyD ≤ finalPriceFactor strategyA ∧
  finalPriceFactor strategyD ≤ finalPriceFactor strategyB ∧
  finalPriceFactor strategyD ≤ finalPriceFactor strategyC :=
sorry

end strategy_D_lowest_price_l2324_232494


namespace game_score_total_l2324_232440

theorem game_score_total (dad_score : ℕ) (olaf_score : ℕ) : 
  dad_score = 7 → 
  olaf_score = 3 * dad_score → 
  olaf_score + dad_score = 28 := by
sorry

end game_score_total_l2324_232440


namespace parabola_inequality_l2324_232456

/-- A parabola with x = 1 as its axis of symmetry -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  axis_symmetry : -b / (2 * a) = 1

theorem parabola_inequality (p : Parabola) : 2 * p.c < 3 * p.b := by
  sorry

end parabola_inequality_l2324_232456


namespace peters_pizza_consumption_l2324_232409

theorem peters_pizza_consumption :
  ∀ (total_slices : ℕ) (whole_slices : ℕ) (shared_slice : ℚ),
    total_slices = 16 →
    whole_slices = 2 →
    shared_slice = 1/3 →
    (whole_slices : ℚ) / total_slices + shared_slice / total_slices = 7/48 := by
  sorry

end peters_pizza_consumption_l2324_232409


namespace least_divisor_for_perfect_square_twenty_one_gives_perfect_square_twenty_one_is_least_l2324_232425

def is_perfect_square (x : ℕ) : Prop := ∃ y : ℕ, x = y * y

theorem least_divisor_for_perfect_square : 
  ∀ n : ℕ, n > 0 → is_perfect_square (16800 / n) → n ≥ 21 :=
by sorry

theorem twenty_one_gives_perfect_square : 
  is_perfect_square (16800 / 21) :=
by sorry

theorem twenty_one_is_least :
  ∀ n : ℕ, n > 0 → is_perfect_square (16800 / n) → n = 21 :=
by sorry

end least_divisor_for_perfect_square_twenty_one_gives_perfect_square_twenty_one_is_least_l2324_232425


namespace impossibility_l2324_232499

/-- The number of piles -/
def n : ℕ := 2018

/-- The i-th prime number -/
def p (i : ℕ) : ℕ := sorry

/-- The initial configuration of piles -/
def initial_config : Fin n → ℕ := λ i => p i.val

/-- The desired final configuration of piles -/
def final_config : Fin n → ℕ := λ _ => n

/-- Split operation: split a pile and add a chip to one of the new piles -/
def split (config : Fin n → ℕ) (i : Fin n) (k : ℕ) (add_to_first : Bool) : Fin n → ℕ := sorry

/-- Merge operation: merge two piles and add a chip to the merged pile -/
def merge (config : Fin n → ℕ) (i j : Fin n) : Fin n → ℕ := sorry

/-- Predicate to check if a configuration is reachable from the initial configuration -/
def is_reachable (config : Fin n → ℕ) : Prop := sorry

theorem impossibility : ¬ is_reachable final_config := by sorry

end impossibility_l2324_232499


namespace difference_twice_x_and_three_less_than_zero_l2324_232405

theorem difference_twice_x_and_three_less_than_zero (x : ℝ) :
  (2 * x - 3 < 0) ↔ (∃ y, y = 2 * x ∧ y - 3 < 0) :=
by sorry

end difference_twice_x_and_three_less_than_zero_l2324_232405


namespace amanda_borrowed_amount_l2324_232483

/-- Calculates the earnings for a given number of hours based on the specified payment cycle -/
def calculateEarnings (hours : Nat) : Nat :=
  let cycleEarnings := [2, 4, 6, 8, 10, 12]
  let fullCycles := hours / 6
  let remainingHours := hours % 6
  fullCycles * (cycleEarnings.sum) + (cycleEarnings.take remainingHours).sum

/-- The amount Amanda borrowed is equal to her earnings from 45 hours of mowing -/
theorem amanda_borrowed_amount : calculateEarnings 45 = 306 := by
  sorry

#eval calculateEarnings 45

end amanda_borrowed_amount_l2324_232483


namespace find_other_number_l2324_232430

theorem find_other_number (A B : ℕ+) (hA : A = 24) (hHCF : Nat.gcd A B = 16) (hLCM : Nat.lcm A B = 312) : B = 208 := by
  sorry

end find_other_number_l2324_232430


namespace race_difference_l2324_232407

/-- In a race, given the total distance and the differences between runners,
    calculate the difference between two runners. -/
theorem race_difference (total_distance : ℕ) (a_beats_b b_beats_c a_beats_c : ℕ) :
  total_distance = 1000 →
  a_beats_b = 70 →
  a_beats_c = 163 →
  b_beats_c = 93 :=
by sorry

end race_difference_l2324_232407


namespace expression_evaluation_l2324_232475

theorem expression_evaluation : 
  2 * (1 / (2 * 3) + 1 / (3 * 4) + 1 / (4 * 5) + 1 / (5 * 6) + 1 / (6 * 7) + 1 / (7 * 8)) = 3 / 4 := by
  sorry

end expression_evaluation_l2324_232475


namespace sine_graph_horizontal_compression_l2324_232455

/-- Given a function f(x) = 2sin(x + π/3), if we shorten the horizontal coordinates
    of its graph to 1/2 of the original while keeping the vertical coordinates unchanged,
    the resulting function is g(x) = 2sin(2x + π/3) -/
theorem sine_graph_horizontal_compression (x : ℝ) :
  let f : ℝ → ℝ := λ x ↦ 2 * Real.sin (x + π/3)
  let g : ℝ → ℝ := λ x ↦ 2 * Real.sin (2*x + π/3)
  let h : ℝ → ℝ := λ x ↦ f (x/2)
  h = g :=
by sorry

end sine_graph_horizontal_compression_l2324_232455


namespace jane_rejection_rate_l2324_232437

theorem jane_rejection_rate 
  (total_rejection_rate : ℝ) 
  (john_rejection_rate : ℝ) 
  (jane_inspection_fraction : ℝ) 
  (h1 : total_rejection_rate = 0.0075) 
  (h2 : john_rejection_rate = 0.005) 
  (h3 : jane_inspection_fraction = 0.8333333333333333) :
  let john_inspection_fraction := 1 - jane_inspection_fraction
  let jane_rejection_rate := (total_rejection_rate - john_rejection_rate * john_inspection_fraction) / jane_inspection_fraction
  jane_rejection_rate = 0.008 := by
sorry

end jane_rejection_rate_l2324_232437


namespace max_a6_value_l2324_232435

theorem max_a6_value (a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) 
  (h_order : a₁ ≤ a₂ ∧ a₂ ≤ a₃ ∧ a₃ ≤ a₄ ∧ a₄ ≤ a₅ ∧ a₅ ≤ a₆)
  (h_sum : a₁ + a₂ + a₃ + a₄ + a₅ + a₆ = 10)
  (h_sq_dev : (a₁ - 1)^2 + (a₂ - 1)^2 + (a₃ - 1)^2 + (a₄ - 1)^2 + (a₅ - 1)^2 + (a₆ - 1)^2 = 6) :
  a₆ ≤ 10/3 := by
sorry

end max_a6_value_l2324_232435


namespace complex_modulus_problem_l2324_232464

theorem complex_modulus_problem (z : ℂ) : (Complex.I^3 * z = 1 + Complex.I) → Complex.abs z = Real.sqrt 2 := by
  sorry

end complex_modulus_problem_l2324_232464


namespace operations_equality_l2324_232457

theorem operations_equality : 3 * 5 + 7 * 9 = 78 := by
  sorry

end operations_equality_l2324_232457


namespace max_triangle_area_l2324_232460

noncomputable section

def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

def focal_distance : ℝ := 2

def eccentricity : ℝ := Real.sqrt 2 / 2

def right_focus : ℝ × ℝ := (1, 0)

def point_k : ℝ × ℝ := (2, 0)

def line_intersects_ellipse (k : ℝ) (x y : ℝ) : Prop :=
  y = k * (x - 2) ∧ ellipse x y

def triangle_area (F P Q : ℝ × ℝ) : ℝ :=
  abs ((P.1 - F.1) * (Q.2 - F.2) - (Q.1 - F.1) * (P.2 - F.2)) / 2

theorem max_triangle_area :
  ∃ (max_area : ℝ),
    max_area = Real.sqrt 2 / 4 ∧
    ∀ (k : ℝ) (P Q : ℝ × ℝ),
      k ≠ 0 →
      line_intersects_ellipse k P.1 P.2 →
      line_intersects_ellipse k Q.1 Q.2 →
      P ≠ Q →
      triangle_area right_focus P Q ≤ max_area :=
sorry

end

end max_triangle_area_l2324_232460


namespace fraction_of_y_l2324_232434

theorem fraction_of_y (w x y : ℝ) (hw : w ≠ 0) (hx : x ≠ 0) (hy : y ≠ 0) : 
  (2 / w + 2 / x = 2 / y) → (w * x = y) → ((w + x) / 2 = 0.5) → (2 / y = 2 / y) := by
  sorry

end fraction_of_y_l2324_232434


namespace solution_set_inequality_l2324_232463

theorem solution_set_inequality (x : ℝ) : x / (x + 1) ≤ 0 ↔ x ∈ Set.Ioc (-1) 0 := by
  sorry

end solution_set_inequality_l2324_232463


namespace nancy_museum_pictures_l2324_232422

theorem nancy_museum_pictures :
  ∀ (zoo_pics museum_pics deleted_pics remaining_pics : ℕ),
    zoo_pics = 49 →
    deleted_pics = 38 →
    remaining_pics = 19 →
    zoo_pics + museum_pics = deleted_pics + remaining_pics →
    museum_pics = 8 :=
by sorry

end nancy_museum_pictures_l2324_232422


namespace correct_calculation_l2324_232495

theorem correct_calculation (x : ℝ) (h : x * 3 - 45 = 159) : (x + 32) * 12 = 1200 := by
  sorry

#check correct_calculation

end correct_calculation_l2324_232495


namespace problem_1_l2324_232432

theorem problem_1 : (-1)^2020 * (2020 - Real.pi)^0 - 1 = 0 := by
  sorry

end problem_1_l2324_232432


namespace ball_arrangements_l2324_232443

-- Define the word structure
def Word := String

-- Define a function to count distinct arrangements
def countDistinctArrangements (w : Word) : ℕ := sorry

-- Theorem statement
theorem ball_arrangements :
  let ball : Word := "BALL"
  countDistinctArrangements ball = 12 := by sorry

end ball_arrangements_l2324_232443


namespace class_average_mark_l2324_232447

theorem class_average_mark (total_students : ℕ) (excluded_students : ℕ) (excluded_avg : ℝ) (remaining_avg : ℝ) :
  total_students = 25 →
  excluded_students = 5 →
  excluded_avg = 20 →
  remaining_avg = 95 →
  (total_students * (total_students * remaining_avg - excluded_students * remaining_avg + excluded_students * excluded_avg)) / (total_students * total_students) = 80 :=
by
  sorry

end class_average_mark_l2324_232447


namespace whispered_numbers_l2324_232484

/-- Represents a digit sum calculation step -/
def DigitSumStep (n : ℕ) : ℕ := sorry

/-- The maximum possible digit sum for a 2022-digit number -/
def MaxInitialSum : ℕ := 2022 * 9

theorem whispered_numbers (initial_number : ℕ) 
  (h1 : initial_number ≤ MaxInitialSum) 
  (whisper1 : ℕ) 
  (h2 : whisper1 = DigitSumStep initial_number)
  (whisper2 : ℕ) 
  (h3 : whisper2 = DigitSumStep whisper1)
  (h4 : 10 ≤ whisper2 ∧ whisper2 ≤ 99)
  (h5 : DigitSumStep whisper2 = 1) :
  whisper1 = 19 ∨ whisper1 = 28 := by sorry

end whispered_numbers_l2324_232484


namespace infinitely_many_divisible_by_power_of_three_l2324_232477

theorem infinitely_many_divisible_by_power_of_three (k : ℕ) (hk : k > 0) :
  ∃ f : ℕ → ℕ, ∀ n : ℕ, (3^k : ℕ) ∣ (f n)^3 + 10 :=
sorry

end infinitely_many_divisible_by_power_of_three_l2324_232477


namespace nancy_homework_problem_l2324_232416

theorem nancy_homework_problem (finished : ℕ) (pages_left : ℕ) (problems_per_page : ℕ) : 
  finished = 47 → pages_left = 6 → problems_per_page = 9 →
  finished + pages_left * problems_per_page = 101 := by
  sorry

end nancy_homework_problem_l2324_232416


namespace fishing_rod_price_theorem_l2324_232467

theorem fishing_rod_price_theorem :
  ∃ (a b c d : ℕ),
    -- Four-digit number condition
    1000 ≤ a * 1000 + b * 100 + c * 10 + d ∧ a * 1000 + b * 100 + c * 10 + d < 10000 ∧
    -- Digit relationships
    a = c + 1 ∧ a = d - 1 ∧
    -- Sum of digits
    a + b + c + d = 6 ∧
    -- Two-digit number difference
    10 * a + b = 10 * c + d + 7 ∧
    -- Product of ages
    a * 1000 + b * 100 + c * 10 + d = 61 * 3 * 11 :=
by sorry

end fishing_rod_price_theorem_l2324_232467


namespace arithmetic_sequence_sum_l2324_232408

/-- Given an arithmetic sequence {aₙ} where (a₂ + a₅ = 4) and (a₆ + a₉ = 20),
    prove that (a₄ + a₇) = 12 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  (∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence condition
  a 2 + a 5 = 4 →
  a 6 + a 9 = 20 →
  a 4 + a 7 = 12 := by
sorry

end arithmetic_sequence_sum_l2324_232408


namespace ellipse_equation_l2324_232448

/-- The equation of an ellipse given specific conditions -/
theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  (∃ (c : ℝ), c = 4 ∧ a^2 - b^2 = c^2) →  -- Right focus coincides with parabola focus
  (a / c = 3 / Real.sqrt 6) →             -- Eccentricity condition
  (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ↔ x^2 / 24 + y^2 / 8 = 1) :=
by sorry

end ellipse_equation_l2324_232448


namespace chips_ounces_amber_chips_problem_l2324_232449

/-- Represents the problem of determining the number of ounces in a bag of chips. -/
theorem chips_ounces (total_money : ℚ) (candy_price : ℚ) (candy_ounces : ℚ) 
  (chips_price : ℚ) (max_ounces : ℚ) : ℚ :=
  let candy_bags := total_money / candy_price
  let candy_total_ounces := candy_bags * candy_ounces
  let chips_bags := total_money / chips_price
  let chips_ounces_per_bag := max_ounces / chips_bags
  chips_ounces_per_bag

/-- Proves that given the conditions in the problem, a bag of chips contains 17 ounces. -/
theorem amber_chips_problem : 
  chips_ounces 7 1 12 (14/10) 85 = 17 := by
  sorry

end chips_ounces_amber_chips_problem_l2324_232449


namespace min_gigabytes_plan_y_more_expensive_l2324_232423

/-- Represents the cost of Plan Y in cents for a given number of gigabytes -/
def planYCost (gigabytes : ℕ) : ℕ := 3000 + 200 * gigabytes

/-- Represents the cost of Plan X in cents -/
def planXCost : ℕ := 5000

/-- Theorem stating that 11 gigabytes is the minimum at which Plan Y becomes more expensive than Plan X -/
theorem min_gigabytes_plan_y_more_expensive :
  ∀ g : ℕ, g ≥ 11 ↔ planYCost g > planXCost :=
by sorry

end min_gigabytes_plan_y_more_expensive_l2324_232423


namespace sum_odd_divisors_300_eq_124_l2324_232454

/-- The sum of all odd divisors of 300 -/
def sum_odd_divisors_300 : ℕ := 124

/-- Theorem: The sum of all odd divisors of 300 is 124 -/
theorem sum_odd_divisors_300_eq_124 : sum_odd_divisors_300 = 124 := by sorry

end sum_odd_divisors_300_eq_124_l2324_232454


namespace expression_value_l2324_232445

theorem expression_value (a b : ℝ) (h : 2 * a - b = -1) : 
  b * 2 - a * 2^2 = 2 := by
sorry

end expression_value_l2324_232445


namespace age_difference_l2324_232474

theorem age_difference (A B : ℕ) : B = 39 → A + 10 = 2 * (B - 10) → A - B = 9 := by
  sorry

end age_difference_l2324_232474


namespace intersection_A_M_range_of_b_l2324_232473

-- Define the sets A, B, M, and U
def A : Set ℝ := {x | -3 < x ∧ x ≤ 6}
def B (b : ℝ) : Set ℝ := {x | b - 3 < x ∧ x < b + 7}
def M : Set ℝ := {x | -4 ≤ x ∧ x < 5}
def U : Set ℝ := Set.univ

-- Theorem 1: A ∩ M = {x | -3 < x < 5}
theorem intersection_A_M : A ∩ M = {x : ℝ | -3 < x ∧ x < 5} := by sorry

-- Theorem 2: If B ∪ (¬UM) = R, then -2 ≤ b < -1
theorem range_of_b (b : ℝ) (h : B b ∪ (Mᶜ) = Set.univ) : -2 ≤ b ∧ b < -1 := by sorry

end intersection_A_M_range_of_b_l2324_232473


namespace nancy_crayon_packs_l2324_232492

theorem nancy_crayon_packs (total_crayons : ℕ) (crayons_per_pack : ℕ) (h1 : total_crayons = 615) (h2 : crayons_per_pack = 15) :
  total_crayons / crayons_per_pack = 41 := by
  sorry

end nancy_crayon_packs_l2324_232492


namespace cosine_inequality_solution_l2324_232421

theorem cosine_inequality_solution (y : Real) : 
  (y ∈ Set.Icc 0 Real.pi ∧ 
   ∀ x ∈ Set.Icc 0 Real.pi, Real.cos (x + y) ≥ Real.cos x + Real.cos y - 1) ↔ 
  (y = 0 ∨ y = Real.pi) := by
sorry

end cosine_inequality_solution_l2324_232421


namespace smallest_q_value_l2324_232429

def sum_of_range (n : ℕ) : ℕ := n * (n + 1) / 2

theorem smallest_q_value (p : ℕ) : 
  let initial_sum := sum_of_range 6
  let total_count := 6 + p + q
  let total_sum := initial_sum + 5 * p + 7 * q
  let mean := 5.3
  ∃ q : ℕ, q ≥ 0 ∧ (total_sum : ℝ) / total_count = mean ∧ 
    ∀ q' : ℕ, q' ≥ 0 → (initial_sum + 5 * p + 7 * q' : ℝ) / (6 + p + q') = mean → q ≤ q'
  := by sorry

end smallest_q_value_l2324_232429


namespace coloring_book_solution_l2324_232479

/-- Represents the problem of determining the initial stock of coloring books. -/
def ColoringBookProblem (initial_stock acquired_books books_per_shelf total_shelves : ℝ) : Prop :=
  initial_stock + acquired_books = books_per_shelf * total_shelves

/-- The theorem stating the solution to the coloring book problem. -/
theorem coloring_book_solution :
  ∃ (initial_stock : ℝ),
    ColoringBookProblem initial_stock 20 4 15 ∧
    initial_stock = 40 := by
  sorry

end coloring_book_solution_l2324_232479


namespace sum_of_reciprocals_l2324_232465

theorem sum_of_reciprocals (x y : ℝ) 
  (h1 : 1/x + 1/y = 4) 
  (h2 : 1/x - 1/y = -2) : 
  x + y = 4/3 := by
sorry

end sum_of_reciprocals_l2324_232465


namespace distance_major_minor_endpoints_l2324_232470

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := 4 * (x + 2)^2 + 16 * y^2 = 64

-- Define the center of the ellipse
def center : ℝ × ℝ := (-2, 0)

-- Define the semi-major axis length
def semi_major : ℝ := 4

-- Define the semi-minor axis length
def semi_minor : ℝ := 1

-- Define an endpoint of the major axis
def major_endpoint : ℝ × ℝ := (center.1 + semi_major, center.2)

-- Define an endpoint of the minor axis
def minor_endpoint : ℝ × ℝ := (center.1, center.2 + semi_minor)

-- Theorem statement
theorem distance_major_minor_endpoints : 
  Real.sqrt ((major_endpoint.1 - minor_endpoint.1)^2 + (major_endpoint.2 - minor_endpoint.2)^2) = Real.sqrt 17 := by
  sorry

end distance_major_minor_endpoints_l2324_232470


namespace base5_to_base7_conversion_l2324_232417

/-- Converts a number from base 5 to base 10 -/
def base5_to_decimal (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 7 -/
def decimal_to_base7 (n : ℕ) : ℕ := sorry

/-- Theorem: The base-7 representation of 412₅ is 212₇ -/
theorem base5_to_base7_conversion :
  decimal_to_base7 (base5_to_decimal 412) = 212 := by sorry

end base5_to_base7_conversion_l2324_232417


namespace yadav_clothes_transport_expenditure_l2324_232438

/-- Represents Mr. Yadav's financial situation --/
structure YadavFinances where
  monthlySalary : ℝ
  consumablePercentage : ℝ
  rentPercentage : ℝ
  utilitiesPercentage : ℝ
  entertainmentPercentage : ℝ
  clothesTransportPercentage : ℝ
  annualSavings : ℝ

/-- Calculates Mr. Yadav's monthly expenditure on clothes and transport --/
def monthlyClothesTransportExpenditure (y : YadavFinances) : ℝ :=
  let totalSpentPercentage := y.consumablePercentage + y.rentPercentage + y.utilitiesPercentage + y.entertainmentPercentage
  let remainingPercentage := 1 - totalSpentPercentage
  let monthlyRemainder := y.monthlySalary * remainingPercentage
  monthlyRemainder * y.clothesTransportPercentage

/-- Theorem stating that Mr. Yadav's monthly expenditure on clothes and transport is 2052 --/
theorem yadav_clothes_transport_expenditure (y : YadavFinances) 
  (h1 : y.consumablePercentage = 0.6)
  (h2 : y.rentPercentage = 0.2)
  (h3 : y.utilitiesPercentage = 0.1)
  (h4 : y.entertainmentPercentage = 0.05)
  (h5 : y.clothesTransportPercentage = 0.5)
  (h6 : y.annualSavings = 24624) :
  monthlyClothesTransportExpenditure y = 2052 := by
  sorry

#check yadav_clothes_transport_expenditure

end yadav_clothes_transport_expenditure_l2324_232438


namespace quadratic_value_theorem_l2324_232468

theorem quadratic_value_theorem (x : ℝ) (h : x^2 + 4*x - 2 = 0) :
  3*x^2 + 12*x - 23 = -17 := by
  sorry

end quadratic_value_theorem_l2324_232468


namespace all_statements_incorrect_l2324_232444

/-- Represents a type of reasoning -/
inductive ReasoningType
| Analogical
| Inductive

/-- Represents the direction of reasoning -/
inductive ReasoningDirection
| GeneralToSpecific
| SpecificToGeneral
| SpecificToSpecific

/-- Represents a statement about analogical reasoning -/
structure AnalogicalReasoningStatement where
  always_correct : Bool
  direction : ReasoningDirection
  can_prove_math : Bool
  same_as_inductive : Bool

/-- Definition of analogical reasoning -/
def analogical_reasoning : ReasoningType := ReasoningType.Analogical

/-- Definition of inductive reasoning -/
def inductive_reasoning : ReasoningType := ReasoningType.Inductive

/-- Inductive reasoning is a form of analogical reasoning -/
axiom inductive_is_analogical : inductive_reasoning = analogical_reasoning

/-- The correct properties of analogical reasoning -/
def correct_properties : AnalogicalReasoningStatement :=
  { always_correct := false
  , direction := ReasoningDirection.SpecificToSpecific
  , can_prove_math := false
  , same_as_inductive := false }

/-- Theorem stating that all given statements about analogical reasoning are incorrect -/
theorem all_statements_incorrect (statement : AnalogicalReasoningStatement) :
  statement.always_correct = true ∨
  statement.direction = ReasoningDirection.GeneralToSpecific ∨
  statement.can_prove_math = true ∨
  statement.same_as_inductive = true →
  statement ≠ correct_properties :=
sorry

end all_statements_incorrect_l2324_232444


namespace quadratic_coefficients_l2324_232485

/-- A quadratic function f(x) = ax^2 + bx + c satisfying specific conditions -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_positive : a > 0
  condition_1 : |a + b + c| = 3
  condition_2 : |4*a + 2*b + c| = 3
  condition_3 : |9*a + 3*b + c| = 3

/-- The theorem stating the possible coefficients of the quadratic function -/
theorem quadratic_coefficients (f : QuadraticFunction) :
  (f.a = 6 ∧ f.b = -24 ∧ f.c = 21) ∨
  (f.a = 3 ∧ f.b = -15 ∧ f.c = 15) ∨
  (f.a = 3 ∧ f.b = -9 ∧ f.c = 3) := by
  sorry

end quadratic_coefficients_l2324_232485


namespace meeting_time_is_48_minutes_l2324_232480

/-- Represents the cycling scenario between Andrea and Lauren -/
structure CyclingScenario where
  total_distance : ℝ
  andrea_speed_ratio : ℝ
  distance_decrease_rate : ℝ
  andrea_stop_time : ℝ

/-- Calculates the total time for Lauren to meet Andrea -/
def total_meeting_time (scenario : CyclingScenario) : ℝ :=
  sorry

/-- Theorem stating that given the specific conditions, the total meeting time is 48 minutes -/
theorem meeting_time_is_48_minutes 
  (scenario : CyclingScenario)
  (h1 : scenario.total_distance = 30)
  (h2 : scenario.andrea_speed_ratio = 2)
  (h3 : scenario.distance_decrease_rate = 1.5)
  (h4 : scenario.andrea_stop_time = 6) :
  total_meeting_time scenario = 48 :=
sorry

end meeting_time_is_48_minutes_l2324_232480


namespace cylinder_max_volume_l2324_232415

/-- Given a cylinder with an axial cross-section circumference of 90 cm,
    prove that its maximum volume is 3375π cm³. -/
theorem cylinder_max_volume (d m : ℝ) (h : d + m = 45) :
  ∃ (V : ℝ), V ≤ 3375 * Real.pi ∧ ∃ (r : ℝ), V = π * r^2 * m ∧ d = 2 * r :=
sorry

end cylinder_max_volume_l2324_232415


namespace small_bottle_volume_proof_l2324_232400

/-- The volume of a big bottle in ounces -/
def big_bottle_volume : ℝ := 30

/-- The cost of a big bottle in pesetas -/
def big_bottle_cost : ℝ := 2700

/-- The cost of a small bottle in pesetas -/
def small_bottle_cost : ℝ := 600

/-- The amount saved in pesetas by buying a big bottle instead of smaller bottles for the same volume -/
def savings : ℝ := 300

/-- The volume of a small bottle in ounces -/
def small_bottle_volume : ℝ := 6

theorem small_bottle_volume_proof :
  small_bottle_volume * (big_bottle_cost / big_bottle_volume) =
  small_bottle_cost + (savings / big_bottle_volume) * small_bottle_volume :=
by sorry

end small_bottle_volume_proof_l2324_232400


namespace measure_one_kg_grain_l2324_232433

/-- Represents a balance scale --/
structure BalanceScale where
  isInaccurate : Bool

/-- Represents a weight --/
structure Weight where
  mass : ℝ
  isAccurate : Bool

/-- Represents a bag of grain --/
structure GrainBag where
  mass : ℝ

/-- Function to measure a specific mass of grain --/
def measureGrain (scale : BalanceScale) (reference : Weight) (bag : GrainBag) (targetMass : ℝ) : Prop :=
  scale.isInaccurate ∧ reference.isAccurate ∧ reference.mass = targetMass

/-- Theorem stating that it's possible to measure 1 kg of grain using inaccurate scales and an accurate 1 kg weight --/
theorem measure_one_kg_grain 
  (scale : BalanceScale) 
  (reference : Weight) 
  (bag : GrainBag) : 
  measureGrain scale reference bag 1 → 
  ∃ (measuredGrain : GrainBag), measuredGrain.mass = 1 :=
sorry

end measure_one_kg_grain_l2324_232433


namespace sin_ten_pi_thirds_l2324_232471

theorem sin_ten_pi_thirds : Real.sin (10 * π / 3) = -Real.sqrt 3 / 2 := by
  sorry

end sin_ten_pi_thirds_l2324_232471


namespace min_perimeter_triangle_l2324_232404

theorem min_perimeter_triangle (a b x : ℕ) (ha : a = 24) (hb : b = 37) : 
  (a + b + x > a + b ∧ a + x > b ∧ b + x > a) → 
  (∀ y : ℕ, (a + b + y > a + b ∧ a + y > b ∧ b + y > a) → x ≤ y) →
  a + b + x = 75 := by sorry

end min_perimeter_triangle_l2324_232404


namespace football_team_throwers_l2324_232413

theorem football_team_throwers :
  ∀ (total_players throwers right_handed : ℕ),
    total_players = 70 →
    throwers ≤ total_players →
    right_handed = 63 →
    3 * (right_handed - throwers) = 2 * (total_players - throwers) →
    throwers = 49 := by
  sorry

end football_team_throwers_l2324_232413


namespace triangle_theorem_l2324_232446

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  angle_sum : A + B + C = π
  cosine_law_a : a^2 = b^2 + c^2 - 2*b*c*Real.cos A
  cosine_law_b : b^2 = a^2 + c^2 - 2*a*c*Real.cos B
  cosine_law_c : c^2 = a^2 + b^2 - 2*a*b*Real.cos C

/-- The main theorem about the triangle -/
theorem triangle_theorem (t : Triangle) (h : 2*t.a*Real.cos t.C = 2*t.b - t.c) :
  /- Part 1 -/
  t.A = π/3 ∧
  /- Part 2 -/
  (t.A < π/2 ∧ t.B < π/2 ∧ t.C < π/2 → 
    3/2 < Real.sin t.B + Real.sin t.C ∧ Real.sin t.B + Real.sin t.C ≤ Real.sqrt 3) ∧
  /- Part 3 -/
  (t.a = 2*Real.sqrt 3 ∧ 1/2*t.b*t.c*Real.sin t.A = 2*Real.sqrt 3 →
    Real.cos (2*t.B) + Real.cos (2*t.C) = -1/2) :=
by sorry

end triangle_theorem_l2324_232446


namespace pen_cost_l2324_232424

/-- The cost of a pen in cents, given the following conditions:
  * Pencils cost 25 cents each
  * Susan spent 20 dollars in total
  * Susan bought a total of 36 pens and pencils
  * Susan bought 16 pencils
-/
theorem pen_cost (pencil_cost : ℕ) (total_spent : ℕ) (total_items : ℕ) (pencils_bought : ℕ) :
  pencil_cost = 25 →
  total_spent = 2000 →
  total_items = 36 →
  pencils_bought = 16 →
  ∃ (pen_cost : ℕ), pen_cost = 80 :=
by sorry

end pen_cost_l2324_232424


namespace chinese_chess_probability_l2324_232488

theorem chinese_chess_probability (p_win p_draw : ℝ) 
  (h_win : p_win = 0.5) 
  (h_draw : p_draw = 0.2) : 
  p_win + p_draw = 0.7 := by
  sorry

end chinese_chess_probability_l2324_232488


namespace planted_fraction_is_thirteen_fifteenths_l2324_232450

/-- Represents a right triangle field with an unplanted rectangle at the right angle -/
structure FieldWithUnplantedRectangle where
  /-- Length of the first leg of the right triangle -/
  leg1 : ℝ
  /-- Length of the second leg of the right triangle -/
  leg2 : ℝ
  /-- Width of the unplanted rectangle -/
  rect_width : ℝ
  /-- Height of the unplanted rectangle -/
  rect_height : ℝ
  /-- Shortest distance from the unplanted rectangle to the hypotenuse -/
  dist_to_hypotenuse : ℝ

/-- Calculates the fraction of the field that is planted -/
def planted_fraction (field : FieldWithUnplantedRectangle) : ℝ :=
  sorry

/-- Theorem stating the planted fraction for the given field configuration -/
theorem planted_fraction_is_thirteen_fifteenths :
  let field := FieldWithUnplantedRectangle.mk 5 12 1 4 3
  planted_fraction field = 13 / 15 := by
  sorry

end planted_fraction_is_thirteen_fifteenths_l2324_232450


namespace max_diagonal_of_rectangle_l2324_232493

/-- The maximum diagonal of a rectangle with perimeter 40 --/
theorem max_diagonal_of_rectangle (l w : ℝ) : 
  l > 0 → w > 0 → l + w = 20 → 
  ∀ (x y : ℝ), x > 0 → y > 0 → x + y = 20 → 
  Real.sqrt (l^2 + w^2) ≤ 20 :=
by sorry

end max_diagonal_of_rectangle_l2324_232493
