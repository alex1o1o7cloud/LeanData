import Mathlib

namespace average_string_length_l1326_132612

theorem average_string_length :
  let string1 : ℚ := 2
  let string2 : ℚ := 5
  let string3 : ℚ := 3
  let num_strings : ℕ := 3
  (string1 + string2 + string3) / num_strings = 10 / 3 := by
sorry

end average_string_length_l1326_132612


namespace second_day_charge_l1326_132622

theorem second_day_charge (day1_charge : ℝ) (day3_charge : ℝ) (attendance_ratio : Fin 3 → ℝ) (average_charge : ℝ) :
  day1_charge = 15 →
  day3_charge = 2.5 →
  attendance_ratio 0 = 2 →
  attendance_ratio 1 = 5 →
  attendance_ratio 2 = 13 →
  average_charge = 5 →
  ∃ day2_charge : ℝ,
    day2_charge = 7.5 ∧
    average_charge * (attendance_ratio 0 + attendance_ratio 1 + attendance_ratio 2) =
      day1_charge * attendance_ratio 0 + day2_charge * attendance_ratio 1 + day3_charge * attendance_ratio 2 :=
by
  sorry


end second_day_charge_l1326_132622


namespace harold_grocery_expense_l1326_132655

/-- Harold's monthly finances --/
def harold_finances (grocery_expense : ℚ) : Prop :=
  let monthly_income : ℚ := 2500
  let rent : ℚ := 700
  let car_payment : ℚ := 300
  let utilities : ℚ := car_payment / 2
  let fixed_expenses : ℚ := rent + car_payment + utilities
  let remaining_after_fixed : ℚ := monthly_income - fixed_expenses
  let retirement_savings : ℚ := (remaining_after_fixed - grocery_expense) / 2
  retirement_savings = 650 ∧ remaining_after_fixed - retirement_savings - grocery_expense = 650

theorem harold_grocery_expense : 
  ∃ (expense : ℚ), harold_finances expense ∧ expense = 50 :=
sorry

end harold_grocery_expense_l1326_132655


namespace f_properties_l1326_132610

noncomputable def f (a b x : ℝ) : ℝ := a * x^2 + b / x^2

theorem f_properties (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let f := f a b
  ∃ (min_value : ℝ) (min_point : ℝ),
    (∀ x, x ≠ 0 → f x ≥ min_value) ∧
    (f min_point = min_value) ∧
    (min_value = 2 * Real.sqrt (a * b)) ∧
    (min_point = Real.sqrt (Real.sqrt (b / a))) ∧
    (∀ x, f (-x) = f x) ∧
    (∀ x y, 0 < x ∧ x < y ∧ y < min_point → f x > f y) ∧
    (∀ x y, min_point < x ∧ x < y → f x < f y) :=
by sorry

end f_properties_l1326_132610


namespace square_roots_sum_l1326_132620

theorem square_roots_sum (x y : ℝ) : 
  x^2 = 16 → y^2 = 9 → x^2 + y^2 + x - 2 = 27 := by
  sorry

end square_roots_sum_l1326_132620


namespace absolute_value_inequality_solution_set_l1326_132667

theorem absolute_value_inequality_solution_set :
  ∀ x : ℝ, |x - 2| < 1 ↔ x ∈ Set.Ioo 1 3 := by sorry

end absolute_value_inequality_solution_set_l1326_132667


namespace perpendicular_line_equation_l1326_132642

/-- Given a line L1 with equation 2x + 3y - 6 = 0 and a point P (0, -3),
    prove that the line L2 passing through P and perpendicular to L1
    has the equation 3x - 2y - 6 = 0 -/
theorem perpendicular_line_equation :
  let L1 : Set (ℝ × ℝ) := {(x, y) | 2 * x + 3 * y - 6 = 0}
  let P : ℝ × ℝ := (0, -3)
  let L2 : Set (ℝ × ℝ) := {(x, y) | 3 * x - 2 * y - 6 = 0}
  (∀ (x y : ℝ), (x, y) ∈ L2 ↔ 3 * x - 2 * y - 6 = 0) ∧
  P ∈ L2 ∧
  (∀ (x₁ y₁ x₂ y₂ : ℝ), (x₁, y₁) ∈ L1 → (x₂, y₂) ∈ L1 → x₁ ≠ x₂ →
    ((x₁ - x₂) * (x - 0) + (y₁ - y₂) * (y + 3) = 0 ↔ (x, y) ∈ L2)) := by
  sorry

end perpendicular_line_equation_l1326_132642


namespace product_equality_l1326_132641

def prod (a : ℕ → ℕ) (m n : ℕ) : ℕ :=
  if m > n then 1 else (List.range (n - m + 1)).foldl (fun acc i => acc * a (i + m)) 1

theorem product_equality :
  (prod (fun k => 2 * k - 1) 1 1008) * (prod (fun k => 2 * k) 1 1007) = prod id 1 2015 := by
  sorry

end product_equality_l1326_132641


namespace root_sum_transformation_l1326_132640

theorem root_sum_transformation (α β γ : ℂ) : 
  (α^3 - α - 1 = 0) → (β^3 - β - 1 = 0) → (γ^3 - γ - 1 = 0) →
  ((1 - α) / (1 + α)) + ((1 - β) / (1 + β)) + ((1 - γ) / (1 + γ)) = 1 := by
sorry

end root_sum_transformation_l1326_132640


namespace intersection_of_A_and_B_l1326_132646

def A : Set ℝ := {x | x + 1 > 0}
def B : Set ℝ := {y | (y - 2) * (y + 3) < 0}

theorem intersection_of_A_and_B : A ∩ B = Set.Ioo (-1 : ℝ) 2 := by sorry

end intersection_of_A_and_B_l1326_132646


namespace sum_of_coefficients_l1326_132605

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) :
  (∀ x : ℝ, (x^2 - x - 2)^5 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + 
                               a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9 + a₁₀*x^10) →
  a + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ = -33 :=
by sorry

end sum_of_coefficients_l1326_132605


namespace exists_valid_coloring_l1326_132678

/-- A coloring of the plane using seven colors -/
def Coloring := ℝ × ℝ → Fin 7

/-- The property that no two points of the same color are exactly 1 unit apart -/
def ValidColoring (c : Coloring) : Prop :=
  ∀ x y : ℝ × ℝ, c x = c y → (x.1 - y.1)^2 + (x.2 - y.2)^2 ≠ 1

/-- There exists a coloring of the plane using seven colors such that
    no two points of the same color are exactly 1 unit apart -/
theorem exists_valid_coloring : ∃ c : Coloring, ValidColoring c := by
  sorry

end exists_valid_coloring_l1326_132678


namespace particular_number_l1326_132686

theorem particular_number (x : ℤ) (h : x - 7 = 2) : x = 9 := by
  sorry

end particular_number_l1326_132686


namespace inequality_holds_iff_k_geq_four_l1326_132648

theorem inequality_holds_iff_k_geq_four :
  ∀ k : ℝ, k > 0 →
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 →
    a / (b + c) + b / (c + a) + k * c / (a + b) ≥ 2) ↔
  k ≥ 4 := by sorry

end inequality_holds_iff_k_geq_four_l1326_132648


namespace purple_shoes_count_l1326_132634

theorem purple_shoes_count (total : ℕ) (blue : ℕ) (h1 : total = 1250) (h2 : blue = 540) :
  let remaining := total - blue
  let purple := remaining / 2
  purple = 355 := by
sorry

end purple_shoes_count_l1326_132634


namespace bottle_production_l1326_132676

/-- Given that 6 identical machines produce 270 bottles per minute at a constant rate,
    prove that 20 such machines will produce 3600 bottles in 4 minutes. -/
theorem bottle_production (rate : ℕ) (h1 : 6 * rate = 270) : 20 * rate * 4 = 3600 := by
  sorry

end bottle_production_l1326_132676


namespace median_room_number_l1326_132627

/-- Given a list of integers from 1 to n with two consecutive numbers removed,
    this function returns the median of the remaining numbers. -/
def medianWithGap (n : ℕ) (gap_start : ℕ) : ℕ :=
  if gap_start ≤ (n + 1) / 2
  then (n + 1) / 2 + 1
  else (n + 1) / 2

theorem median_room_number :
  medianWithGap 23 14 = 13 :=
by sorry

end median_room_number_l1326_132627


namespace correct_num_pregnant_dogs_l1326_132662

/-- The number of pregnant dogs Chuck has. -/
def num_pregnant_dogs : ℕ := 3

/-- The number of puppies each pregnant dog gives birth to. -/
def puppies_per_dog : ℕ := 4

/-- The number of shots each puppy needs. -/
def shots_per_puppy : ℕ := 2

/-- The cost of each shot in dollars. -/
def cost_per_shot : ℕ := 5

/-- The total cost of all shots in dollars. -/
def total_cost : ℕ := 120

/-- Theorem stating that the number of pregnant dogs is correct given the conditions. -/
theorem correct_num_pregnant_dogs :
  num_pregnant_dogs * puppies_per_dog * shots_per_puppy * cost_per_shot = total_cost :=
by sorry

end correct_num_pregnant_dogs_l1326_132662


namespace prob_both_female_given_one_female_l1326_132613

/-- Represents the number of male students -/
def num_male : ℕ := 3

/-- Represents the number of female students -/
def num_female : ℕ := 2

/-- Represents the total number of students -/
def total_students : ℕ := num_male + num_female

/-- Represents the number of students drawn -/
def students_drawn : ℕ := 2

/-- The probability of drawing both female students given that one female student is drawn -/
theorem prob_both_female_given_one_female :
  (students_drawn = 2) →
  (num_male = 3) →
  (num_female = 2) →
  (∃ (p : ℚ), p = 1 / 7 ∧ 
    p = (1 : ℚ) / (total_students.choose students_drawn - num_male.choose students_drawn)) :=
sorry

end prob_both_female_given_one_female_l1326_132613


namespace no_solution_for_inequality_l1326_132695

theorem no_solution_for_inequality :
  ¬ ∃ x : ℝ, |x| + |2023 - x| < 2023 := by
  sorry

end no_solution_for_inequality_l1326_132695


namespace triangle_property_l1326_132607

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that if a∙sin(A) + c∙sin(C) - √2∙a∙sin(C) = b∙sin(B) and cos(A) = 1/3,
    then B = π/4 and sin(C) = (4 + √2) / 6. -/
theorem triangle_property (a b c A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  0 < A → A < π →
  0 < B → B < π →
  0 < C → C < π →
  A + B + C = π →
  a * Real.sin A + c * Real.sin C - Real.sqrt 2 * a * Real.sin C = b * Real.sin B →
  Real.cos A = 1 / 3 →
  B = π / 4 ∧ Real.sin C = (4 + Real.sqrt 2) / 6 := by
  sorry

end triangle_property_l1326_132607


namespace cirrus_to_cumulus_ratio_l1326_132654

theorem cirrus_to_cumulus_ratio :
  ∀ (cirrus cumulus cumulonimbus : ℕ),
    cirrus = 144 →
    cumulonimbus = 3 →
    cumulus = 12 * cumulonimbus →
    ∃ k : ℕ, cirrus = k * cumulus →
    cirrus / cumulus = 4 :=
by sorry

end cirrus_to_cumulus_ratio_l1326_132654


namespace receipts_change_after_price_reduction_and_sales_increase_l1326_132625

/-- Calculates the percentage change in total receipts when price is reduced and sales increase -/
theorem receipts_change_after_price_reduction_and_sales_increase
  (original_price : ℝ)
  (original_sales : ℝ)
  (price_reduction_percent : ℝ)
  (sales_increase_percent : ℝ)
  (h1 : price_reduction_percent = 30)
  (h2 : sales_increase_percent = 50)
  : (((1 - price_reduction_percent / 100) * (1 + sales_increase_percent / 100) - 1) * 100 = 5) := by
  sorry

end receipts_change_after_price_reduction_and_sales_increase_l1326_132625


namespace water_evaporation_rate_l1326_132687

/-- Proves that given a bowl with 10 ounces of water, if 4% of the original amount
    evaporates over 50 days, then the amount of water that evaporates each day is 0.008 ounces. -/
theorem water_evaporation_rate (initial_water : ℝ) (days : ℕ) (evaporation_percent : ℝ) :
  initial_water = 10 →
  days = 50 →
  evaporation_percent = 4 →
  (evaporation_percent / 100 * initial_water) / days = 0.008 := by
  sorry

end water_evaporation_rate_l1326_132687


namespace chess_draw_probability_l1326_132619

theorem chess_draw_probability (prob_A_win prob_A_not_lose : ℝ) 
  (h1 : prob_A_win = 0.4)
  (h2 : prob_A_not_lose = 0.9) :
  prob_A_not_lose - prob_A_win = 0.5 := by
  sorry

end chess_draw_probability_l1326_132619


namespace repeating_decimal_to_fraction_l1326_132630

theorem repeating_decimal_to_fraction :
  ∃ (x : ℚ), (x = 2 + 35 / 99) ∧ (x = 233 / 99) := by sorry

end repeating_decimal_to_fraction_l1326_132630


namespace vasyas_numbers_l1326_132603

theorem vasyas_numbers (x y : ℝ) : x + y = x * y ∧ x + y = x / y ∧ x * y = x / y → x = (1 : ℝ) / 2 ∧ y = -1 := by
  sorry

end vasyas_numbers_l1326_132603


namespace positive_real_inequality_l1326_132661

theorem positive_real_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y^2016 ≥ 1) :
  x^2016 + y > 1 - 1/100 := by
  sorry

end positive_real_inequality_l1326_132661


namespace books_per_shelf_l1326_132611

theorem books_per_shelf (total_shelves : ℕ) (total_books : ℕ) (h1 : total_shelves = 8) (h2 : total_books = 32) :
  total_books / total_shelves = 4 := by
  sorry

end books_per_shelf_l1326_132611


namespace triangle_angle_B_l1326_132666

theorem triangle_angle_B (A B C : Real) (a b c : Real) : 
  0 < A ∧ A < π ∧
  0 < B ∧ B < π ∧
  0 < C ∧ C < π ∧
  A + B + C = π ∧
  a * Real.sin B * Real.cos C + c * Real.sin B * Real.cos A = 1/2 * b ∧
  a > b →
  B = π/6 := by sorry

end triangle_angle_B_l1326_132666


namespace infinite_series_sum_l1326_132639

theorem infinite_series_sum : 
  let series := fun k : ℕ => (3^(2^k)) / ((5^(2^k)) - 2)
  ∑' k, series k = 1 / (Real.sqrt 5 - 1) := by
  sorry

end infinite_series_sum_l1326_132639


namespace largest_subset_size_150_l1326_132614

/-- A function that returns the size of the largest subset of integers from 1 to n 
    where no member is 4 times another member -/
def largest_subset_size (n : ℕ) : ℕ := 
  sorry

/-- The theorem to be proved -/
theorem largest_subset_size_150 : largest_subset_size 150 = 142 := by
  sorry

end largest_subset_size_150_l1326_132614


namespace bus_passengers_l1326_132653

theorem bus_passengers (initial_students : ℕ) (num_stops : ℕ) : 
  initial_students = 64 →
  num_stops = 4 →
  (initial_students : ℚ) * (2/3)^num_stops = 1024/81 := by
  sorry

end bus_passengers_l1326_132653


namespace problem_solution_l1326_132674

-- Define the condition from the problem
def condition (m : ℝ) : Prop :=
  ∀ t : ℝ, |t + 3| - |t - 2| ≤ 6 * m - m^2

-- Define the theorem
theorem problem_solution :
  (∃ m : ℝ, condition m) →
  (∀ m : ℝ, condition m → 1 ≤ m ∧ m ≤ 5) ∧
  (∀ x y z : ℝ, 3*x + 4*y + 5*z = 5 → x^2 + y^2 + z^2 ≥ 1/2) :=
by sorry

end problem_solution_l1326_132674


namespace existence_equivalence_l1326_132658

theorem existence_equivalence (a b : ℤ) :
  (∃ c d : ℤ, a + b + c + d = 0 ∧ a * c + b * d = 0) ↔ (∃ k : ℤ, 2 * a * b = k * (a - b)) :=
by sorry

end existence_equivalence_l1326_132658


namespace equation_solution_l1326_132601

theorem equation_solution :
  ∃ x : ℚ, (x - 55) / 3 = (2 - 3*x + x^2) / 4 ∧ (x = 20/3 ∨ x = -11) :=
by sorry

end equation_solution_l1326_132601


namespace same_color_probability_l1326_132660

/-- The number of red plates -/
def red_plates : ℕ := 6

/-- The number of blue plates -/
def blue_plates : ℕ := 5

/-- The number of green plates -/
def green_plates : ℕ := 3

/-- The total number of plates -/
def total_plates : ℕ := red_plates + blue_plates + green_plates

/-- The number of ways to choose 3 plates from the total number of plates -/
def total_ways : ℕ := Nat.choose total_plates 3

/-- The number of ways to choose 3 red plates -/
def red_ways : ℕ := Nat.choose red_plates 3

/-- The number of ways to choose 3 blue plates -/
def blue_ways : ℕ := Nat.choose blue_plates 3

/-- The number of ways to choose 3 green plates -/
def green_ways : ℕ := Nat.choose green_plates 3

/-- The total number of favorable outcomes (all same color) -/
def favorable_outcomes : ℕ := red_ways + blue_ways + green_ways

/-- The probability of selecting three plates of the same color -/
theorem same_color_probability : 
  (favorable_outcomes : ℚ) / total_ways = 31 / 364 := by
  sorry

end same_color_probability_l1326_132660


namespace equal_coins_count_l1326_132665

/-- Represents the value of a coin in cents -/
def coin_value (coin_type : String) : ℕ :=
  match coin_type with
  | "nickel" => 5
  | "dime" => 10
  | "quarter" => 25
  | _ => 0

/-- Represents the total value of coins in cents -/
def total_value : ℕ := 120

/-- Represents the number of different types of coins -/
def num_coin_types : ℕ := 3

theorem equal_coins_count (num_each : ℕ) :
  (num_each * coin_value "nickel" +
   num_each * coin_value "dime" +
   num_each * coin_value "quarter" = total_value) →
  (num_each * num_coin_types = 9) := by
  sorry

#check equal_coins_count

end equal_coins_count_l1326_132665


namespace max_sides_equal_longest_diagonal_l1326_132643

/-- A convex polygon -/
structure ConvexPolygon where
  -- Add necessary fields here
  -- This is a placeholder structure

/-- The longest diagonal of a convex polygon -/
def longest_diagonal (p : ConvexPolygon) : ℝ :=
  sorry

/-- The number of sides equal to the longest diagonal in a convex polygon -/
def num_sides_equal_longest_diagonal (p : ConvexPolygon) : ℕ :=
  sorry

/-- Theorem: The maximum number of sides that can be equal to the longest diagonal in a convex polygon is 2 -/
theorem max_sides_equal_longest_diagonal (p : ConvexPolygon) :
  num_sides_equal_longest_diagonal p ≤ 2 :=
sorry

end max_sides_equal_longest_diagonal_l1326_132643


namespace afternoon_shells_l1326_132649

theorem afternoon_shells (morning_shells : ℕ) (total_shells : ℕ) 
  (h1 : morning_shells = 292) 
  (h2 : total_shells = 616) : 
  total_shells - morning_shells = 324 := by
sorry

end afternoon_shells_l1326_132649


namespace specific_bill_amount_l1326_132664

/-- Calculates the amount of a bill given its true discount, due time, and interest rate. -/
def bill_amount (true_discount : ℚ) (due_time : ℚ) (interest_rate : ℚ) : ℚ :=
  (true_discount * (100 + interest_rate * due_time)) / (interest_rate * due_time)

/-- Theorem stating that given the specific conditions, the bill amount is 1680. -/
theorem specific_bill_amount :
  let true_discount : ℚ := 180
  let due_time : ℚ := 9 / 12  -- 9 months expressed in years
  let interest_rate : ℚ := 16 -- 16% per annum
  bill_amount true_discount due_time interest_rate = 1680 :=
by sorry


end specific_bill_amount_l1326_132664


namespace parabola_sum_is_vertical_l1326_132629

/-- Original parabola function -/
def original_parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- Reflected and left-translated parabola -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 - b * (x + 3) + c

/-- Reflected and right-translated parabola -/
def g (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * (x - 4) + c

/-- Sum of f and g -/
def f_plus_g (a b c : ℝ) (x : ℝ) : ℝ := f a b c x + g a b c x

theorem parabola_sum_is_vertical (a b c : ℝ) :
  ∃ A C : ℝ, ∀ x : ℝ, f_plus_g a b c x = A * x^2 + C :=
sorry

end parabola_sum_is_vertical_l1326_132629


namespace range_of_k_is_real_l1326_132633

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the property of f being an increasing function
def IsIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- Theorem statement
theorem range_of_k_is_real (h : IsIncreasing f) : 
  ∀ k : ℝ, ∃ x : ℝ, f x = k :=
sorry

end range_of_k_is_real_l1326_132633


namespace widgets_sold_is_3125_l1326_132696

/-- Represents Jenna's wholesale business --/
structure WholesaleBusiness where
  buy_price : ℝ
  sell_price : ℝ
  rent : ℝ
  tax_rate : ℝ
  worker_salary : ℝ
  num_workers : ℕ
  profit_after_tax : ℝ

/-- Calculates the number of widgets sold given the business parameters --/
def widgets_sold (b : WholesaleBusiness) : ℕ :=
  sorry

/-- Theorem stating that the number of widgets sold is 3125 --/
theorem widgets_sold_is_3125 (jenna : WholesaleBusiness) 
  (h1 : jenna.buy_price = 3)
  (h2 : jenna.sell_price = 8)
  (h3 : jenna.rent = 10000)
  (h4 : jenna.tax_rate = 0.2)
  (h5 : jenna.worker_salary = 2500)
  (h6 : jenna.num_workers = 4)
  (h7 : jenna.profit_after_tax = 4000) :
  widgets_sold jenna = 3125 :=
sorry

end widgets_sold_is_3125_l1326_132696


namespace special_number_property_l1326_132609

/-- Sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Check if a natural number has all identical digits -/
def has_identical_digits (n : ℕ) : Prop := sorry

/-- The unique three-digit number satisfying the given conditions -/
def special_number : ℕ := 105

theorem special_number_property :
  ∃! (N : ℕ), 
    100 ≤ N ∧ N < 1000 ∧ 
    has_identical_digits (N + digit_sum N) ∧
    has_identical_digits (N - digit_sum N) ∧
    N = special_number := by
  sorry

end special_number_property_l1326_132609


namespace sequence_ratio_range_l1326_132616

theorem sequence_ratio_range (x y a₁ a₂ b₁ b₂ : ℝ) 
  (h_arith : a₁ - x = a₂ - a₁ ∧ a₂ - a₁ = y - a₂)
  (h_geom : b₁ / x = b₂ / b₁ ∧ b₂ / b₁ = y / b₂) :
  (a₁ + a₂)^2 / (b₁ * b₂) ≥ 4 ∨ (a₁ + a₂)^2 / (b₁ * b₂) ≤ 0 :=
by sorry

end sequence_ratio_range_l1326_132616


namespace max_sections_five_lines_l1326_132693

/-- The maximum number of sections a rectangle can be divided into by n line segments -/
def max_sections (n : ℕ) : ℕ := n * (n + 1) / 2 + 1

/-- Theorem: The maximum number of sections a rectangle can be divided into by 5 line segments is 16 -/
theorem max_sections_five_lines :
  max_sections 5 = 16 := by
  sorry

end max_sections_five_lines_l1326_132693


namespace log_difference_cube_l1326_132669

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Theorem statement
theorem log_difference_cube (x y a : ℝ) (h : x > 0) (h' : y > 0) :
  lg x - lg y = a → lg ((x / 2) ^ 3) - lg ((y / 2) ^ 3) = 3 * a := by
  sorry

end log_difference_cube_l1326_132669


namespace arithmetic_sequence_sum_l1326_132668

theorem arithmetic_sequence_sum (a : ℕ → ℤ) (S : ℕ → ℤ) : 
  (∀ n, S n = n * a 1 + (n * (n - 1) / 2) * (a 2 - a 1)) →  -- Definition of S_n
  a 1 = -2011 →                                            -- Given a_1
  (S 2010 / 2010) - (S 2008 / 2008) = 2 →                  -- Given condition
  S 2011 = -2011 :=                                        -- Conclusion to prove
by
  sorry

end arithmetic_sequence_sum_l1326_132668


namespace cubic_equation_solution_l1326_132615

theorem cubic_equation_solution :
  let f : ℝ → ℝ := λ x => (x + 1)^3 + (3 - x)^3
  ∃ (x₁ x₂ : ℝ), 
    (f x₁ = 35 ∧ f x₂ = 35) ∧ 
    (x₁ = 1 + Real.sqrt (19/3) / 2) ∧ 
    (x₂ = 1 - Real.sqrt (19/3) / 2) ∧
    (∀ x : ℝ, f x = 35 → x = x₁ ∨ x = x₂) :=
by sorry

end cubic_equation_solution_l1326_132615


namespace no_square_on_four_circles_l1326_132652

/-- Represents a circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a square in a plane -/
structure Square where
  vertices : Fin 4 → ℝ × ℝ

/-- Checks if four radii form a strictly increasing arithmetic progression -/
def is_strict_arithmetic_progression (r₁ r₂ r₃ r₄ : ℝ) : Prop :=
  ∃ (a d : ℝ), d > 0 ∧ r₁ = a ∧ r₂ = a + d ∧ r₃ = a + 2*d ∧ r₄ = a + 3*d

/-- Checks if a point lies on a circle -/
def point_on_circle (p : ℝ × ℝ) (c : Circle) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

/-- Main theorem statement -/
theorem no_square_on_four_circles (c₁ c₂ c₃ c₄ : Circle) 
  (h_common_center : c₁.center = c₂.center ∧ c₂.center = c₃.center ∧ c₃.center = c₄.center)
  (h_radii : is_strict_arithmetic_progression c₁.radius c₂.radius c₃.radius c₄.radius) :
  ¬ ∃ (s : Square), 
    (point_on_circle (s.vertices 0) c₁) ∧
    (point_on_circle (s.vertices 1) c₂) ∧
    (point_on_circle (s.vertices 2) c₃) ∧
    (point_on_circle (s.vertices 3) c₄) :=
by sorry

end no_square_on_four_circles_l1326_132652


namespace fence_cost_per_foot_l1326_132689

theorem fence_cost_per_foot 
  (area : ℝ) 
  (total_cost : ℝ) 
  (h1 : area = 289) 
  (h2 : total_cost = 3944) : 
  total_cost / (4 * Real.sqrt area) = 58 := by
sorry

end fence_cost_per_foot_l1326_132689


namespace expression_equals_two_l1326_132604

/-- Given real numbers a, b, and c satisfying two conditions, 
    prove that a certain expression equals 2 -/
theorem expression_equals_two (a b c : ℝ) 
  (h1 : a * c / (a + b) + b * a / (b + c) + c * b / (c + a) = 8)
  (h2 : b * c / (a + b) + c * a / (b + c) + a * b / (c + a) = 7) :
  b / (a + b) + c / (b + c) + a / (c + a) = 2 := by
  sorry

end expression_equals_two_l1326_132604


namespace hotel_profit_equation_correct_l1326_132688

/-- Represents a hotel's pricing and occupancy model -/
structure Hotel where
  baseRooms : ℕ
  basePrice : ℝ
  costPerRoom : ℝ
  vacancyRate : ℝ
  priceIncrease : ℝ
  desiredProfit : ℝ

/-- The profit equation for the hotel -/
def profitEquation (h : Hotel) : Prop :=
  (h.basePrice + h.priceIncrease - h.costPerRoom) * 
  (h.baseRooms - h.priceIncrease / h.vacancyRate) = h.desiredProfit

/-- Theorem stating that the given equation correctly represents the hotel's profit scenario -/
theorem hotel_profit_equation_correct (h : Hotel) 
  (hRooms : h.baseRooms = 50)
  (hBasePrice : h.basePrice = 180)
  (hCost : h.costPerRoom = 20)
  (hVacancy : h.vacancyRate = 10)
  (hProfit : h.desiredProfit = 10890) :
  profitEquation h := by sorry

end hotel_profit_equation_correct_l1326_132688


namespace chloe_trivia_score_l1326_132691

/-- The total points scored in a trivia game with three rounds -/
def trivia_game_score (round1 : Int) (round2 : Int) (round3 : Int) : Int :=
  round1 + round2 + round3

/-- Theorem: The total points at the end of Chloe's trivia game is 86 -/
theorem chloe_trivia_score : trivia_game_score 40 50 (-4) = 86 := by
  sorry

end chloe_trivia_score_l1326_132691


namespace jerky_order_fulfillment_l1326_132650

/-- Calculates the number of days required to fulfill a jerky order -/
def days_to_fulfill_order (order : ℕ) (in_stock : ℕ) (production_rate : ℕ) : ℕ :=
  ((order - in_stock) + production_rate - 1) / production_rate

/-- Theorem: Given the specific conditions, it takes 4 days to fulfill the order -/
theorem jerky_order_fulfillment :
  days_to_fulfill_order 60 20 10 = 4 := by
  sorry

end jerky_order_fulfillment_l1326_132650


namespace zara_sheep_count_l1326_132663

/-- The number of sheep Zara bought -/
def num_sheep : ℕ := 7

/-- The number of cows Zara bought -/
def num_cows : ℕ := 24

/-- The number of goats Zara bought -/
def num_goats : ℕ := 113

/-- The number of groups for transporting animals -/
def num_groups : ℕ := 3

/-- The number of animals in each group -/
def animals_per_group : ℕ := 48

theorem zara_sheep_count :
  num_sheep + num_cows + num_goats = num_groups * animals_per_group := by
  sorry

end zara_sheep_count_l1326_132663


namespace least_addition_for_divisibility_l1326_132682

theorem least_addition_for_divisibility : 
  ∃ (n : ℕ), n = 15 ∧ 
  (∀ (m : ℕ), (433124 + m) % 17 = 0 → m ≥ n) ∧ 
  (433124 + n) % 17 = 0 := by
  sorry

end least_addition_for_divisibility_l1326_132682


namespace problem_1_l1326_132617

theorem problem_1 : Real.sqrt 18 - 4 * Real.sqrt (1/2) + Real.sqrt 24 / Real.sqrt 3 = 3 * Real.sqrt 2 := by
  sorry

end problem_1_l1326_132617


namespace parabola_intercepts_sum_l1326_132600

def parabola (y : ℝ) : ℝ := 2 * y^2 - 6 * y + 3

theorem parabola_intercepts_sum :
  ∃ (a b c : ℝ),
    (parabola 0 = a) ∧
    (parabola b = 0) ∧
    (parabola c = 0) ∧
    (b ≠ c) ∧
    (a + b + c = 6) := by
  sorry

end parabola_intercepts_sum_l1326_132600


namespace four_nested_s_of_6_l1326_132645

-- Define the function s
def s (x : ℚ) : ℚ := 1 / (2 - x)

-- State the theorem
theorem four_nested_s_of_6 : s (s (s (s 6))) = 14 / 19 := by sorry

end four_nested_s_of_6_l1326_132645


namespace product_in_B_l1326_132608

-- Define the sets A and B
def A : Set ℤ := {x | ∃ (a b k m : ℤ), x = m * a^2 + k * a * b + m * b^2}
def B : Set ℤ := {x | ∃ (a b k m : ℤ), x = a^2 + k * a * b + m^2 * b^2}

-- State the theorem
theorem product_in_B (x y : ℤ) (hx : x ∈ A) (hy : y ∈ A) : x * y ∈ B := by
  sorry

end product_in_B_l1326_132608


namespace least_three_digit_multiple_of_nine_l1326_132673

theorem least_three_digit_multiple_of_nine : ∃ n : ℕ, 
  (n ≥ 100 ∧ n ≤ 999) ∧ 
  n % 9 = 0 ∧
  ∀ m : ℕ, (m ≥ 100 ∧ m ≤ 999 ∧ m % 9 = 0) → n ≤ m :=
by
  -- The proof goes here
  sorry

end least_three_digit_multiple_of_nine_l1326_132673


namespace repeating_decimal_quotient_l1326_132679

/-- Represents a repeating decimal with a two-digit repetend -/
def RepeatingDecimal (n : ℕ) : ℚ :=
  n / 99

theorem repeating_decimal_quotient :
  (RepeatingDecimal 54) / (RepeatingDecimal 18) = 3 := by
  sorry

end repeating_decimal_quotient_l1326_132679


namespace contest_order_l1326_132631

-- Define the contestants
variable (Andy Beth Carol Dave : ℝ)

-- Define the conditions
axiom sum_equality : Andy + Carol = Beth + Dave
axiom interchange_inequality : Beth + Andy > Dave + Carol
axiom carol_highest : Carol > Andy + Beth
axiom nonnegative_scores : Andy ≥ 0 ∧ Beth ≥ 0 ∧ Carol ≥ 0 ∧ Dave ≥ 0

-- Theorem to prove
theorem contest_order : Carol > Beth ∧ Beth > Andy ∧ Andy > Dave := by
  sorry

end contest_order_l1326_132631


namespace ball_distribution_theorem_l1326_132671

def num_white_balls : ℕ := 3
def num_red_balls : ℕ := 4
def num_yellow_balls : ℕ := 5
def num_boxes : ℕ := 3

def distribute_balls : ℕ := (Nat.choose (num_boxes + num_white_balls - 1) (num_boxes - 1)) *
                             (Nat.choose (num_boxes + num_red_balls - 1) (num_boxes - 1)) *
                             (Nat.choose (num_boxes + num_yellow_balls - 1) (num_boxes - 1))

theorem ball_distribution_theorem : distribute_balls = 3150 := by
  sorry

end ball_distribution_theorem_l1326_132671


namespace sqrt_3_plus_sqrt_7_less_than_2sqrt_5_l1326_132621

theorem sqrt_3_plus_sqrt_7_less_than_2sqrt_5 : Real.sqrt 3 + Real.sqrt 7 < 2 * Real.sqrt 5 := by
  sorry

end sqrt_3_plus_sqrt_7_less_than_2sqrt_5_l1326_132621


namespace dads_strawberry_weight_l1326_132602

/-- Given the total weight of strawberries picked by Marco and his dad,
    and the weight of Marco's strawberries, calculate the weight of his dad's strawberries. -/
theorem dads_strawberry_weight 
  (total_weight : ℕ) 
  (marcos_weight : ℕ) 
  (h1 : total_weight = 20)
  (h2 : marcos_weight = 3) : 
  total_weight - marcos_weight = 17 := by
  sorry

#check dads_strawberry_weight

end dads_strawberry_weight_l1326_132602


namespace pollard_complexity_l1326_132657

/-- Represents the state of the algorithm at each iteration -/
structure AlgorithmState where
  u : Nat
  v : Nat

/-- The update function for the algorithm -/
def update (state : AlgorithmState) : AlgorithmState := sorry

/-- The main loop of the algorithm -/
def mainLoop (n : Nat) (initialState : AlgorithmState) : Nat := sorry

theorem pollard_complexity {n p : Nat} (hprime : Nat.Prime p) (hfactor : p ∣ n) :
  ∃ (c : Nat), mainLoop n (AlgorithmState.mk 1 1) ≤ 2 * p ∧
  mainLoop n (AlgorithmState.mk 1 1) ≤ c * p * (Nat.log n)^2 := by
  sorry

end pollard_complexity_l1326_132657


namespace salary_increase_percentage_l1326_132670

/-- Proves that the percentage increase resulting in a $324 weekly salary is 8%,
    given that a 10% increase results in a $330 weekly salary. -/
theorem salary_increase_percentage (current_salary : ℝ) : 
  (current_salary * 1.1 = 330) →
  (current_salary * (1 + 0.08) = 324) := by
  sorry

end salary_increase_percentage_l1326_132670


namespace inequality_proof_l1326_132690

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x^2 + y^4 + z^6 ≥ x*y^2 + y^2*z^3 + x*z^3 := by
  sorry

end inequality_proof_l1326_132690


namespace oplus_nested_equation_l1326_132606

def oplus (x y : ℝ) : ℝ := x^2 + 2*y

theorem oplus_nested_equation (a : ℝ) : oplus a (oplus a a) = 3*a^2 + 4*a := by
  sorry

end oplus_nested_equation_l1326_132606


namespace antonella_remaining_money_l1326_132626

/-- Represents the types of Canadian coins -/
inductive CanadianCoin
  | Loonie
  | Toonie

/-- The value of a Canadian coin in dollars -/
def coin_value : CanadianCoin → ℕ
  | CanadianCoin.Loonie => 1
  | CanadianCoin.Toonie => 2

/-- Calculates the total value of coins -/
def total_value (coins : List CanadianCoin) : ℕ :=
  coins.map coin_value |>.sum

theorem antonella_remaining_money :
  let total_coins : ℕ := 10
  let toonie_count : ℕ := 4
  let loonie_count : ℕ := total_coins - toonie_count
  let initial_coins : List CanadianCoin := 
    List.replicate toonie_count CanadianCoin.Toonie ++ List.replicate loonie_count CanadianCoin.Loonie
  let frappuccino_cost : ℕ := 3
  total_value initial_coins - frappuccino_cost = 11 := by
  sorry

end antonella_remaining_money_l1326_132626


namespace problem_solution_l1326_132647

def f (x : ℝ) := x^2 - 2*x

theorem problem_solution :
  (∀ x : ℝ, (|f x| + |x^2 + 2*x| ≥ 6*|x|) ↔ (x ≤ -3 ∨ x ≥ 3 ∨ x = 0)) ∧
  (∀ x a : ℝ, |x - a| < 1 → |f x - f a| < 2*|a| + 3) :=
sorry

end problem_solution_l1326_132647


namespace root_cubic_expression_l1326_132699

theorem root_cubic_expression (m : ℝ) : 
  m^2 + 3*m - 2022 = 0 → m^3 + 4*m^2 - 2019*m - 2023 = -1 := by
  sorry

end root_cubic_expression_l1326_132699


namespace sqrt_eight_and_nine_sixteenths_l1326_132677

theorem sqrt_eight_and_nine_sixteenths (x : ℝ) : 
  x = Real.sqrt (8 + 9/16) → x = Real.sqrt 137 / 4 := by
  sorry

end sqrt_eight_and_nine_sixteenths_l1326_132677


namespace journey_length_is_70_l1326_132623

-- Define the journey
def Journey (length : ℝ) : Prop :=
  -- Time taken at 40 kmph
  let time_at_40 := length / 40
  -- Time taken at 35 kmph
  let time_at_35 := length / 35
  -- The difference in time is 0.25 hours (15 minutes)
  time_at_35 - time_at_40 = 0.25

-- Theorem stating that the journey length is 70 km
theorem journey_length_is_70 : 
  ∃ (length : ℝ), Journey length ∧ length = 70 :=
sorry

end journey_length_is_70_l1326_132623


namespace students_without_gift_l1326_132694

theorem students_without_gift (total_students : ℕ) (h : total_students = 2016) :
  (∃ (no_gift : ℕ), no_gift = Nat.totient total_students ∧
    ∀ (n : ℕ), n ≥ 2 → no_gift = total_students - (total_students / n) * n) := by
  sorry

end students_without_gift_l1326_132694


namespace necessary_not_sufficient_condition_l1326_132636

theorem necessary_not_sufficient_condition (a : ℝ) (h : a ≠ 0) :
  (∀ a, a > 1 → 1/a < 1) ∧ (∃ a, 1/a < 1 ∧ a ≤ 1) :=
by sorry

end necessary_not_sufficient_condition_l1326_132636


namespace amys_house_height_l1326_132656

/-- The height of Amy's house given shadow lengths -/
theorem amys_house_height (house_shadow : ℝ) (tree_height : ℝ) (tree_shadow : ℝ)
  (h1 : house_shadow = 63)
  (h2 : tree_height = 14)
  (h3 : tree_shadow = 28)
  : ∃ (house_height : ℝ), 
    (house_height / tree_height = house_shadow / tree_shadow) ∧ 
    (round house_height = 32) := by
  sorry

end amys_house_height_l1326_132656


namespace max_m_value_l1326_132628

/-- A quadratic function satisfying specific conditions -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0
  symmetry : ∀ x : ℝ, a * (x - 4)^2 + b * (x - 4) + c = a * (2 - x)^2 + b * (2 - x) + c
  inequality : ∀ x ∈ Set.Ioo 0 2, a * x^2 + b * x + c ≤ ((x + 1) / 2)^2
  min_value : ∃ x : ℝ, ∀ y : ℝ, a * x^2 + b * x + c ≤ a * y^2 + b * y + c ∧ a * x^2 + b * x + c = 0

/-- The theorem stating the maximum value of m -/
theorem max_m_value (f : QuadraticFunction) :
  ∃ m : ℝ, m = 9 ∧ m > 1 ∧
  (∀ m' > m, ¬∃ t : ℝ, ∀ x ∈ Set.Icc 1 m', f.a * (x + t)^2 + f.b * (x + t) + f.c ≤ x) ∧
  (∃ t : ℝ, ∀ x ∈ Set.Icc 1 m, f.a * (x + t)^2 + f.b * (x + t) + f.c ≤ x) :=
sorry

end max_m_value_l1326_132628


namespace square_minus_product_plus_square_l1326_132681

theorem square_minus_product_plus_square : 7^2 - 5*6 + 6^2 = 55 := by
  sorry

end square_minus_product_plus_square_l1326_132681


namespace angle_D_measure_l1326_132683

-- Define the pentagon and its properties
structure Pentagon where
  A : ℝ  -- Measure of angle A
  B : ℝ  -- Measure of angle B
  C : ℝ  -- Measure of angle C
  D : ℝ  -- Measure of angle D
  E : ℝ  -- Measure of angle E
  convex : A > 0 ∧ B > 0 ∧ C > 0 ∧ D > 0 ∧ E > 0
  sum_angles : A + B + C + D + E = 540
  congruent_ABC : A = B ∧ B = C
  congruent_DE : D = E
  A_less_D : A = D - 40

-- Theorem statement
theorem angle_D_measure (p : Pentagon) : p.D = 132 := by
  sorry

end angle_D_measure_l1326_132683


namespace light_ray_reflection_angle_l1326_132624

/-- Regular hexagon with mirrored inner surface -/
structure RegularHexagon :=
  (side : ℝ)
  (A B C D E F : ℝ × ℝ)

/-- Light ray path in the hexagon -/
structure LightRayPath (hex : RegularHexagon) :=
  (M N : ℝ × ℝ)
  (start_at_A : M.1 = hex.A.1 ∨ M.2 = hex.A.2)
  (end_at_D : N.1 = hex.D.1 ∨ N.2 = hex.D.2)
  (on_sides : (M.1 = hex.A.1 ∨ M.1 = hex.B.1 ∨ M.2 = hex.A.2 ∨ M.2 = hex.B.2) ∧
              (N.1 = hex.B.1 ∨ N.1 = hex.C.1 ∨ N.2 = hex.B.2 ∨ N.2 = hex.C.2))

/-- Main theorem -/
theorem light_ray_reflection_angle (hex : RegularHexagon) (path : LightRayPath hex) :
  let tan_EAM := (hex.E.2 - hex.A.2) / (hex.E.1 - hex.A.1)
  tan_EAM = 1 / (3 * Real.sqrt 3) := by sorry

end light_ray_reflection_angle_l1326_132624


namespace hyperbola_eccentricity_l1326_132672

/-- The eccentricity of a hyperbola with specific properties -/
theorem hyperbola_eccentricity (a b c : ℝ) : 
  a > 0 → 
  b > 0 → 
  c > 0 →
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) → 
  (b * c / Real.sqrt (a^2 + b^2) = Real.sqrt 3 / 2 * c) →
  c / a = 2 := by
  sorry

#check hyperbola_eccentricity

end hyperbola_eccentricity_l1326_132672


namespace orangeade_price_day2_l1326_132651

/-- Represents the price of orangeade per glass on a given day -/
structure OrangeadePrice where
  price : ℝ
  day : Nat

/-- Represents the volume of orangeade made on a given day -/
structure OrangeadeVolume where
  volume : ℝ
  day : Nat

/-- Calculates the revenue from selling orangeade -/
def revenue (price : OrangeadePrice) (volume : OrangeadeVolume) : ℝ :=
  price.price * volume.volume

theorem orangeade_price_day2 
  (price_day1 : OrangeadePrice)
  (price_day2 : OrangeadePrice)
  (volume_day1 : OrangeadeVolume)
  (volume_day2 : OrangeadeVolume)
  (h1 : price_day1.day = 1)
  (h2 : price_day2.day = 2)
  (h3 : volume_day1.day = 1)
  (h4 : volume_day2.day = 2)
  (h5 : price_day1.price = 0.82)
  (h6 : volume_day2.volume = (3/2) * volume_day1.volume)
  (h7 : revenue price_day1 volume_day1 = revenue price_day2 volume_day2) :
  price_day2.price = (2 * 0.82) / 3 := by
  sorry

end orangeade_price_day2_l1326_132651


namespace max_wins_l1326_132632

/-- 
Given that the ratio of Chloe's wins to Max's wins is 8:3, and Chloe won 24 times,
prove that Max won 9 times.
-/
theorem max_wins (chloe_wins : ℕ) (max_wins : ℕ) 
  (h1 : chloe_wins = 24)
  (h2 : chloe_wins * 3 = max_wins * 8) : 
  max_wins = 9 := by
  sorry

end max_wins_l1326_132632


namespace vector_properties_l1326_132698

/-- Given vectors a and b in ℝ², prove they are perpendicular and satisfy certain magnitude properties -/
theorem vector_properties (a b : ℝ × ℝ) (h1 : a = (2, 4)) (h2 : b = (-2, 1)) :
  (a.1 * b.1 + a.2 * b.2 = 0) ∧ 
  (Real.sqrt ((a.1 + b.1)^2 + (a.2 + b.2)^2) = 5) ∧
  (Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 5) := by
  sorry

end vector_properties_l1326_132698


namespace painted_cube_probability_l1326_132684

theorem painted_cube_probability : 
  let cube_side : ℕ := 5
  let total_cubes : ℕ := cube_side ^ 3
  let painted_faces : ℕ := 2
  let two_face_painted : ℕ := 4 * (cube_side - 1)
  let no_face_painted : ℕ := total_cubes - 2 * cube_side^2 + two_face_painted
  let total_combinations : ℕ := total_cubes.choose 2
  let favorable_outcomes : ℕ := two_face_painted * no_face_painted
  (favorable_outcomes : ℚ) / total_combinations = 728 / 3875 := by
sorry

end painted_cube_probability_l1326_132684


namespace sum_of_reciprocals_of_quadratic_roots_l1326_132618

theorem sum_of_reciprocals_of_quadratic_roots :
  let a := 1
  let b := -17
  let c := 8
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  (1 / r₁ + 1 / r₂) = 17 / 8 := by
  sorry

end sum_of_reciprocals_of_quadratic_roots_l1326_132618


namespace power_equality_l1326_132675

theorem power_equality (p : ℕ) : 16^5 = 4^p → p = 10 := by
  sorry

end power_equality_l1326_132675


namespace max_sign_changes_is_n_minus_one_sign_changes_bounded_l1326_132637

/-- The maximum number of sign changes for the first element in a sequence of n real numbers 
    under the described averaging process. -/
def max_sign_changes (n : ℕ) : ℕ :=
  n - 1

/-- The theorem stating that the maximum number of sign changes for the first element
    is n-1 for any positive integer n. -/
theorem max_sign_changes_is_n_minus_one (n : ℕ) (hn : n > 0) : 
  max_sign_changes n = n - 1 := by
  sorry

/-- A helper function to represent the averaging operation on a sequence of real numbers. -/
def average_operation (seq : List ℝ) (i : ℕ) : List ℝ :=
  sorry

/-- A predicate to check if a number has changed sign. -/
def sign_changed (a b : ℝ) : Prop :=
  (a ≥ 0 ∧ b < 0) ∨ (a < 0 ∧ b ≥ 0)

/-- A function to count the number of sign changes in a₁ after a sequence of operations. -/
def count_sign_changes (initial_seq : List ℝ) (operations : List ℕ) : ℕ :=
  sorry

/-- The main theorem stating that for any initial sequence and any sequence of operations,
    the number of sign changes in a₁ is at most n-1. -/
theorem sign_changes_bounded (n : ℕ) (hn : n > 0) 
  (initial_seq : List ℝ) (h_seq : initial_seq.length = n)
  (operations : List ℕ) :
  count_sign_changes initial_seq operations ≤ max_sign_changes n := by
  sorry

end max_sign_changes_is_n_minus_one_sign_changes_bounded_l1326_132637


namespace complex_number_modulus_l1326_132644

theorem complex_number_modulus : 
  let z : ℂ := (-1 - 2*I) / (1 - I)^2
  ‖z‖ = Real.sqrt 5 / 2 := by
sorry

end complex_number_modulus_l1326_132644


namespace factor_difference_of_squares_l1326_132635

theorem factor_difference_of_squares (y : ℝ) :
  25 - 16 * y^2 = (5 - 4*y) * (5 + 4*y) := by
  sorry

end factor_difference_of_squares_l1326_132635


namespace mac_loses_three_dollars_l1326_132680

/-- Represents the value of a coin in cents -/
def coin_value (coin : String) : ℕ :=
  match coin with
  | "quarter" => 25
  | "dime" => 10
  | "nickel" => 5
  | _ => 0

/-- Calculates the total value of coins in dollars -/
def total_value (coin : String) (count : ℕ) : ℚ :=
  (coin_value coin * count : ℚ) / 100

/-- Represents Mac's trade of dimes for quarters -/
def dimes_for_quarter : ℕ := 3

/-- Represents Mac's trade of nickels for quarters -/
def nickels_for_quarter : ℕ := 7

/-- Number of quarters Mac trades for using dimes -/
def quarters_from_dimes : ℕ := 20

/-- Number of quarters Mac trades for using nickels -/
def quarters_from_nickels : ℕ := 20

/-- Theorem stating that Mac loses $3.00 in his trades -/
theorem mac_loses_three_dollars :
  total_value "quarter" (quarters_from_dimes + quarters_from_nickels) -
  (total_value "dime" (dimes_for_quarter * quarters_from_dimes) +
   total_value "nickel" (nickels_for_quarter * quarters_from_nickels)) = -3 := by
  sorry

end mac_loses_three_dollars_l1326_132680


namespace triangle_problem_l1326_132685

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_problem (t : Triangle) 
  (h1 : t.c = 5/2)
  (h2 : t.b = Real.sqrt 6)
  (h3 : 4 * t.a - 3 * Real.sqrt 6 * Real.cos t.A = 0) :
  t.a = 3/2 ∧ t.B = 2 * t.A := by
  sorry

end triangle_problem_l1326_132685


namespace rectangle_area_change_l1326_132697

theorem rectangle_area_change (initial_area : ℝ) : 
  initial_area = 500 →
  (0.8 * 1.2 * initial_area) = 480 := by
sorry

end rectangle_area_change_l1326_132697


namespace sqrt_10_irrational_l1326_132638

theorem sqrt_10_irrational : Irrational (Real.sqrt 10) := by
  sorry

end sqrt_10_irrational_l1326_132638


namespace apple_distribution_l1326_132659

theorem apple_distribution (x y : ℕ) : 
  (y - 1 = x + 1) →
  (y + 1 = 3 * (x - 1)) →
  (x = 3 ∧ y = 5) :=
by sorry

end apple_distribution_l1326_132659


namespace binomial_coefficient_n_plus_one_choose_two_l1326_132692

theorem binomial_coefficient_n_plus_one_choose_two (n : ℕ) : 
  Nat.choose (n + 1) 2 = (n + 1) * n / 2 := by
  sorry

end binomial_coefficient_n_plus_one_choose_two_l1326_132692
