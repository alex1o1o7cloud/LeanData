import Mathlib

namespace factory_max_profit_l805_80591

/-- The annual profit function for the factory -/
noncomputable def L (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 80 then
    -(1/3) * x^2 + 40 * x - 250
  else if x ≥ 80 then
    50 * x - 10000 / x + 1200
  else
    0

/-- The maximum profit and corresponding production level -/
theorem factory_max_profit :
  (∃ (x : ℝ), L x = 1000 ∧ x = 100) ∧
  (∀ (y : ℝ), L y ≤ 1000) := by
  sorry

end factory_max_profit_l805_80591


namespace train_speed_l805_80504

/-- The speed of a train crossing a bridge -/
theorem train_speed (train_length bridge_length crossing_time : ℝ) 
  (h1 : train_length = 100)
  (h2 : bridge_length = 300)
  (h3 : crossing_time = 12) :
  (train_length + bridge_length) / crossing_time = 400 / 12 := by
sorry

end train_speed_l805_80504


namespace triangle_5_7_14_not_exists_l805_80522

/-- Triangle inequality theorem: the sum of the lengths of any two sides of a triangle
    must be greater than the length of the remaining side. -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- A function that determines if a triangle with given side lengths can exist. -/
def triangle_exists (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

/-- Theorem stating that a triangle with side lengths 5, 7, and 14 cannot exist. -/
theorem triangle_5_7_14_not_exists : ¬ triangle_exists 5 7 14 := by
  sorry

end triangle_5_7_14_not_exists_l805_80522


namespace probability_first_box_given_defective_l805_80536

def box1_total : ℕ := 5
def box1_defective : ℕ := 2
def box2_total : ℕ := 10
def box2_defective : ℕ := 3

def prob_select_box1 : ℚ := 1/2
def prob_select_box2 : ℚ := 1/2

def prob_defective_given_box1 : ℚ := box1_defective / box1_total
def prob_defective_given_box2 : ℚ := box2_defective / box2_total

theorem probability_first_box_given_defective :
  (prob_select_box1 * prob_defective_given_box1) /
  (prob_select_box1 * prob_defective_given_box1 + prob_select_box2 * prob_defective_given_box2) = 4/7 :=
by sorry

end probability_first_box_given_defective_l805_80536


namespace polar_curve_is_circle_l805_80598

/-- The curve defined by the polar equation r = 1 / (sin θ + cos θ) is a circle. -/
theorem polar_curve_is_circle :
  ∀ θ : ℝ, ∃ r : ℝ, r = 1 / (Real.sin θ + Real.cos θ) → ∃ c x₀ y₀ : ℝ, 
    (r * Real.cos θ - x₀)^2 + (r * Real.sin θ - y₀)^2 = c^2 := by
  sorry

end polar_curve_is_circle_l805_80598


namespace boy_age_proof_l805_80567

/-- Given a group of boys with specific average ages, prove the age of the boy not in either subgroup -/
theorem boy_age_proof (total_boys : ℕ) (total_avg : ℚ) (first_six_avg : ℚ) (last_six_avg : ℚ) :
  total_boys = 13 ∧ 
  total_avg = 50 ∧ 
  first_six_avg = 49 ∧ 
  last_six_avg = 52 →
  ∃ (middle_boy_age : ℚ), middle_boy_age = 50 :=
by sorry


end boy_age_proof_l805_80567


namespace average_speed_two_walks_l805_80515

theorem average_speed_two_walks 
  (v₁ v₂ t₁ t₂ : ℝ) 
  (h₁ : t₁ > 0) 
  (h₂ : t₂ > 0) :
  let d₁ := v₁ * t₁
  let d₂ := v₂ * t₂
  let total_distance := d₁ + d₂
  let total_time := t₁ + t₂
  (total_distance / total_time) = (v₁ * t₁ + v₂ * t₂) / (t₁ + t₂) := by
sorry

end average_speed_two_walks_l805_80515


namespace second_student_marks_l805_80596

/-- Proves that given two students' marks satisfying specific conditions, 
    the student with the lower score obtained 33 marks. -/
theorem second_student_marks : 
  ∀ (x y : ℝ), 
  x = y + 9 →  -- First student scored 9 marks more
  x = 0.56 * (x + y) →  -- Higher score is 56% of sum
  y = 33 :=
by
  sorry

end second_student_marks_l805_80596


namespace mn_value_l805_80533

theorem mn_value (m n : ℕ+) (h : m.val^4 - n.val^4 = 3439) : m.val * n.val = 90 := by
  sorry

end mn_value_l805_80533


namespace three_person_subcommittees_from_eight_l805_80527

-- Define the number of people in the main committee
def n : ℕ := 8

-- Define the number of people to be selected for each sub-committee
def k : ℕ := 3

-- Define the combination function
def combination (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- Theorem statement
theorem three_person_subcommittees_from_eight :
  combination n k = 56 := by
  sorry

end three_person_subcommittees_from_eight_l805_80527


namespace shelbys_scooter_problem_l805_80558

/-- Shelby's scooter problem -/
theorem shelbys_scooter_problem 
  (speed_no_rain : ℝ) 
  (speed_rain : ℝ) 
  (total_distance : ℝ) 
  (total_time : ℝ) 
  (h1 : speed_no_rain = 25)
  (h2 : speed_rain = 15)
  (h3 : total_distance = 18)
  (h4 : total_time = 36)
  : ∃ (time_no_rain : ℝ), 
    time_no_rain = 6 ∧ 
    speed_no_rain * (time_no_rain / 60) + speed_rain * ((total_time - time_no_rain) / 60) = total_distance :=
by
  sorry


end shelbys_scooter_problem_l805_80558


namespace root_sum_absolute_value_l805_80592

theorem root_sum_absolute_value (n : ℤ) (p q r : ℤ) : 
  (∀ x : ℤ, x^3 - 2027*x + n = 0 ↔ x = p ∨ x = q ∨ x = r) →
  |p| + |q| + |r| = 98 := by
  sorry

end root_sum_absolute_value_l805_80592


namespace divisible_by_seven_l805_80531

theorem divisible_by_seven (n : ℕ) : 7 ∣ (3^(12*n + 1) + 2^(6*n + 2)) := by
  sorry

end divisible_by_seven_l805_80531


namespace inequality_proof_l805_80546

theorem inequality_proof (a b c : ℝ) 
  (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) : a * b > a * c := by
  sorry

end inequality_proof_l805_80546


namespace arithmetic_expression_proof_l805_80518

theorem arithmetic_expression_proof : (6 + 6 * 3 - 3) / 3 = 7 := by
  sorry

end arithmetic_expression_proof_l805_80518


namespace total_cost_is_correct_l805_80523

def off_rack_suit_price : ℝ := 300
def tailored_suit_price (off_rack_price : ℝ) : ℝ := 3 * off_rack_price + 200
def dress_shirt_price : ℝ := 80
def shoes_price : ℝ := 120
def tie_price : ℝ := 40
def discount_rate : ℝ := 0.1
def sales_tax_rate : ℝ := 0.08
def shipping_fee : ℝ := 25

def total_cost : ℝ :=
  let discounted_suit_price := off_rack_suit_price * (1 - discount_rate)
  let suits_cost := off_rack_suit_price + discounted_suit_price
  let tailored_suit_cost := tailored_suit_price off_rack_suit_price
  let accessories_cost := dress_shirt_price + shoes_price + tie_price
  let subtotal := suits_cost + tailored_suit_cost + accessories_cost
  let tax := subtotal * sales_tax_rate
  subtotal + tax + shipping_fee

theorem total_cost_is_correct : total_cost = 2087.80 := by
  sorry

end total_cost_is_correct_l805_80523


namespace parabola_values_l805_80594

/-- A parabola passing through specific points -/
structure Parabola where
  a : ℝ
  b : ℝ
  eq : ℝ → ℝ := λ x => x^2 + a * x + b
  point1 : eq 2 = 20
  point2 : eq (-2) = 0
  point3 : eq 0 = b

/-- The values of a and b for the given parabola -/
theorem parabola_values (p : Parabola) : p.a = 5 ∧ p.b = 6 := by
  sorry

end parabola_values_l805_80594


namespace A_subset_B_l805_80550

def A : Set ℤ := {x | ∃ k : ℤ, x = 4 * k + 1}
def B : Set ℤ := {x | ∃ k : ℤ, x = 2 * k - 1}

theorem A_subset_B : A ⊆ B := by
  sorry

end A_subset_B_l805_80550


namespace corner_sum_is_164_l805_80525

/-- Represents a 9x9 grid filled with numbers 1 to 81 in row-wise order -/
def Grid := Fin 9 → Fin 9 → Fin 81

/-- The value at position (i, j) in the grid -/
def gridValue (i j : Fin 9) : Fin 81 :=
  ⟨i.val * 9 + j.val + 1, by sorry⟩

/-- The sum of the corner values in the grid -/
def cornerSum (g : Grid) : ℕ :=
  (g 0 0).val + (g 0 8).val + (g 8 0).val + (g 8 8).val

/-- Theorem stating that the sum of corner values in the defined grid is 164 -/
theorem corner_sum_is_164 :
  ∃ (g : Grid), cornerSum g = 164 :=
by sorry

end corner_sum_is_164_l805_80525


namespace circle_area_square_gt_ngon_areas_product_l805_80578

/-- Given a circle and two regular n-gons, one inscribed and one circumscribed,
    prove that the square of the circle's area is greater than the product of the n-gons' areas. -/
theorem circle_area_square_gt_ngon_areas_product (n : ℕ) (S S₁ S₂ : ℝ) 
    (h_n : n ≥ 3)
    (h_S : S > 0)
    (h_S₁ : S₁ > 0)
    (h_S₂ : S₂ > 0)
    (h_inscribed : S₁ = (n / 2) * S * Real.sin (2 * π / n))
    (h_circumscribed : S₂ = (n / 2) * S * Real.tan (π / n)) :
  S^2 > S₁ * S₂ := by
  sorry

end circle_area_square_gt_ngon_areas_product_l805_80578


namespace line_passes_through_fixed_point_l805_80585

/-- Given that k, -1, and b form an arithmetic sequence,
    prove that the line y = kx + b passes through the point (1, -2) -/
theorem line_passes_through_fixed_point (k b : ℝ) :
  (-1 = (k + b) / 2) →
  ∀ x y : ℝ, y = k * x + b → (x = 1 ∧ y = -2) :=
by sorry

end line_passes_through_fixed_point_l805_80585


namespace increasing_function_conditions_l805_80555

-- Define the piecewise function f
noncomputable def f (a b : ℝ) : ℝ → ℝ := fun x =>
  if x ≥ 0 then x^2 + 3 else a*x + b

-- State the theorem
theorem increasing_function_conditions (a b : ℝ) :
  (∀ x y : ℝ, x < y → f a b x < f a b y) →
  (a > 0 ∧ b ≤ 3) :=
by sorry

end increasing_function_conditions_l805_80555


namespace attendance_difference_l805_80538

/-- Calculates the total attendance for a week of football games given the conditions. -/
def totalAttendance (saturdayAttendance : ℕ) (expectedTotal : ℕ) : ℕ :=
  let mondayAttendance := saturdayAttendance - 20
  let wednesdayAttendance := mondayAttendance + 50
  let fridayAttendance := saturdayAttendance + mondayAttendance
  saturdayAttendance + mondayAttendance + wednesdayAttendance + fridayAttendance

/-- Theorem stating that the actual attendance exceeds the expected attendance by 40 people. -/
theorem attendance_difference (saturdayAttendance : ℕ) (expectedTotal : ℕ) 
  (h1 : saturdayAttendance = 80) 
  (h2 : expectedTotal = 350) : 
  totalAttendance saturdayAttendance expectedTotal - expectedTotal = 40 := by
  sorry

#eval totalAttendance 80 350 - 350

end attendance_difference_l805_80538


namespace f_composition_fixed_points_l805_80574

def f (x : ℝ) : ℝ := x^2 - 5*x

theorem f_composition_fixed_points :
  ∀ x : ℝ, f (f x) = f x ↔ x = 0 ∨ x = 5 ∨ x = 6 ∨ x = -1 := by
  sorry

end f_composition_fixed_points_l805_80574


namespace polynomial_factorization_l805_80549

theorem polynomial_factorization : 
  ∀ x : ℝ, x^2 - x + (1/4 : ℝ) = (x - 1/2)^2 := by
  sorry

end polynomial_factorization_l805_80549


namespace quadratic_polynomial_from_sum_and_product_l805_80534

theorem quadratic_polynomial_from_sum_and_product (s r : ℝ) :
  ∃ (a b : ℝ), a + b = s ∧ a * b = r^3 →
  ∀ x : ℝ, (x - a) * (x - b) = x^2 - s*x + r^3 :=
by sorry

end quadratic_polynomial_from_sum_and_product_l805_80534


namespace sale_price_is_63_percent_l805_80569

/-- The sale price of an item after two successive discounts -/
def sale_price (original_price : ℝ) : ℝ :=
  let first_discount := 0.1
  let second_discount := 0.3
  let price_after_first_discount := original_price * (1 - first_discount)
  price_after_first_discount * (1 - second_discount)

/-- Theorem stating that the sale price is 63% of the original price -/
theorem sale_price_is_63_percent (x : ℝ) : sale_price x = 0.63 * x := by
  sorry

end sale_price_is_63_percent_l805_80569


namespace jeff_running_schedule_l805_80508

/-- Jeff's running schedule problem -/
theorem jeff_running_schedule (x : ℕ) : 
  (3 * x + (x - 20) + (x + 10) = 290) → x = 60 := by
  sorry

end jeff_running_schedule_l805_80508


namespace extra_bottles_eq_three_l805_80532

/-- The number of juice bottles Paul drinks per day -/
def paul_bottles : ℕ := 3

/-- The number of juice bottles Donald drinks per day -/
def donald_bottles : ℕ := 9

/-- The difference between Donald's daily juice consumption and twice Paul's daily juice consumption -/
def extra_bottles : ℕ := donald_bottles - 2 * paul_bottles

theorem extra_bottles_eq_three : extra_bottles = 3 := by
  sorry

end extra_bottles_eq_three_l805_80532


namespace circular_pond_area_l805_80590

/-- Given a circular pond with a diameter of 20 feet and a line from the midpoint 
    of this diameter to the circumference of 18 feet, prove that the area of the 
    pond is 224π square feet. -/
theorem circular_pond_area (diameter : ℝ) (midpoint_to_circle : ℝ) : 
  diameter = 20 → midpoint_to_circle = 18 → 
  ∃ (radius : ℝ), radius^2 * π = 224 * π := by sorry

end circular_pond_area_l805_80590


namespace power_sum_sequence_l805_80544

/-- Given a sequence of sums of powers of a and b, prove that a^10 + b^10 = 123 -/
theorem power_sum_sequence (a b : ℝ) 
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11) :
  a^10 + b^10 = 123 := by
  sorry

end power_sum_sequence_l805_80544


namespace regular_soda_count_l805_80560

/-- The number of bottles of diet soda -/
def diet_soda : ℕ := 60

/-- The number of bottles of lite soda -/
def lite_soda : ℕ := 60

/-- The difference between regular and diet soda bottles -/
def regular_diet_difference : ℕ := 21

/-- The number of bottles of regular soda -/
def regular_soda : ℕ := diet_soda + regular_diet_difference

theorem regular_soda_count : regular_soda = 81 := by
  sorry

end regular_soda_count_l805_80560


namespace equal_money_in_five_weeks_l805_80516

/-- Represents the number of weeks it takes for two people to have the same amount of money -/
def weeks_to_equal_money (carol_initial : ℕ) (carol_weekly : ℕ) (mike_initial : ℕ) (mike_weekly : ℕ) : ℕ :=
  sorry

/-- Theorem stating that it takes 5 weeks for Carol and Mike to have the same amount of money -/
theorem equal_money_in_five_weeks :
  weeks_to_equal_money 60 9 90 3 = 5 := by
  sorry

end equal_money_in_five_weeks_l805_80516


namespace crazy_silly_school_books_l805_80577

theorem crazy_silly_school_books (movies : ℕ) (books : ℕ) 
  (h1 : movies = 14) 
  (h2 : books = movies + 1) : 
  books = 15 := by
  sorry

end crazy_silly_school_books_l805_80577


namespace total_fruits_is_78_l805_80565

-- Define the number of fruits for Louis
def louis_oranges : ℕ := 5
def louis_apples : ℕ := 3

-- Define the number of fruits for Samantha
def samantha_oranges : ℕ := 8
def samantha_apples : ℕ := 7

-- Define the number of fruits for Marley
def marley_oranges : ℕ := 2 * louis_oranges
def marley_apples : ℕ := 3 * samantha_apples

-- Define the number of fruits for Edward
def edward_oranges : ℕ := 3 * louis_oranges
def edward_apples : ℕ := 3 * louis_apples

-- Define the total number of fruits
def total_fruits : ℕ := 
  louis_oranges + louis_apples + 
  samantha_oranges + samantha_apples + 
  marley_oranges + marley_apples + 
  edward_oranges + edward_apples

-- Theorem statement
theorem total_fruits_is_78 : total_fruits = 78 := by
  sorry

end total_fruits_is_78_l805_80565


namespace min_value_theorem_l805_80561

def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (a + 8) * x + a^2 + a - 12

theorem min_value_theorem (a : ℝ) (h1 : a < 0) 
  (h2 : f a (a^2 - 4) = f a (2*a - 8)) :
  ∀ n : ℕ+, (f a n - 4*a) / (n + 1) ≥ 37/4 :=
sorry

end min_value_theorem_l805_80561


namespace prob_all_blue_is_one_twelfth_l805_80572

/-- The number of balls in the urn -/
def total_balls : ℕ := 10

/-- The number of blue balls in the urn -/
def blue_balls : ℕ := 5

/-- The number of balls drawn -/
def drawn_balls : ℕ := 3

/-- Combination function -/
def C (n k : ℕ) : ℚ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

/-- The probability of drawing all blue balls -/
def prob_all_blue : ℚ := C blue_balls drawn_balls / C total_balls drawn_balls

theorem prob_all_blue_is_one_twelfth : 
  prob_all_blue = 1 / 12 := by sorry

end prob_all_blue_is_one_twelfth_l805_80572


namespace linear_function_property_l805_80568

theorem linear_function_property (k b : ℝ) : 
  (3 = k + b) → (2 = -k + b) → k^2 - b^2 = -6 := by sorry

end linear_function_property_l805_80568


namespace smallest_q_property_l805_80500

theorem smallest_q_property : ∃ (q : ℕ), q > 0 ∧ q = 2015 ∧
  (∀ (m : ℕ), 1 ≤ m ∧ m ≤ 1006 →
    ∃ (n : ℤ), (m : ℚ) / 1007 * q < n ∧ n < (m + 1 : ℚ) / 1008 * q) ∧
  (∀ (q' : ℕ), 0 < q' ∧ q' < q →
    ¬(∀ (m : ℕ), 1 ≤ m ∧ m ≤ 1006 →
      ∃ (n : ℤ), (m : ℚ) / 1007 * q' < n ∧ n < (m + 1 : ℚ) / 1008 * q')) :=
by sorry

end smallest_q_property_l805_80500


namespace sequence_increasing_k_bound_l805_80530

theorem sequence_increasing_k_bound (k : ℝ) :
  (∀ n : ℕ+, (2 * n^2 + k * n) < (2 * (n + 1)^2 + k * (n + 1))) →
  k > -6 := by
  sorry

end sequence_increasing_k_bound_l805_80530


namespace plane_division_theorem_l805_80554

/-- Represents the number of regions formed by lines in a plane -/
def num_regions (h s : ℕ) : ℕ := h * (s + 1) + 1 + s * (s + 1) / 2

/-- Checks if a pair (h, s) satisfies the problem conditions -/
def is_valid_pair (h s : ℕ) : Prop :=
  h > 0 ∧ s > 0 ∧ num_regions h s = 1992

theorem plane_division_theorem :
  ∀ h s : ℕ, is_valid_pair h s ↔ (h = 995 ∧ s = 1) ∨ (h = 176 ∧ s = 10) ∨ (h = 80 ∧ s = 21) :=
by sorry

end plane_division_theorem_l805_80554


namespace pirate_treasure_chests_l805_80553

theorem pirate_treasure_chests : ∀ (gold silver bronze chests : ℕ),
  gold = 3500 →
  silver = 500 →
  bronze = 2 * silver →
  (gold + silver + bronze) / 1000 = chests →
  chests * 1000 = gold + silver + bronze →
  chests = 5 := by
  sorry

end pirate_treasure_chests_l805_80553


namespace limit_of_sequence_a_l805_80551

def a : ℕ → ℚ
  | 0 => 1
  | 1 => 2
  | (n + 2) => (4/7) * a (n + 1) + (3/7) * a n

theorem limit_of_sequence_a :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |a n - 17/10| < ε :=
sorry

end limit_of_sequence_a_l805_80551


namespace production_days_calculation_l805_80570

theorem production_days_calculation (n : ℕ) : 
  (∀ k : ℕ, k > 0 → (60 * n + 90) / (n + 1) = 65) → n = 5 :=
by
  sorry

end production_days_calculation_l805_80570


namespace shower_tiles_l805_80543

/-- Calculates the total number of tiles in a shower --/
def total_tiles (sides : ℕ) (width : ℕ) (height : ℕ) : ℕ :=
  sides * width * height

/-- Theorem: The total number of tiles in a 3-sided shower with 8 tiles in width and 20 tiles in height is 480 --/
theorem shower_tiles : total_tiles 3 8 20 = 480 := by
  sorry

end shower_tiles_l805_80543


namespace sales_tax_reduction_difference_l805_80579

/-- The difference in sales tax between two rates for a given market price -/
def sales_tax_difference (original_rate new_rate market_price : ℝ) : ℝ :=
  market_price * original_rate - market_price * new_rate

/-- Theorem stating the difference in sales tax for the given problem -/
theorem sales_tax_reduction_difference :
  let original_rate : ℝ := 3.5 / 100
  let new_rate : ℝ := 10 / 3 / 100
  let market_price : ℝ := 7800
  abs (sales_tax_difference original_rate new_rate market_price - 13.26) < 0.01 := by
sorry

end sales_tax_reduction_difference_l805_80579


namespace banana_groups_count_l805_80501

def total_bananas : ℕ := 290
def bananas_per_group : ℕ := 145

theorem banana_groups_count : total_bananas / bananas_per_group = 2 := by
  sorry

end banana_groups_count_l805_80501


namespace least_common_multiple_first_ten_l805_80512

theorem least_common_multiple_first_ten : ∃ n : ℕ, 
  (∀ k : ℕ, k ≤ 10 → k > 0 → n % k = 0) ∧ 
  (∀ m : ℕ, m < n → ∃ j : ℕ, j ≤ 10 ∧ j > 0 ∧ m % j ≠ 0) ∧
  n = 2520 := by
sorry

end least_common_multiple_first_ten_l805_80512


namespace b_initial_investment_l805_80524

/-- Represents the business scenario with two partners A and B -/
structure BusinessScenario where
  a_initial : ℕ  -- A's initial investment
  b_initial : ℕ  -- B's initial investment (unknown)
  a_withdraw : ℕ  -- Amount A withdraws after 8 months
  b_add : ℕ  -- Amount B adds after 8 months
  total_profit : ℕ  -- Total profit at the end of the year
  a_profit : ℕ  -- A's share of the profit

/-- Calculates the investment value for a partner -/
def investment_value (initial : ℕ) (change : ℕ) (is_withdraw : Bool) : ℕ :=
  if is_withdraw then
    8 * initial + 4 * (initial - change)
  else
    8 * initial + 4 * (initial + change)

/-- Theorem stating that B's initial investment was 4000 -/
theorem b_initial_investment
  (scenario : BusinessScenario)
  (h1 : scenario.a_initial = 6000)
  (h2 : scenario.a_withdraw = 1000)
  (h3 : scenario.b_add = 1000)
  (h4 : scenario.total_profit = 630)
  (h5 : scenario.a_profit = 357)
  : scenario.b_initial = 4000 := by
  sorry

end b_initial_investment_l805_80524


namespace pure_imaginary_complex_number_l805_80548

theorem pure_imaginary_complex_number (x : ℝ) :
  (((x^2 - 2*x - 3) : ℂ) + (x + 1)*I).re = 0 ∧ (((x^2 - 2*x - 3) : ℂ) + (x + 1)*I).im ≠ 0 → x = 3 := by
  sorry

end pure_imaginary_complex_number_l805_80548


namespace ellipse_intersection_l805_80573

-- Define the ellipse
def ellipse (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + (y-4)^2) + Real.sqrt ((x-6)^2 + y^2) = 10

-- Define the foci
def F1 : ℝ × ℝ := (0, 4)
def F2 : ℝ × ℝ := (6, 0)

-- Theorem statement
theorem ellipse_intersection :
  ∃ (x : ℝ), x ≠ 0 ∧ ellipse x 0 ∧ x = 7.5 := by
  sorry

end ellipse_intersection_l805_80573


namespace final_area_fraction_l805_80526

/-- The fraction of area remaining after one iteration -/
def remaining_fraction : ℚ := 8 / 9

/-- The number of iterations -/
def num_iterations : ℕ := 6

/-- The theorem stating the final fraction of area remaining -/
theorem final_area_fraction :
  remaining_fraction ^ num_iterations = 262144 / 531441 := by
  sorry

end final_area_fraction_l805_80526


namespace contradiction_assumption_l805_80513

theorem contradiction_assumption (x y z : ℝ) :
  (¬ (x > 0 ∨ y > 0 ∨ z > 0)) ↔ (x ≤ 0 ∧ y ≤ 0 ∧ z ≤ 0) := by
  sorry

end contradiction_assumption_l805_80513


namespace sum_of_bottom_circles_l805_80562

-- Define the type for circle positions
inductive Position
| Top | UpperLeft | UpperMiddle | UpperRight | Middle | LowerLeft | LowerMiddle | LowerRight

-- Define the function type for number placement
def Placement := Position → Nat

-- Define the conditions of the problem
def validPlacement (p : Placement) : Prop :=
  (∀ i : Position, p i ∈ Finset.range 9 \ {0}) ∧ 
  (∀ i j : Position, i ≠ j → p i ≠ p j) ∧
  p Position.Top * p Position.UpperLeft * p Position.UpperMiddle = 30 ∧
  p Position.Top * p Position.UpperMiddle * p Position.UpperRight = 40 ∧
  p Position.UpperLeft * p Position.Middle * p Position.LowerLeft = 28 ∧
  p Position.UpperRight * p Position.Middle * p Position.LowerRight = 35 ∧
  p Position.LowerLeft * p Position.LowerMiddle * p Position.LowerRight = 20

-- State the theorem
theorem sum_of_bottom_circles (p : Placement) (h : validPlacement p) : 
  p Position.LowerLeft + p Position.LowerMiddle + p Position.LowerRight = 17 := by
  sorry

end sum_of_bottom_circles_l805_80562


namespace consecutive_product_sum_l805_80559

theorem consecutive_product_sum (a b c : ℤ) : 
  b = a + 1 → c = b + 1 → a * b * c = 210 → a + b = 11 := by
  sorry

end consecutive_product_sum_l805_80559


namespace fraction_integer_values_l805_80503

theorem fraction_integer_values (a : ℤ) :
  (∃ k : ℤ, (a^3 + 1) / (a - 1) = k) ↔ a = -1 ∨ a = 0 ∨ a = 2 ∨ a = 3 := by
  sorry

end fraction_integer_values_l805_80503


namespace correct_pages_per_booklet_l805_80506

/-- The number of booklets in Jack's short story section -/
def num_booklets : ℕ := 49

/-- The total number of pages in all booklets -/
def total_pages : ℕ := 441

/-- The number of pages per booklet -/
def pages_per_booklet : ℕ := total_pages / num_booklets

theorem correct_pages_per_booklet : pages_per_booklet = 9 := by
  sorry

end correct_pages_per_booklet_l805_80506


namespace crayons_lost_l805_80540

theorem crayons_lost (initial : ℕ) (given_away : ℕ) (final : ℕ) 
  (h1 : initial = 440)
  (h2 : given_away = 111)
  (h3 : final = 223) :
  initial - given_away - final = 106 := by
  sorry

end crayons_lost_l805_80540


namespace simplify_sqrt_fraction_l805_80535

theorem simplify_sqrt_fraction (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (Real.sqrt (3 * a)) / (Real.sqrt (12 * a * b)) = (Real.sqrt b) / (2 * b) := by
  sorry

end simplify_sqrt_fraction_l805_80535


namespace even_monotone_inequality_l805_80541

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def monotone_increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x ≤ y → f x ≤ f y

theorem even_monotone_inequality (f : ℝ → ℝ) 
  (h_even : is_even_function f)
  (h_mono : monotone_increasing_on f (Set.Ici 0)) :
  f (-2) > f 1 := by
  sorry

end even_monotone_inequality_l805_80541


namespace base_10_to_base_8_l805_80509

theorem base_10_to_base_8 : 
  (3 * 8^3 + 4 * 8^2 + 1 * 8^1 + 1 * 8^0 : ℕ) = 1801 :=
by sorry

end base_10_to_base_8_l805_80509


namespace magic_8_ball_probability_l805_80556

/-- The probability of getting exactly k successes in n independent trials,
    where each trial has a success probability of p -/
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  Nat.choose n k * p^k * (1 - p)^(n - k)

/-- The probability of getting exactly 3 positive answers out of 8 questions
    with the Magic 8 Ball, where each question has a 2/5 chance of a positive answer -/
theorem magic_8_ball_probability : 
  binomial_probability 8 3 (2/5) = 108864/390625 := by
  sorry

end magic_8_ball_probability_l805_80556


namespace x_formula_l805_80529

def double_factorial : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => (n + 2) * double_factorial n

def x : ℕ → ℚ
  | 0 => 1
  | 1 => 1
  | (n + 2) => (x (n + 1))^2 / (x n + 2 * x (n + 1))

theorem x_formula (n : ℕ) : x n = 1 / (double_factorial (2 * n - 1)) := by
  sorry

end x_formula_l805_80529


namespace fahrenheit_to_celsius_l805_80539

theorem fahrenheit_to_celsius (C F : ℝ) : C = (4 / 7) * (F - 40) → C = 35 → F = 101.25 := by
  sorry

end fahrenheit_to_celsius_l805_80539


namespace four_digit_sum_l805_80580

theorem four_digit_sum (A B C D : Nat) : 
  A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D →
  A < 10 → B < 10 → C < 10 → D < 10 →
  (A + B + C) % 9 = 0 →
  (B + C + D) % 9 = 0 →
  A + B + C + D = 18 := by
sorry

end four_digit_sum_l805_80580


namespace problem_solution_l805_80564

theorem problem_solution : ∃ x : ℝ, (0.15 * 40 = 0.25 * x + 2) ∧ (x = 16) := by
  sorry

end problem_solution_l805_80564


namespace green_shirt_pairs_green_green_pairs_count_l805_80595

theorem green_shirt_pairs (total_students : ℕ) (red_students : ℕ) (green_students : ℕ) 
  (total_pairs : ℕ) (red_red_pairs : ℕ) : ℕ :=
  let green_green_pairs := 
    have _ : total_students = 144 := by sorry
    have _ : red_students = 63 := by sorry
    have _ : green_students = 81 := by sorry
    have _ : total_pairs = 72 := by sorry
    have _ : red_red_pairs = 27 := by sorry
    have _ : total_students = red_students + green_students := by sorry
    have _ : red_students * 2 ≥ red_red_pairs * 2 := by sorry
    let red_in_mixed_pairs := red_students - (red_red_pairs * 2)
    let remaining_green := green_students - red_in_mixed_pairs
    remaining_green / 2
  green_green_pairs

theorem green_green_pairs_count : 
  green_shirt_pairs 144 63 81 72 27 = 36 := by sorry

end green_shirt_pairs_green_green_pairs_count_l805_80595


namespace quiche_egg_volume_l805_80583

/-- Given the initial volume of raw spinach, the percentage it reduces to when cooked,
    the volume of cream cheese added, and the total volume of the quiche,
    calculate the volume of eggs used. -/
theorem quiche_egg_volume
  (raw_spinach : ℝ)
  (cooked_spinach_percentage : ℝ)
  (cream_cheese : ℝ)
  (total_quiche : ℝ)
  (h1 : raw_spinach = 40)
  (h2 : cooked_spinach_percentage = 0.20)
  (h3 : cream_cheese = 6)
  (h4 : total_quiche = 18) :
  total_quiche - (raw_spinach * cooked_spinach_percentage + cream_cheese) = 4 := by
  sorry

end quiche_egg_volume_l805_80583


namespace sugar_calculation_l805_80571

theorem sugar_calculation (recipe_sugar : ℕ) (additional_sugar : ℕ) 
  (h1 : recipe_sugar = 7)
  (h2 : additional_sugar = 3) :
  recipe_sugar - additional_sugar = 4 := by
  sorry

end sugar_calculation_l805_80571


namespace negative_sqrt_seven_greater_than_negative_sqrt_eleven_l805_80505

theorem negative_sqrt_seven_greater_than_negative_sqrt_eleven :
  -Real.sqrt 7 > -Real.sqrt 11 := by
  sorry

end negative_sqrt_seven_greater_than_negative_sqrt_eleven_l805_80505


namespace sin_shift_equivalence_l805_80507

open Real

theorem sin_shift_equivalence (x : ℝ) :
  sin (2 * (x + π / 6)) = sin (2 * x + π / 3) := by sorry

end sin_shift_equivalence_l805_80507


namespace quadratic_inequality_max_value_l805_80545

theorem quadratic_inequality_max_value (a b c : ℝ) :
  (∀ x : ℝ, a * x^2 + b * x + c ≥ 2 * a * x + b) →
  (∃ M : ℝ, M = 2/3 ∧ ∀ a b c : ℝ, b^2 / (3 * a^2 + c^2) ≤ M) :=
by sorry

end quadratic_inequality_max_value_l805_80545


namespace teacher_school_arrangements_l805_80563

theorem teacher_school_arrangements :
  let n : ℕ := 4  -- number of teachers and schools
  let arrangements := {f : Fin n → Fin n | Function.Surjective f}  -- surjective functions represent valid arrangements
  Fintype.card arrangements = 24 := by
sorry

end teacher_school_arrangements_l805_80563


namespace henrikhs_commute_l805_80576

def blocks_to_office (x : ℕ) : Prop :=
  let walking_time := 60 * x
  let bicycle_time := 20 * x
  let skateboard_time := 40 * x
  (walking_time = bicycle_time + 480) ∧ 
  (walking_time = skateboard_time + 240)

theorem henrikhs_commute : ∃ (x : ℕ), blocks_to_office x ∧ x = 12 := by
  sorry

end henrikhs_commute_l805_80576


namespace min_value_quadratic_sum_l805_80599

theorem min_value_quadratic_sum (x y z : ℝ) (h : x + y + z = 2) :
  ∃ (m : ℝ), m = 24 / 11 ∧ ∀ (a b c : ℝ), a + b + c = 2 → 2 * a^2 + 3 * b^2 + c^2 ≥ m :=
by sorry

end min_value_quadratic_sum_l805_80599


namespace inverse_proportion_problem_l805_80521

/-- Two real numbers are inversely proportional if their product is constant -/
def InverselyProportional (p q : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ p * q = k

theorem inverse_proportion_problem (p q : ℝ) :
  InverselyProportional p q →
  p + q = 40 →
  p - q = 10 →
  p = 7 →
  q = 375 / 7 := by
  sorry

end inverse_proportion_problem_l805_80521


namespace coefficient_x5y3_in_expansion_l805_80593

def binomial_expansion (a b : ℤ) (n : ℕ) : Polynomial ℤ := sorry

def coefficient_of_term (p : Polynomial ℤ) (x_power y_power : ℕ) : ℤ := sorry

theorem coefficient_x5y3_in_expansion :
  let p := binomial_expansion 2 (-3) 6
  coefficient_of_term (p - Polynomial.C (-1) * Polynomial.X ^ 6) 5 3 = 720 := by sorry

end coefficient_x5y3_in_expansion_l805_80593


namespace divisible_by_seven_l805_80597

theorem divisible_by_seven (k : ℕ) : 
  7 ∣ (2^(6*k+1) + 3^(6*k+1) + 5^(6*k+1)) := by
sorry

end divisible_by_seven_l805_80597


namespace sample_size_is_thirteen_l805_80502

/-- Represents a workshop with its production quantity -/
structure Workshop where
  quantity : ℕ

/-- Represents a stratified sampling scenario -/
structure StratifiedSampling where
  workshops : List Workshop
  sampleFromSmallest : ℕ

/-- Calculates the total sample size for a stratified sampling scenario -/
def totalSampleSize (s : StratifiedSampling) : ℕ :=
  sorry

/-- The main theorem stating that for the given scenario, the total sample size is 13 -/
theorem sample_size_is_thirteen :
  let scenario := StratifiedSampling.mk
    [Workshop.mk 120, Workshop.mk 80, Workshop.mk 60]
    3
  totalSampleSize scenario = 13 := by
  sorry

end sample_size_is_thirteen_l805_80502


namespace abs_frac_inequality_l805_80510

theorem abs_frac_inequality (x : ℝ) : 
  |((3 * x - 2) / (x - 2))| < 3 ↔ 4/3 < x ∧ x < 2 :=
sorry

end abs_frac_inequality_l805_80510


namespace milk_carton_delivery_l805_80581

theorem milk_carton_delivery (total_cartons : ℕ) (damaged_per_customer : ℕ) (total_accepted : ℕ) :
  total_cartons = 400 →
  damaged_per_customer = 60 →
  total_accepted = 160 →
  ∃ (num_customers : ℕ),
    num_customers > 0 ∧
    num_customers * (total_cartons / num_customers - damaged_per_customer) = total_accepted ∧
    num_customers = 4 :=
by sorry

end milk_carton_delivery_l805_80581


namespace no_integer_roots_l805_80514

/-- Polynomial P(x) = x^2019 + 2x^2018 + 3x^2017 + ... + 2019x + 2020 -/
def P (x : ℤ) : ℤ := 
  (Finset.range 2020).sum (fun i => (i + 1) * x^(2019 - i))

theorem no_integer_roots : ∀ x : ℤ, P x ≠ 0 := by
  sorry

end no_integer_roots_l805_80514


namespace basketball_lineup_combinations_l805_80547

/-- The number of players on the basketball team -/
def total_players : ℕ := 15

/-- The number of players in the starting lineup -/
def starting_lineup_size : ℕ := 6

/-- The number of predetermined players in the starting lineup -/
def predetermined_players : ℕ := 3

/-- The number of different possible starting lineups -/
def different_lineups : ℕ := 220

theorem basketball_lineup_combinations :
  Nat.choose (total_players - predetermined_players) (starting_lineup_size - predetermined_players) = different_lineups := by
  sorry

end basketball_lineup_combinations_l805_80547


namespace odd_function_extension_l805_80575

-- Define the function f
noncomputable def f : ℝ → ℝ := fun x =>
  if x < 0 then Real.exp (-x) + 2 * x - 1
  else -Real.exp x + 2 * x + 1

-- State the theorem
theorem odd_function_extension :
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x < 0, f x = Real.exp (-x) + 2 * x - 1) →
  (∀ x ≥ 0, f x = -Real.exp x + 2 * x + 1) := by
sorry

end odd_function_extension_l805_80575


namespace ball_count_theorem_l805_80586

/-- Represents the count of balls of each color in a jar. -/
structure BallCount where
  white : ℕ
  red : ℕ
  blue : ℕ

/-- Checks if the given ball count satisfies the 4:3:2 ratio. -/
def satisfiesRatio (bc : BallCount) : Prop :=
  3 * bc.white = 4 * bc.red ∧ 2 * bc.white = 4 * bc.blue

theorem ball_count_theorem (bc : BallCount) 
    (h_ratio : satisfiesRatio bc) (h_white : bc.white = 20) : 
    bc.red = 15 ∧ bc.blue = 10 := by
  sorry

#check ball_count_theorem

end ball_count_theorem_l805_80586


namespace inequality_proof_l805_80542

theorem inequality_proof (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) :
  (1/2) * (a + b)^2 + (1/4) * (a + b) ≥ a * Real.sqrt b + b * Real.sqrt a := by
  sorry

end inequality_proof_l805_80542


namespace sweet_potato_sharing_l805_80552

theorem sweet_potato_sharing (total : ℝ) (per_person : ℝ) (h1 : total = 52.5) (h2 : per_person = 5) :
  total - (⌊total / per_person⌋ * per_person) = 2.5 := by
  sorry

end sweet_potato_sharing_l805_80552


namespace b_95_mod_49_l805_80587

def b (n : ℕ) : ℕ := 5^n + 7^n

theorem b_95_mod_49 : b 95 ≡ 42 [ZMOD 49] := by sorry

end b_95_mod_49_l805_80587


namespace point_set_equivalence_l805_80517

theorem point_set_equivalence (x y : ℝ) : 
  y^2 - y = x^2 - x ↔ y = x ∨ y = 1 - x := by sorry

end point_set_equivalence_l805_80517


namespace remaining_money_is_24_l805_80528

/-- Given an initial amount of money, calculates the remaining amount after a series of transactions. -/
def remainingMoney (initialAmount : ℚ) : ℚ :=
  let afterIceCream := initialAmount - 5
  let afterTShirt := afterIceCream / 2
  let afterDeposit := afterTShirt * (4/5)
  afterDeposit

/-- Proves that given an initial amount of $65, the remaining money after transactions is $24. -/
theorem remaining_money_is_24 :
  remainingMoney 65 = 24 := by
  sorry

end remaining_money_is_24_l805_80528


namespace sally_picked_peaches_l805_80582

/-- Calculates the number of peaches Sally picked at the orchard. -/
def peaches_picked (initial_peaches final_peaches : ℕ) : ℕ :=
  final_peaches - initial_peaches

/-- Theorem stating that Sally picked 55 peaches at the orchard. -/
theorem sally_picked_peaches : peaches_picked 13 68 = 55 := by
  sorry

end sally_picked_peaches_l805_80582


namespace rugby_banquet_min_guests_l805_80520

/-- The minimum number of guests at a banquet given the total food consumed and maximum individual consumption --/
def min_guests (total_food : ℕ) (max_individual_consumption : ℕ) : ℕ :=
  (total_food + max_individual_consumption - 1) / max_individual_consumption

/-- Theorem stating the minimum number of guests at the rugby banquet --/
theorem rugby_banquet_min_guests :
  min_guests 4875 3 = 1625 := by
  sorry

end rugby_banquet_min_guests_l805_80520


namespace percentage_equality_l805_80537

theorem percentage_equality : ∃ x : ℝ, (x / 100) * 75 = (2.5 / 100) * 450 ∧ x = 15 := by
  sorry

end percentage_equality_l805_80537


namespace square_property_l805_80519

theorem square_property (a b c : ℤ) 
  (h : (a + 3)^2 + (b + 4)^2 - (c + 5)^2 = a^2 + b^2 - c^2) :
  ∃ k : ℤ, a^2 + b^2 - c^2 = k^2 := by
  sorry

end square_property_l805_80519


namespace golden_ratio_equivalences_l805_80511

open Real

theorem golden_ratio_equivalences :
  let φ : ℝ := 2 * sin (18 * π / 180)
  (sin (102 * π / 180) + Real.sqrt 3 * cos (102 * π / 180) = φ) ∧
  (sin (36 * π / 180) / sin (108 * π / 180) = φ) := by
  sorry

end golden_ratio_equivalences_l805_80511


namespace inverse_contrapositive_relation_l805_80557

theorem inverse_contrapositive_relation (p q r : Prop) :
  (¬p ↔ q) →  -- inverse of p is q
  ((¬p ↔ r) ↔ p) →  -- contrapositive of p is r
  (q ↔ ¬r) :=  -- q and r are negations of each other
by sorry

end inverse_contrapositive_relation_l805_80557


namespace last_toggled_locker_l805_80584

theorem last_toggled_locker (n : Nat) (h : n = 2048) :
  (Nat.sqrt n) ^ 2 = 1936 := by
  sorry

end last_toggled_locker_l805_80584


namespace right_triangle_rotation_forms_cone_l805_80589

/-- A right-angled triangle -/
structure RightTriangle where
  /-- One of the right-angled edges of the triangle -/
  edge : ℝ
  /-- The other right-angled edge of the triangle -/
  base : ℝ
  /-- The hypotenuse of the triangle -/
  hypotenuse : ℝ
  /-- Condition for a right-angled triangle -/
  right_angle : edge^2 + base^2 = hypotenuse^2

/-- A solid formed by rotating a plane figure -/
inductive RotatedSolid
  | Cone
  | Cylinder
  | Sphere

/-- Function to determine the solid formed by rotating a right-angled triangle -/
def solidFormedByRotation (triangle : RightTriangle) (rotationAxis : ℝ) : RotatedSolid :=
  sorry

/-- Theorem stating that rotating a right-angled triangle about one of its right-angled edges forms a cone -/
theorem right_triangle_rotation_forms_cone (triangle : RightTriangle) :
  solidFormedByRotation triangle triangle.edge = RotatedSolid.Cone := by
  sorry

end right_triangle_rotation_forms_cone_l805_80589


namespace perpendicular_vectors_x_value_l805_80588

/-- Two vectors in R² are perpendicular if their dot product is zero -/
def isPerpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

theorem perpendicular_vectors_x_value :
  ∀ x : ℝ,
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (x, -2)
  isPerpendicular a b → x = 4 := by
sorry

end perpendicular_vectors_x_value_l805_80588


namespace quadratic_vertex_l805_80566

/-- Represents a quadratic function of the form y = ax^2 + bx - 3 -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ

/-- Checks if a point (x, y) lies on the quadratic function -/
def QuadraticFunction.contains (f : QuadraticFunction) (x y : ℝ) : Prop :=
  y = f.a * x^2 + f.b * x - 3

theorem quadratic_vertex (f : QuadraticFunction) :
  f.contains (-2) 5 →
  f.contains (-1) 0 →
  f.contains 0 (-3) →
  f.contains 1 (-4) →
  f.contains 2 (-3) →
  ∃ (a b : ℝ), f = ⟨a, b⟩ ∧ (1, -4) = (- b / (2 * a), - (b^2 - 4*a*3) / (4 * a)) :=
by sorry

end quadratic_vertex_l805_80566
