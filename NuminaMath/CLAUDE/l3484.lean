import Mathlib

namespace negation_of_proposition_negation_of_inequality_l3484_348432

theorem negation_of_proposition (P : ℝ → Prop) :
  (∀ x : ℝ, P x) ↔ ¬(∃ x : ℝ, ¬(P x)) :=
by sorry

theorem negation_of_inequality :
  ¬(∀ x : ℝ, x^2 + 2 > 2*x) ↔ (∃ x : ℝ, x^2 + 2 ≤ 2*x) :=
by sorry

end negation_of_proposition_negation_of_inequality_l3484_348432


namespace red_beads_count_l3484_348470

theorem red_beads_count (green : ℕ) (brown : ℕ) (taken : ℕ) (left : ℕ) : 
  green = 1 → brown = 2 → taken = 2 → left = 4 → 
  ∃ (red : ℕ), red = (green + brown + taken + left) - (green + brown) :=
by sorry

end red_beads_count_l3484_348470


namespace laborer_average_salary_l3484_348443

/-- Calculates the average monthly salary of laborers in a factory --/
theorem laborer_average_salary
  (total_workers : ℕ)
  (total_average_salary : ℚ)
  (num_supervisors : ℕ)
  (supervisor_average_salary : ℚ)
  (num_laborers : ℕ)
  (h_total_workers : total_workers = num_supervisors + num_laborers)
  (h_num_supervisors : num_supervisors = 6)
  (h_num_laborers : num_laborers = 42)
  (h_total_average_salary : total_average_salary = 1250)
  (h_supervisor_average_salary : supervisor_average_salary = 2450) :
  let laborer_total_salary := total_workers * total_average_salary - num_supervisors * supervisor_average_salary
  (laborer_total_salary / num_laborers) = 1078.57 := by
sorry

#eval (48 * 1250 - 6 * 2450) / 42

end laborer_average_salary_l3484_348443


namespace sum_differences_theorem_l3484_348490

def numeral1 : ℕ := 987348621829
def numeral2 : ℕ := 74693251

def local_value (digit : ℕ) (position : ℕ) : ℕ := digit * (10 ^ position)

def face_value (digit : ℕ) : ℕ := digit

def difference_local_face (digit : ℕ) (position : ℕ) : ℕ :=
  local_value digit position - face_value digit

theorem sum_differences_theorem : 
  let first_8_pos := 8
  let second_8_pos := 1
  let seven_pos := 7
  (difference_local_face 8 first_8_pos + difference_local_face 8 second_8_pos) * 
  difference_local_face 7 seven_pos = 55999994048000192 := by
sorry

end sum_differences_theorem_l3484_348490


namespace unique_solution_l3484_348402

theorem unique_solution : ∃! (x : ℝ), x > 0 ∧ 5^29 * x^15 = 2 * 10^29 := by
  sorry

end unique_solution_l3484_348402


namespace existence_of_tangent_circle_l3484_348404

/-- Given three circles with radii 1, 3, and 4 touching each other and the sides of a rectangle,
    there exists a circle touching all three circles and one side of the rectangle. -/
theorem existence_of_tangent_circle (r₁ r₂ r₃ : ℝ) 
  (h₁ : r₁ = 1) (h₂ : r₂ = 3) (h₃ : r₃ = 4) : 
  ∃ x : ℝ, 
    (x + r₁)^2 - (x - r₁)^2 = (r₂ + x)^2 - (r₂ + r₁ - x)^2 ∧
    ∃ y : ℝ, 
      (y + r₂)^2 - (r₂ + r₁ - y)^2 = (r₃ + y)^2 - (r₃ - y)^2 ∧
      x = y := by
  sorry

end existence_of_tangent_circle_l3484_348404


namespace system_solution_l3484_348471

theorem system_solution (a b c x y z : ℝ) : 
  (a * y + b * x = c ∧ c * x + a * z = b ∧ b * z + c * y = a) ↔
  ((a * b * c ≠ 0 ∧ 
    x = (b^2 + c^2 - a^2) / (2*b*c) ∧ 
    y = (a^2 + c^2 - b^2) / (2*a*c) ∧ 
    z = (a^2 + b^2 - c^2) / (2*a*b)) ∨
   (a = 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ 
    ((x = 1 ∧ y = z) ∨ (x = 1 ∧ y = -z))) ∨
   (b = 0 ∧ a ≠ 0 ∧ c ≠ 0 ∧ 
    ((y = 1 ∧ x = z) ∨ (y = 1 ∧ x = -z))) ∨
   (c = 0 ∧ a ≠ 0 ∧ b ≠ 0 ∧ 
    ((z = 1 ∧ x = y) ∨ (z = 1 ∧ x = -y))) ∨
   (a = 0 ∧ b = 0 ∧ c = 0)) :=
by sorry


end system_solution_l3484_348471


namespace waste_bread_price_is_correct_l3484_348464

/-- Calculates the price per pound of wasted bread products given the following conditions:
  * Minimum wage is $8/hour
  * 20 pounds of meat wasted at $5/pound
  * 15 pounds of fruits and vegetables wasted at $4/pound
  * 60 pounds of bread products wasted (price unknown)
  * 10 hours of time-and-a-half pay for janitorial staff (normal pay $10/hour)
  * Total work hours to pay for everything is 50 hours
-/
def wasteBreadPrice (
  minWage : ℚ)
  (meatWeight : ℚ)
  (meatPrice : ℚ)
  (fruitVegWeight : ℚ)
  (fruitVegPrice : ℚ)
  (breadWeight : ℚ)
  (janitorHours : ℚ)
  (janitorWage : ℚ)
  (totalWorkHours : ℚ) : ℚ :=
  let meatCost := meatWeight * meatPrice
  let fruitVegCost := fruitVegWeight * fruitVegPrice
  let janitorCost := janitorHours * (janitorWage * 1.5)
  let totalEarnings := totalWorkHours * minWage
  let breadCost := totalEarnings - (meatCost + fruitVegCost + janitorCost)
  breadCost / breadWeight

theorem waste_bread_price_is_correct :
  wasteBreadPrice 8 20 5 15 4 60 10 10 50 = 1.5 := by
  sorry

end waste_bread_price_is_correct_l3484_348464


namespace units_digit_of_sum_l3484_348493

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The result of raising a number to a power, considering only the units digit -/
def powerMod10 (base : ℕ) (exp : ℕ) : ℕ :=
  (base ^ exp) % 10

theorem units_digit_of_sum : unitsDigit ((33 : ℕ)^43 + (43 : ℕ)^32) = 8 := by
  sorry

end units_digit_of_sum_l3484_348493


namespace min_value_expression_l3484_348407

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ((2*a + 2*a*b - b*(b + 1))^2 + (b - 4*a^2 + 2*a*(b + 1))^2) / (4*a^2 + b^2) ≥ 1 ∧
  ((2*1 + 2*1*1 - 1*(1 + 1))^2 + (1 - 4*1^2 + 2*1*(1 + 1))^2) / (4*1^2 + 1^2) = 1 :=
sorry

end min_value_expression_l3484_348407


namespace NH4I_molecular_weight_l3484_348463

/-- The molecular weight of NH4I in grams per mole -/
def molecular_weight_NH4I : ℝ := 145

/-- The number of moles used in the given condition -/
def given_moles : ℝ := 9

/-- The total weight in grams for the given number of moles -/
def given_total_weight : ℝ := 1305

theorem NH4I_molecular_weight :
  molecular_weight_NH4I = given_total_weight / given_moles :=
by sorry

end NH4I_molecular_weight_l3484_348463


namespace cone_min_lateral_area_l3484_348446

/-- For a cone with volume π/6, when its lateral area is minimum, 
    the tangent of the angle between the slant height and the base is √2 -/
theorem cone_min_lateral_area (r h : ℝ) : 
  r > 0 → h > 0 → 
  (1/3) * π * r^2 * h = π/6 →
  (∀ r' h', r' > 0 → h' > 0 → (1/3) * π * r'^2 * h' = π/6 → 
    π * r * (r^2 + h^2).sqrt ≤ π * r' * (r'^2 + h'^2).sqrt) →
  h / r = Real.sqrt 2 := by
sorry

end cone_min_lateral_area_l3484_348446


namespace books_bound_calculation_remaining_paper_condition_l3484_348411

/-- Represents the number of books bound in a bookbinding workshop. -/
def books_bound (initial_white : ℕ) (initial_colored : ℕ) : ℕ :=
  initial_white - (initial_colored - initial_white)

/-- Theorem stating the number of books bound given the initial quantities and conditions. -/
theorem books_bound_calculation :
  let initial_white := 92
  let initial_colored := 135
  books_bound initial_white initial_colored = 178 :=
by
  sorry

/-- Theorem verifying the remaining paper condition after binding. -/
theorem remaining_paper_condition (initial_white initial_colored : ℕ) :
  let bound := books_bound initial_white initial_colored
  initial_white - bound = (initial_colored - bound) / 2 :=
by
  sorry

end books_bound_calculation_remaining_paper_condition_l3484_348411


namespace total_weight_of_CaO_l3484_348458

/-- The atomic weight of calcium in g/mol -/
def atomic_weight_Ca : ℝ := 40.08

/-- The atomic weight of oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The number of moles of CaO -/
def moles_CaO : ℝ := 7

/-- The molecular weight of CaO in g/mol -/
def molecular_weight_CaO : ℝ := atomic_weight_Ca + atomic_weight_O

/-- The total weight of CaO in grams -/
def total_weight_CaO : ℝ := molecular_weight_CaO * moles_CaO

/-- Theorem stating the total weight of 7 moles of CaO -/
theorem total_weight_of_CaO : total_weight_CaO = 392.56 := by
  sorry

end total_weight_of_CaO_l3484_348458


namespace divisors_of_180_l3484_348436

/-- The number of positive divisors of 180 is 18. -/
theorem divisors_of_180 : Finset.card (Nat.divisors 180) = 18 := by
  sorry

end divisors_of_180_l3484_348436


namespace wizard_elixir_combinations_l3484_348441

/-- The number of magical herbs available. -/
def num_herbs : ℕ := 4

/-- The number of mystical crystals available. -/
def num_crystals : ℕ := 6

/-- The number of crystals incompatible with one specific herb. -/
def num_incompatible : ℕ := 3

/-- The number of viable combinations for the wizard's elixir. -/
def viable_combinations : ℕ := num_herbs * num_crystals - num_incompatible

theorem wizard_elixir_combinations :
  viable_combinations = 21 :=
sorry

end wizard_elixir_combinations_l3484_348441


namespace music_purchase_total_spent_l3484_348496

/-- Represents the purchase of music albums -/
structure MusicPurchase where
  country_albums : ℕ
  pop_albums : ℕ
  country_price : ℕ
  pop_price : ℕ
  songs_per_album : ℕ
  discount_threshold : ℕ
  discount_amount : ℕ

/-- Calculates the total cost before discounts -/
def total_cost_before_discounts (purchase : MusicPurchase) : ℕ :=
  purchase.country_albums * purchase.country_price + purchase.pop_albums * purchase.pop_price

/-- Calculates the number of discounts -/
def number_of_discounts (purchase : MusicPurchase) : ℕ :=
  (purchase.country_albums + purchase.pop_albums) / purchase.discount_threshold

/-- Calculates the total amount spent after discounts -/
def total_amount_spent (purchase : MusicPurchase) : ℕ :=
  total_cost_before_discounts purchase - number_of_discounts purchase * purchase.discount_amount

/-- Theorem: The total amount spent on music albums after applying discounts is $108 -/
theorem music_purchase_total_spent (purchase : MusicPurchase) 
  (h1 : purchase.country_albums = 4)
  (h2 : purchase.pop_albums = 5)
  (h3 : purchase.country_price = 12)
  (h4 : purchase.pop_price = 15)
  (h5 : purchase.songs_per_album = 8)
  (h6 : purchase.discount_threshold = 3)
  (h7 : purchase.discount_amount = 5) :
  total_amount_spent purchase = 108 := by
  sorry

#eval total_amount_spent {
  country_albums := 4,
  pop_albums := 5,
  country_price := 12,
  pop_price := 15,
  songs_per_album := 8,
  discount_threshold := 3,
  discount_amount := 5
}

end music_purchase_total_spent_l3484_348496


namespace karen_group_size_l3484_348431

/-- Proves that if Zack tutors students in groups of 14, and both Zack and Karen tutor
    the same total number of 70 students, then Karen must also tutor students in groups of 14. -/
theorem karen_group_size (zack_group_size : ℕ) (total_students : ℕ) (karen_group_size : ℕ) :
  zack_group_size = 14 →
  total_students = 70 →
  total_students % zack_group_size = 0 →
  total_students % karen_group_size = 0 →
  total_students / zack_group_size = total_students / karen_group_size →
  karen_group_size = 14 := by
sorry

end karen_group_size_l3484_348431


namespace units_digit_factorial_sum_20_l3484_348413

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def units_digit (n : ℕ) : ℕ := n % 10

def factorial_sum (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem units_digit_factorial_sum_20 :
  units_digit (factorial_sum 20) = units_digit (factorial 1 + factorial 2 + factorial 3 + factorial 4) :=
by sorry

end units_digit_factorial_sum_20_l3484_348413


namespace digits_of_2_pow_15_times_5_pow_10_l3484_348477

/-- The number of digits in 2^15 * 5^10 is 12 -/
theorem digits_of_2_pow_15_times_5_pow_10 : 
  (Nat.digits 10 (2^15 * 5^10)).length = 12 := by sorry

end digits_of_2_pow_15_times_5_pow_10_l3484_348477


namespace davids_chemistry_marks_l3484_348452

/-- Given David's marks in various subjects and his average, prove his Chemistry marks -/
theorem davids_chemistry_marks
  (english_marks : ℕ)
  (math_marks : ℕ)
  (physics_marks : ℕ)
  (biology_marks : ℕ)
  (average_marks : ℚ)
  (h1 : english_marks = 96)
  (h2 : math_marks = 95)
  (h3 : physics_marks = 82)
  (h4 : biology_marks = 95)
  (h5 : average_marks = 93)
  (h6 : (english_marks + math_marks + physics_marks + biology_marks + chemistry_marks : ℚ) / 5 = average_marks) :
  chemistry_marks = 97 :=
by
  sorry

end davids_chemistry_marks_l3484_348452


namespace rectangle_width_and_ratio_l3484_348485

-- Define the rectangle
structure Rectangle where
  initial_length : ℝ
  new_length : ℝ
  new_perimeter : ℝ

-- Define the theorem
theorem rectangle_width_and_ratio 
  (rect : Rectangle) 
  (h1 : rect.initial_length = 8) 
  (h2 : rect.new_length = 12) 
  (h3 : rect.new_perimeter = 36) : 
  ∃ (new_width : ℝ), 
    new_width = 6 ∧ 
    new_width / rect.new_length = 1 / 2 := by
  sorry


end rectangle_width_and_ratio_l3484_348485


namespace remaining_distance_l3484_348472

/-- Calculates the remaining distance in a bike course -/
theorem remaining_distance (total_course : ℝ) (before_break : ℝ) (after_break : ℝ) :
  total_course = 10.5 ∧ before_break = 1.5 ∧ after_break = 3.73 →
  (total_course - (before_break + after_break)) * 1000 = 5270 := by
  sorry

#check remaining_distance

end remaining_distance_l3484_348472


namespace custom_operation_result_l3484_348447

/-- Custom operation ⊕ -/
def oplus (x y : ℝ) : ℝ := x + 2*y + 3

theorem custom_operation_result :
  ∀ (a b : ℝ), 
  (oplus (oplus (a^3) (a^2)) a = oplus (a^3) (oplus (a^2) a)) ∧
  (oplus (oplus (a^3) (a^2)) a = b) →
  a + b = 21/8 := by
sorry

end custom_operation_result_l3484_348447


namespace f_derivative_at_one_l3484_348438

noncomputable def f (x : ℝ) : ℝ := (1 - 2 * x^3)^10

theorem f_derivative_at_one : 
  deriv f 1 = 60 := by sorry

end f_derivative_at_one_l3484_348438


namespace sin_cos_225_degrees_l3484_348453

theorem sin_cos_225_degrees :
  Real.sin (225 * π / 180) = -Real.sqrt 2 / 2 ∧
  Real.cos (225 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end sin_cos_225_degrees_l3484_348453


namespace commission_problem_l3484_348457

/-- Calculates the total sales amount given the commission rate and commission amount -/
def calculateTotalSales (commissionRate : ℚ) (commissionAmount : ℚ) : ℚ :=
  commissionAmount / (commissionRate / 100)

/-- Theorem: Given a commission rate of 5% and a commission amount of 12.50, the total sales amount is 250 -/
theorem commission_problem :
  let commissionRate : ℚ := 5
  let commissionAmount : ℚ := 12.50
  calculateTotalSales commissionRate commissionAmount = 250 := by
  sorry

end commission_problem_l3484_348457


namespace magic_square_solution_l3484_348416

/-- Represents a 3x3 magic square with some known entries -/
structure MagicSquare where
  x : ℤ
  sum : ℤ

/-- The magic square property: all rows, columns, and diagonals sum to the same value -/
def magic_square_property (m : MagicSquare) : Prop :=
  ∃ (d e f g h : ℤ),
    m.x + 21 + 50 = m.sum ∧
    m.x + 3 + f = m.sum ∧
    50 + e + h = m.sum ∧
    m.x + d + h = m.sum ∧
    3 + d + e = m.sum ∧
    f + g + h = m.sum

/-- The theorem stating that x must be 106 in the given magic square -/
theorem magic_square_solution (m : MagicSquare) 
  (h : magic_square_property m) : m.x = 106 := by
  sorry

#check magic_square_solution

end magic_square_solution_l3484_348416


namespace quadratic_function_constraint_l3484_348442

/-- Given a quadratic function f(x) = ax^2 + bx + c where a ≠ 0,
    if f(-1) = 0 and x ≤ f(x) ≤ (1/2)(x^2 + 1) for all x ∈ ℝ,
    then a = 1/4 -/
theorem quadratic_function_constraint (a b c : ℝ) (ha : a ≠ 0) :
  let f := fun (x : ℝ) => a * x^2 + b * x + c
  (f (-1) = 0) →
  (∀ x : ℝ, x ≤ f x ∧ f x ≤ (1/2) * (x^2 + 1)) →
  a = 1/4 := by
  sorry

end quadratic_function_constraint_l3484_348442


namespace cubic_function_c_range_l3484_348494

theorem cubic_function_c_range (a b c : ℝ) :
  let f := fun x => x^3 + a*x^2 + b*x + c
  (0 < f (-1) ∧ f (-1) = f (-2) ∧ f (-2) = f (-3) ∧ f (-3) ≤ 3) →
  (6 < c ∧ c ≤ 9) := by
sorry

end cubic_function_c_range_l3484_348494


namespace weight_problem_l3484_348449

/-- Proves that the initial number of students is 19 given the conditions of the weight problem. -/
theorem weight_problem (initial_avg : ℝ) (new_avg : ℝ) (new_student_weight : ℝ)
  (h1 : initial_avg = 15)
  (h2 : new_avg = 14.6)
  (h3 : new_student_weight = 7) :
  ∃ n : ℕ, n = 19 ∧ 
    (n : ℝ) * initial_avg + new_student_weight = (n + 1 : ℝ) * new_avg :=
by sorry

end weight_problem_l3484_348449


namespace exam_full_marks_l3484_348429

theorem exam_full_marks (A B C D F : ℝ) 
  (hA : A = 0.9 * B)
  (hB : B = 1.25 * C)
  (hC : C = 0.8 * D)
  (hAmarks : A = 360)
  (hDpercent : D = 0.8 * F) : 
  F = 500 := by
sorry

end exam_full_marks_l3484_348429


namespace train_speed_calculation_l3484_348479

/-- Calculates the speed of a train given the parameters of a passing goods train --/
theorem train_speed_calculation (goods_train_speed : ℝ) (goods_train_length : ℝ) (passing_time : ℝ) :
  goods_train_speed = 50.4 →
  goods_train_length = 240 →
  passing_time = 10 →
  ∃ (man_train_speed : ℝ), man_train_speed = 36 := by
  sorry

end train_speed_calculation_l3484_348479


namespace davids_math_marks_l3484_348424

/-- Given David's marks in various subjects and his average, prove his Mathematics marks --/
theorem davids_math_marks
  (english : ℕ)
  (physics : ℕ)
  (chemistry : ℕ)
  (biology : ℕ)
  (average : ℚ)
  (num_subjects : ℕ)
  (h1 : english = 96)
  (h2 : physics = 82)
  (h3 : chemistry = 87)
  (h4 : biology = 92)
  (h5 : average = 90.4)
  (h6 : num_subjects = 5)
  : ∃ (math : ℕ), math = 95 ∧ 
    (english + math + physics + chemistry + biology) / num_subjects = average :=
by sorry

end davids_math_marks_l3484_348424


namespace distance_is_90km_l3484_348473

/-- Calculates the distance traveled downstream by a boat -/
def distance_downstream (boat_speed : ℝ) (stream_speed : ℝ) (time : ℝ) : ℝ :=
  (boat_speed + stream_speed) * time

/-- Theorem: The distance traveled downstream is 90 km -/
theorem distance_is_90km (boat_speed : ℝ) (stream_speed : ℝ) (time : ℝ)
  (h1 : boat_speed = 25)
  (h2 : stream_speed = 5)
  (h3 : time = 3) :
  distance_downstream boat_speed stream_speed time = 90 := by
  sorry

#eval distance_downstream 25 5 3

end distance_is_90km_l3484_348473


namespace solve_exponential_equation_l3484_348481

theorem solve_exponential_equation : 
  ∃ x : ℝ, (16 : ℝ)^x * (16 : ℝ)^x * (16 : ℝ)^(2*x) = (64 : ℝ)^6 ∧ x = 3 := by
  sorry

end solve_exponential_equation_l3484_348481


namespace gwen_book_count_l3484_348428

/-- The number of books on each shelf. -/
def books_per_shelf : ℕ := 4

/-- The number of shelves containing mystery books. -/
def mystery_shelves : ℕ := 5

/-- The number of shelves containing picture books. -/
def picture_shelves : ℕ := 3

/-- The total number of books Gwen has. -/
def total_books : ℕ := books_per_shelf * (mystery_shelves + picture_shelves)

theorem gwen_book_count : total_books = 32 := by
  sorry

end gwen_book_count_l3484_348428


namespace incorrect_correct_sum_l3484_348421

theorem incorrect_correct_sum : ∃ x : ℤ, 
  (x - 5 + 14 = 39) ∧ (39 + (5 * x + 14) = 203) := by
  sorry

end incorrect_correct_sum_l3484_348421


namespace population_growth_determinants_l3484_348450

-- Define the factors that can potentially influence population growth
structure PopulationFactors where
  birthRate : ℝ
  deathRate : ℝ
  totalPopulation : ℝ
  socialProductionRate : ℝ
  naturalGrowthRate : ℝ

-- Define population growth pattern as a function of factors
def populationGrowthPattern (factors : PopulationFactors) : ℝ := sorry

-- Theorem stating that population growth pattern is determined by birth rate, death rate, and natural growth rate
theorem population_growth_determinants (factors : PopulationFactors) :
  populationGrowthPattern factors =
    populationGrowthPattern ⟨factors.birthRate, factors.deathRate, 0, 0, factors.naturalGrowthRate⟩ :=
by sorry

end population_growth_determinants_l3484_348450


namespace complex_root_quadratic_equation_l3484_348467

theorem complex_root_quadratic_equation (a b : ℝ) :
  (∃ (x : ℂ), x = 1 + Complex.I * Real.sqrt 3 ∧ a * x^2 + b * x + 1 = 0) →
  a = (1 : ℝ) / 4 ∧ b = -(1 : ℝ) / 2 := by
  sorry

end complex_root_quadratic_equation_l3484_348467


namespace candy_total_l3484_348483

def candy_problem (tabitha stan : ℕ) : Prop :=
  ∃ (julie carlos veronica benjamin : ℕ),
    tabitha = 22 ∧
    stan = 16 ∧
    julie = tabitha / 2 ∧
    carlos = 2 * stan ∧
    veronica = julie + stan ∧
    benjamin = (tabitha + carlos) / 2 + 9 ∧
    tabitha + stan + julie + carlos + veronica + benjamin = 144

theorem candy_total : candy_problem 22 16 := by
  sorry

end candy_total_l3484_348483


namespace squats_on_fourth_day_l3484_348469

/-- Calculates the number of squats on a given day, given the initial number of squats and daily increase. -/
def squats_on_day (initial_squats : ℕ) (daily_increase : ℕ) (day : ℕ) : ℕ :=
  initial_squats + (day - 1) * daily_increase

/-- Theorem stating that on the fourth day, the number of squats will be 45, given the initial conditions. -/
theorem squats_on_fourth_day :
  squats_on_day 30 5 4 = 45 := by
  sorry

end squats_on_fourth_day_l3484_348469


namespace five_items_three_bags_l3484_348466

/-- The number of ways to distribute n distinct items into k identical bags --/
def distributionWays (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 5 distinct items into 3 identical bags --/
theorem five_items_three_bags : distributionWays 5 3 = 36 := by sorry

end five_items_three_bags_l3484_348466


namespace euler_minus_i_pi_is_real_l3484_348488

theorem euler_minus_i_pi_is_real : Complex.im (Complex.exp (-Complex.I * Real.pi)) = 0 := by
  sorry

end euler_minus_i_pi_is_real_l3484_348488


namespace largest_intersection_is_one_l3484_348445

/-- The polynomial function f(x) = x^5 - 5x^4 + 10x^3 - 10x^2 + 5x - b -/
def f (b : ℝ) (x : ℝ) : ℝ := x^5 - 5*x^4 + 10*x^3 - 10*x^2 + 5*x - b

/-- The linear function g(x) = cx - d -/
def g (c d : ℝ) (x : ℝ) : ℝ := c*x - d

/-- The difference between f and g -/
def h (b c d : ℝ) (x : ℝ) : ℝ := f b x - g c d x

theorem largest_intersection_is_one (b c d : ℝ) :
  (∃ p q r : ℝ, p < q ∧ q < r ∧ 
    (∀ x : ℝ, h b c d x = 0 ↔ x = p ∨ x = q ∨ x = r)) →
  r = 1 :=
sorry

end largest_intersection_is_one_l3484_348445


namespace quadrilateral_offset_l3484_348433

-- Define the quadrilateral properties
def diagonal : ℝ := 30
def offset1 : ℝ := 6
def area : ℝ := 225

-- Theorem to prove
theorem quadrilateral_offset (offset2 : ℝ) :
  area = (diagonal * (offset1 + offset2)) / 2 → offset2 = 9 := by
  sorry

end quadrilateral_offset_l3484_348433


namespace august_eighth_is_saturday_l3484_348414

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a month with specific properties -/
structure Month where
  days : Nat
  mondays : Nat
  tuesdays : Nat

/-- Function to determine the day of the week for a given date -/
def dayOfWeek (m : Month) (date : Nat) : DayOfWeek := sorry

theorem august_eighth_is_saturday 
  (m : Month) 
  (h1 : m.days = 31) 
  (h2 : m.mondays = 5) 
  (h3 : m.tuesdays = 4) : 
  dayOfWeek m 8 = DayOfWeek.Saturday := by sorry

end august_eighth_is_saturday_l3484_348414


namespace xy_value_l3484_348486

theorem xy_value (x y : ℝ) (h1 : x + y = 10) (h2 : x^3 + y^3 = 370) : x * y = 21 := by
  sorry

end xy_value_l3484_348486


namespace max_k_for_even_quadratic_min_one_l3484_348400

/-- A quadratic function f(x) = x^2 + mx + n -/
def f (m n x : ℝ) : ℝ := x^2 + m*x + n

/-- The absolute value function h(x) = |f(x)| -/
def h (m n x : ℝ) : ℝ := |f m n x|

/-- Theorem: Maximum value of k for even quadratic function with minimum 1 -/
theorem max_k_for_even_quadratic_min_one :
  ∃ (k : ℝ), k = 1/2 ∧
  ∀ (m n : ℝ),
    (∀ x, f m n (-x) = f m n x) →  -- f is even
    (∀ x, f m n x ≥ 1) →           -- minimum of f is 1
    (∃ M, M ≥ k ∧
      ∀ x, x ∈ Set.Icc (-1 : ℝ) 1 → h m n x ≤ M) →  -- max of h in [-1,1] is M ≥ k
    k ≤ 1/2 :=
by sorry

end max_k_for_even_quadratic_min_one_l3484_348400


namespace stratified_sampling_proof_l3484_348461

/-- Represents the total population -/
def total_population : ℕ := 27 + 54 + 81

/-- Represents the number of elderly people in the population -/
def elderly_population : ℕ := 27

/-- Represents the number of elderly people in the sample -/
def elderly_sample : ℕ := 3

/-- Represents the total sample size -/
def sample_size : ℕ := 18

/-- Proves that the given sample size is correct for the stratified sampling -/
theorem stratified_sampling_proof :
  (elderly_sample : ℚ) / elderly_population = sample_size / total_population :=
by sorry

end stratified_sampling_proof_l3484_348461


namespace students_walking_home_l3484_348498

theorem students_walking_home (total : ℚ) (bus : ℚ) (auto : ℚ) (bike : ℚ) (walk : ℚ) : 
  bus = 1/3 * total → auto = 1/5 * total → bike = 1/15 * total → 
  walk = total - (bus + auto + bike) →
  walk = 2/5 * total :=
sorry

end students_walking_home_l3484_348498


namespace quadratic_roots_l3484_348455

theorem quadratic_roots (a b : ℝ) (h1 : a * b ≠ 0) 
  (h2 : a^2 + 2*b*a + a = 0) (h3 : b^2 + 2*b*b + a = 0) : 
  a = -3 ∧ b = 1 := by
sorry

end quadratic_roots_l3484_348455


namespace group_size_is_21_l3484_348495

/-- Represents the Pinterest group --/
structure PinterestGroup where
  /-- Number of people in the group --/
  people : ℕ
  /-- Average number of pins contributed per person per day --/
  pinsPerDay : ℕ
  /-- Number of pins deleted per person per week --/
  pinsDeletedPerWeek : ℕ
  /-- Initial number of pins --/
  initialPins : ℕ
  /-- Number of pins after 4 weeks --/
  pinsAfterMonth : ℕ

/-- Calculates the number of people in the Pinterest group --/
def calculateGroupSize (group : PinterestGroup) : ℕ :=
  let netPinsPerWeek := group.pinsPerDay * 7 - group.pinsDeletedPerWeek
  let pinsAddedPerPerson := netPinsPerWeek * 4
  let totalPinsAdded := group.pinsAfterMonth - group.initialPins
  totalPinsAdded / pinsAddedPerPerson

/-- Theorem stating that the number of people in the group is 21 --/
theorem group_size_is_21 (group : PinterestGroup) 
  (h1 : group.pinsPerDay = 10)
  (h2 : group.pinsDeletedPerWeek = 5)
  (h3 : group.initialPins = 1000)
  (h4 : group.pinsAfterMonth = 6600) :
  calculateGroupSize group = 21 := by
  sorry

#eval calculateGroupSize { 
  people := 0,  -- This value doesn't matter for the calculation
  pinsPerDay := 10, 
  pinsDeletedPerWeek := 5, 
  initialPins := 1000, 
  pinsAfterMonth := 6600 
}

end group_size_is_21_l3484_348495


namespace polynomial_functional_equation_l3484_348408

noncomputable def P : (ℝ → ℝ → ℝ) → Prop :=
  λ p => ∀ x y : ℝ, p (x^2) (y^2) = p ((x + y)^2 / 2) ((x - y)^2 / 2)

theorem polynomial_functional_equation :
  ∀ p : ℝ → ℝ → ℝ, P p ↔ ∃ q : ℝ → ℝ → ℝ, ∀ x y : ℝ, p x y = q (x + y) (x * y * (x - y)^2) :=
by sorry

end polynomial_functional_equation_l3484_348408


namespace estimate_fish_population_verify_fish_estimate_l3484_348465

/-- Estimates the total number of fish in a pond using the mark-recapture method. -/
theorem estimate_fish_population (initial_catch : ℕ) (recapture : ℕ) (marked_recaught : ℕ) : ℕ :=
  let estimated_population := initial_catch * recapture / marked_recaught
  -- Proof that estimated_population = 750 given the conditions
  sorry

/-- Verifies the estimated fish population for the given problem. -/
theorem verify_fish_estimate : estimate_fish_population 30 50 2 = 750 := by
  -- Proof that the estimate is correct for the given values
  sorry

end estimate_fish_population_verify_fish_estimate_l3484_348465


namespace gcd_8251_6105_l3484_348492

theorem gcd_8251_6105 : Nat.gcd 8251 6105 = 37 := by
  sorry

end gcd_8251_6105_l3484_348492


namespace problem_1_problem_2_l3484_348448

theorem problem_1 : 3 * Real.sqrt 3 - Real.sqrt 8 + Real.sqrt 2 - Real.sqrt 27 = -Real.sqrt 2 := by sorry

theorem problem_2 : (4 * Real.sqrt 6 - 6 * Real.sqrt 3) / (2 * Real.sqrt 3) = 2 * Real.sqrt 2 - 3 := by sorry

end problem_1_problem_2_l3484_348448


namespace unique_square_divisible_by_five_in_range_l3484_348480

theorem unique_square_divisible_by_five_in_range : 
  ∃! x : ℕ, x^2 = x ∧ x % 5 = 0 ∧ 50 < x^2 ∧ x^2 < 120 :=
by sorry

end unique_square_divisible_by_five_in_range_l3484_348480


namespace worker_time_relationship_l3484_348482

/-- Given a batch of parts and a production rate, this theorem establishes
    the relationship between the number of workers and the time needed to complete the task. -/
theorem worker_time_relationship 
  (total_parts : ℕ) 
  (production_rate : ℕ) 
  (h1 : total_parts = 200)
  (h2 : production_rate = 10) :
  ∀ x y : ℝ, x > 0 → (y = (total_parts : ℝ) / (production_rate * x)) ↔ y = 20 / x :=
by sorry

end worker_time_relationship_l3484_348482


namespace units_digit_of_1505_odd_squares_sum_l3484_348499

/-- The units digit of the sum of the squares of the first n odd, positive integers -/
def unitsDigitOfOddSquaresSum (n : ℕ) : ℕ :=
  (n / 5 * 5) % 10

theorem units_digit_of_1505_odd_squares_sum :
  unitsDigitOfOddSquaresSum 1505 = 5 := by
  sorry

end units_digit_of_1505_odd_squares_sum_l3484_348499


namespace maintenance_check_increase_l3484_348426

theorem maintenance_check_increase (original_time : ℝ) (increase_percent : ℝ) (new_time : ℝ) :
  original_time = 20 →
  increase_percent = 25 →
  new_time = original_time * (1 + increase_percent / 100) →
  new_time = 25 :=
by sorry

end maintenance_check_increase_l3484_348426


namespace quadratic_roots_problem_l3484_348491

theorem quadratic_roots_problem (a b m p q : ℝ) : 
  (a^2 - m*a + 5 = 0) →
  (b^2 - m*b + 5 = 0) →
  ((a + 1/b)^2 - p*(a + 1/b) + q = 0) →
  ((b + 1/a)^2 - p*(b + 1/a) + q = 0) →
  q = 36/5 := by
sorry

end quadratic_roots_problem_l3484_348491


namespace ferris_wheel_small_seat_capacity_l3484_348422

/-- The number of small seats on the Ferris wheel -/
def small_seats : ℕ := 2

/-- The number of people each small seat can hold -/
def small_seat_capacity : ℕ := 14

/-- The total number of people who can ride on small seats -/
def total_small_seat_riders : ℕ := small_seats * small_seat_capacity

theorem ferris_wheel_small_seat_capacity : total_small_seat_riders = 28 := by
  sorry

end ferris_wheel_small_seat_capacity_l3484_348422


namespace equation_one_solutions_l3484_348409

theorem equation_one_solutions : 
  ∀ x : ℝ, (x - 3)^2 = 4 ↔ x = 5 ∨ x = 1 := by sorry

end equation_one_solutions_l3484_348409


namespace two_solutions_for_second_trace_l3484_348434

/-- Represents a trace of a plane -/
structure Trace where
  -- Add necessary fields

/-- Represents an inclination angle -/
structure InclinationAngle where
  -- Add necessary fields

/-- Represents a plane -/
structure Plane where
  firstTrace : Trace
  firstInclinationAngle : InclinationAngle
  axisPointOutside : Bool

/-- Represents a solution for the second trace -/
structure SecondTraceSolution where
  -- Add necessary fields

/-- 
Given a plane's first trace, first inclination angle, and the condition that the axis point 
is outside the drawing frame, there exist exactly two possible solutions for the second trace.
-/
theorem two_solutions_for_second_trace (p : Plane) : 
  p.axisPointOutside → ∃! (s : Finset SecondTraceSolution), s.card = 2 := by
  sorry

end two_solutions_for_second_trace_l3484_348434


namespace divisibility_implies_difference_one_l3484_348427

theorem divisibility_implies_difference_one
  (a b c d : ℕ)
  (h1 : (a * b - c * d) ∣ a)
  (h2 : (a * b - c * d) ∣ b)
  (h3 : (a * b - c * d) ∣ c)
  (h4 : (a * b - c * d) ∣ d) :
  a * b - c * d = 1 := by
sorry

end divisibility_implies_difference_one_l3484_348427


namespace y_intercept_of_line_l3484_348454

/-- The y-intercept of the line x - 2y = 5 is -5/2 -/
theorem y_intercept_of_line (x y : ℝ) : x - 2*y = 5 → y = -5/2 := by
  sorry

end y_intercept_of_line_l3484_348454


namespace arithmetic_sequence_equality_l3484_348430

theorem arithmetic_sequence_equality (n : ℕ) (a b : Fin n → ℕ) :
  n ≥ 2018 →
  (∀ i : Fin n, a i ≤ 5 * n ∧ b i ≤ 5 * n) →
  (∀ i j : Fin n, i ≠ j → a i ≠ a j ∧ b i ≠ b j) →
  (∃ d : ℚ, ∀ i j : Fin n, (a i : ℚ) / (b i : ℚ) - (a j : ℚ) / (b j : ℚ) = d * (i - j)) →
  ∀ i j : Fin n, (a i : ℚ) / (b i : ℚ) = (a j : ℚ) / (b j : ℚ) :=
by sorry

end arithmetic_sequence_equality_l3484_348430


namespace simplify_expression_l3484_348401

theorem simplify_expression (x : ℝ) : 2*x - 3*(2+x) + 4*(2-x) - 5*(2+3*x) = -20*x - 8 := by
  sorry

end simplify_expression_l3484_348401


namespace feuerbach_centers_parallelogram_or_collinear_l3484_348410

/-- A point in the plane -/
structure Point := (x y : ℝ)

/-- A quadrilateral defined by four points -/
structure Quadrilateral := (A B C D : Point)

/-- The intersection point of the diagonals of a quadrilateral -/
def diagonalIntersection (q : Quadrilateral) : Point := sorry

/-- The center of the Feuerbach circle of a triangle -/
def feuerbachCenter (A B C : Point) : Point := sorry

/-- Predicate to check if four points form a parallelogram -/
def isParallelogram (P Q R S : Point) : Prop := sorry

/-- Predicate to check if four points are collinear -/
def areCollinear (P Q R S : Point) : Prop := sorry

/-- Main theorem -/
theorem feuerbach_centers_parallelogram_or_collinear (q : Quadrilateral) :
  let E := diagonalIntersection q
  let F1 := feuerbachCenter q.A q.B E
  let F2 := feuerbachCenter q.B q.C E
  let F3 := feuerbachCenter q.C q.D E
  let F4 := feuerbachCenter q.D q.A E
  isParallelogram F1 F2 F3 F4 ∨ areCollinear F1 F2 F3 F4 := by
  sorry

end feuerbach_centers_parallelogram_or_collinear_l3484_348410


namespace david_score_l3484_348487

/-- Calculates the score of a player in a Scrabble game given the opponent's initial lead,
    the opponent's play, and the opponent's final lead. -/
def calculate_score (initial_lead : ℕ) (opponent_play : ℕ) (final_lead : ℕ) : ℕ :=
  initial_lead + opponent_play - final_lead

/-- Theorem stating that David's score in the Scrabble game is 32 points. -/
theorem david_score :
  calculate_score 22 15 5 = 32 := by
  sorry

end david_score_l3484_348487


namespace ratio_of_numbers_with_sum_gcd_equal_lcm_l3484_348419

theorem ratio_of_numbers_with_sum_gcd_equal_lcm (A B : ℕ) (h1 : A ≥ B) :
  A + B + Nat.gcd A B = Nat.lcm A B → (A : ℚ) / B = 3 / 2 := by
  sorry

end ratio_of_numbers_with_sum_gcd_equal_lcm_l3484_348419


namespace line_parallel_to_intersection_of_parallel_planes_l3484_348459

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between lines and planes
variable (parallelLinePlane : Line → Plane → Prop)

-- Define the parallel relation between lines
variable (parallelLine : Line → Line → Prop)

-- Define the intersection of two planes
variable (intersection : Plane → Plane → Line)

-- Theorem statement
theorem line_parallel_to_intersection_of_parallel_planes
  (a b : Line) (α β : Plane)
  (h1 : parallelLinePlane a α)
  (h2 : parallelLinePlane a β)
  (h3 : intersection α β = b) :
  parallelLine a b :=
sorry

end line_parallel_to_intersection_of_parallel_planes_l3484_348459


namespace john_apartment_number_l3484_348437

/-- Represents a skyscraper with 10 apartments on each floor. -/
structure Skyscraper where
  /-- John's apartment number -/
  john_apartment : ℕ
  /-- Mary's apartment number -/
  mary_apartment : ℕ
  /-- John's floor number -/
  john_floor : ℕ

/-- 
Given a skyscraper with 10 apartments on each floor, 
if John's floor number is equal to Mary's apartment number 
and the sum of their apartment numbers is 239, 
then John lives in apartment 217.
-/
theorem john_apartment_number (s : Skyscraper) : 
  s.john_floor = s.mary_apartment → 
  s.john_apartment + s.mary_apartment = 239 → 
  s.john_apartment = 217 := by
sorry

end john_apartment_number_l3484_348437


namespace factorial_ratio_l3484_348474

theorem factorial_ratio : (11 : ℕ).factorial / ((7 : ℕ).factorial * (4 : ℕ).factorial) = 330 := by
  sorry

end factorial_ratio_l3484_348474


namespace semicircle_area_ratio_l3484_348497

theorem semicircle_area_ratio :
  let AB : ℝ := 10
  let AC : ℝ := 6
  let CB : ℝ := 4
  let large_semicircle_area : ℝ := (1/2) * Real.pi * (AB/2)^2
  let small_semicircle1_area : ℝ := (1/2) * Real.pi * (AC/2)^2
  let small_semicircle2_area : ℝ := (1/2) * Real.pi * (CB/2)^2
  let shaded_area : ℝ := large_semicircle_area - small_semicircle1_area - small_semicircle2_area
  let circle_area : ℝ := Real.pi * (CB/2)^2
  (shaded_area / circle_area) = (3/2) := by
sorry

end semicircle_area_ratio_l3484_348497


namespace train_speed_problem_l3484_348489

/-- Proves that given two trains of specified lengths running in opposite directions,
    where one train has a known speed and the time to cross each other is known,
    the speed of the other train can be determined. -/
theorem train_speed_problem (length1 length2 known_speed crossing_time : ℝ) :
  length1 = 140 ∧
  length2 = 190 ∧
  known_speed = 40 ∧
  crossing_time = 11.879049676025918 →
  ∃ other_speed : ℝ,
    other_speed = 60 ∧
    (length1 + length2) / crossing_time * 3.6 = known_speed + other_speed :=
by sorry

end train_speed_problem_l3484_348489


namespace gcf_60_90_l3484_348420

theorem gcf_60_90 : Nat.gcd 60 90 = 30 := by
  sorry

end gcf_60_90_l3484_348420


namespace dima_guarantee_win_or_draw_l3484_348435

/-- Represents a player in the game -/
inductive Player : Type
| Gosha : Player
| Dima : Player

/-- Represents a cell on the board -/
structure Cell :=
(row : Nat)
(col : Nat)

/-- Represents the game board -/
def Board := List Cell

/-- Represents a game state -/
structure GameState :=
(board : Board)
(currentPlayer : Player)

/-- Checks if a sequence of 7 consecutive cells is occupied -/
def isWinningSequence (sequence : List Cell) (board : Board) : Bool :=
  sorry

/-- Checks if the game is in a winning state for the current player -/
def isWinningState (state : GameState) : Bool :=
  sorry

/-- Represents a game strategy -/
def Strategy := GameState → Cell

/-- Theorem: Dima can guarantee a win or draw -/
theorem dima_guarantee_win_or_draw :
  ∃ (strategy : Strategy),
    ∀ (game : GameState),
      game.currentPlayer = Player.Dima →
      (∃ (future_game : GameState), 
        isWinningState future_game ∧ future_game.currentPlayer = Player.Dima) ∨
      (∀ (future_game : GameState), ¬isWinningState future_game) :=
sorry

end dima_guarantee_win_or_draw_l3484_348435


namespace culture_medium_composition_l3484_348406

/-- Represents the composition of a culture medium --/
structure CultureMedium where
  salineWater : ℝ
  nutrientBroth : ℝ
  pureWater : ℝ

/-- The initial mixture ratio --/
def initialMixture : CultureMedium := {
  salineWater := 0.1
  nutrientBroth := 0.05
  pureWater := 0
}

/-- The required total volume of the culture medium in liters --/
def totalVolume : ℝ := 1

/-- The required percentage of pure water in the final mixture --/
def pureWaterPercentage : ℝ := 0.3

theorem culture_medium_composition :
  ∃ (final : CultureMedium),
    final.salineWater + final.nutrientBroth + final.pureWater = totalVolume ∧
    final.nutrientBroth / (final.salineWater + final.nutrientBroth) = initialMixture.nutrientBroth / (initialMixture.salineWater + initialMixture.nutrientBroth) ∧
    final.pureWater = totalVolume * pureWaterPercentage ∧
    final.nutrientBroth = 1/3 ∧
    final.pureWater = 0.3 := by
  sorry

end culture_medium_composition_l3484_348406


namespace other_radius_length_l3484_348468

/-- A circle is defined by its center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- All radii of a circle have the same length -/
axiom circle_radii_equal (c : Circle) (p q : ℝ × ℝ) :
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2 →
  (q.1 - c.center.1)^2 + (q.2 - c.center.2)^2 = c.radius^2 →
  ((p.1 - c.center.1)^2 + (p.2 - c.center.2)^2).sqrt =
  ((q.1 - c.center.1)^2 + (q.2 - c.center.2)^2).sqrt

theorem other_radius_length (c : Circle) (p q : ℝ × ℝ) 
    (hp : (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2)
    (hq : (q.1 - c.center.1)^2 + (q.2 - c.center.2)^2 = c.radius^2)
    (h_radius : ((p.1 - c.center.1)^2 + (p.2 - c.center.2)^2).sqrt = 2) :
    ((q.1 - c.center.1)^2 + (q.2 - c.center.2)^2).sqrt = 2 := by
  sorry

end other_radius_length_l3484_348468


namespace problem_statement_l3484_348462

theorem problem_statement (x : ℝ) (y : ℝ) (h_y_pos : y > 0) : 
  let A : Set ℝ := {x^2 + x + 1, -x, -x - 1}
  let B : Set ℝ := {-y, -y/2, y + 1}
  A = B → x^2 + y^2 = 5 := by
sorry

end problem_statement_l3484_348462


namespace rejected_products_percentage_l3484_348415

theorem rejected_products_percentage
  (john_reject_rate : ℝ)
  (jane_reject_rate : ℝ)
  (jane_inspect_fraction : ℝ)
  (h1 : john_reject_rate = 0.007)
  (h2 : jane_reject_rate = 0.008)
  (h3 : jane_inspect_fraction = 0.5)
  : (john_reject_rate * (1 - jane_inspect_fraction) + jane_reject_rate * jane_inspect_fraction) * 100 = 0.75 := by
  sorry

end rejected_products_percentage_l3484_348415


namespace intersection_of_lines_l3484_348423

/-- Given four points in 3D space, this theorem states that the intersection
    of the lines formed by these points is at a specific coordinate. -/
theorem intersection_of_lines (A B C D : ℝ × ℝ × ℝ) :
  A = (5, -6, 8) →
  B = (15, -16, 13) →
  C = (1, 4, -5) →
  D = (3, -4, 11) →
  ∃ t s : ℝ,
    (5 + 10*t, -6 - 10*t, 8 + 5*t) = (1 + 2*s, 4 - 8*s, -5 + 16*s) ∧
    (5 + 10*t, -6 - 10*t, 8 + 5*t) = (-10/3, 14/3, -1/3) :=
by sorry

end intersection_of_lines_l3484_348423


namespace banana_arrangements_l3484_348412

def word := "BANANA"

def letter_count : Nat := word.length

def b_count : Nat := (word.toList.filter (· == 'B')).length
def n_count : Nat := (word.toList.filter (· == 'N')).length
def a_count : Nat := (word.toList.filter (· == 'A')).length

def distinct_arrangements : Nat := letter_count.factorial / (b_count.factorial * n_count.factorial * a_count.factorial)

theorem banana_arrangements :
  letter_count = 6 ∧ b_count = 1 ∧ n_count = 2 ∧ a_count = 3 →
  distinct_arrangements = 60 := by
  sorry

end banana_arrangements_l3484_348412


namespace expenditure_estimate_l3484_348460

/-- The regression line equation for a company's expenditure (y) based on revenue (x) -/
def regression_line (x : ℝ) (a : ℝ) : ℝ := 0.8 * x + a

/-- Theorem: Given the regression line equation, when revenue is 7 billion yuan, 
    the estimated expenditure is 4.4 billion yuan -/
theorem expenditure_estimate (a : ℝ) : 
  ∃ (y : ℝ), regression_line 7 a = y ∧ y = 4.4 := by
  sorry

end expenditure_estimate_l3484_348460


namespace orange_harvest_per_day_l3484_348439

theorem orange_harvest_per_day (total_sacks : ℕ) (total_days : ℕ) (sacks_per_day : ℕ) :
  total_sacks = 24 →
  total_days = 3 →
  sacks_per_day = total_sacks / total_days →
  sacks_per_day = 8 := by
sorry

end orange_harvest_per_day_l3484_348439


namespace max_value_expression_l3484_348444

theorem max_value_expression (a b c x : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  2 * (a - x) * (x + Real.sqrt (x^2 + b^2 + c)) ≤ a^2 + b^2 + c :=
by sorry

end max_value_expression_l3484_348444


namespace function_from_derivative_and_point_l3484_348456

/-- Given a function f: ℝ → ℝ with f'(x) = 4x³ for all x and f(1) = -1, 
    prove that f(x) = x⁴ - 2 for all x ∈ ℝ -/
theorem function_from_derivative_and_point (f : ℝ → ℝ) 
    (h1 : ∀ x, deriv f x = 4 * x^3)
    (h2 : f 1 = -1) :
    ∀ x, f x = x^4 - 2 := by
  sorry

end function_from_derivative_and_point_l3484_348456


namespace number_decrease_theorem_l3484_348405

theorem number_decrease_theorem :
  (∃ (N k a x : ℕ), k ≥ 1 ∧ 1 ≤ a ∧ a ≤ 9 ∧ x < 10^k ∧ N = 10^k * a + x ∧ N = 57 * x) ∧
  (¬ ∃ (N k a x : ℕ), k ≥ 1 ∧ 1 ≤ a ∧ a ≤ 9 ∧ x < 10^k ∧ N = 10^k * a + x ∧ N = 58 * x) :=
by sorry

end number_decrease_theorem_l3484_348405


namespace f_properties_l3484_348403

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 / x + a * Real.log x

theorem f_properties (a : ℝ) (h_a : a > 0) :
  (∃ (x_min : ℝ), x_min > 0 ∧ x_min = 1/2 ∧ 
    (∀ (x : ℝ), x > 0 → f a x ≥ f a x_min)) ∧
  (¬∃ (x_max : ℝ), x_max > 0 ∧ 
    (∀ (x : ℝ), x > 0 → f a x ≤ f a x_max)) ∧
  ((∃ (x : ℝ), x > 0 ∧ f a x < 2) ↔ (a > 0 ∧ a ≠ 2)) :=
by sorry

end f_properties_l3484_348403


namespace no_identical_lines_l3484_348425

theorem no_identical_lines : ¬∃ (d k : ℝ), ∀ (x y : ℝ),
  (4 * x + d * y + k = 0 ↔ k * x - 3 * y + 18 = 0) :=
sorry

end no_identical_lines_l3484_348425


namespace cricket_time_calculation_l3484_348418

/-- The total time Sean and Indira played cricket together -/
def total_cricket_time (sean_daily_time : ℕ) (sean_days : ℕ) (indira_time : ℕ) : ℕ :=
  sean_daily_time * sean_days + indira_time

/-- Theorem stating the total time Sean and Indira played cricket -/
theorem cricket_time_calculation :
  total_cricket_time 50 14 812 = 1512 := by
  sorry

end cricket_time_calculation_l3484_348418


namespace betty_cookie_consumption_l3484_348478

/-- The number of cookies Betty eats per day -/
def cookies_per_day : ℕ := 7

/-- The number of brownies Betty eats per day -/
def brownies_per_day : ℕ := 1

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The difference between cookies and brownies after a week -/
def cookie_brownie_difference : ℕ := 36

theorem betty_cookie_consumption :
  cookies_per_day * days_in_week - brownies_per_day * days_in_week = cookie_brownie_difference :=
by sorry

end betty_cookie_consumption_l3484_348478


namespace price_reduction_equation_l3484_348484

/-- Proves the correct equation for a price reduction scenario -/
theorem price_reduction_equation (x : ℝ) : 
  (∃ (original_price final_price : ℝ),
    original_price = 200 ∧ 
    final_price = 162 ∧ 
    final_price = original_price * (1 - x)^2) ↔ 
  200 * (1 - x)^2 = 162 := by
sorry

end price_reduction_equation_l3484_348484


namespace z_in_fourth_quadrant_l3484_348417

def determinant (a b c d : ℂ) : ℂ := a * d - b * c

theorem z_in_fourth_quadrant (z : ℂ) 
  (h : determinant z 1 Complex.I Complex.I = 2 + Complex.I) : 
  0 < z.re ∧ z.im < 0 := by sorry

end z_in_fourth_quadrant_l3484_348417


namespace work_day_ends_at_target_time_l3484_348440

-- Define the start time, lunch time, and total work hours
def start_time : Nat := 8 * 60  -- 8:00 AM in minutes
def lunch_time : Nat := 13 * 60  -- 1:00 PM in minutes
def total_work_minutes : Nat := 9 * 60  -- 9 hours in minutes
def lunch_break_minutes : Nat := 30

-- Define the end time we want to prove
def target_end_time : Nat := 17 * 60 + 30  -- 5:30 PM in minutes

-- Theorem to prove
theorem work_day_ends_at_target_time :
  start_time + total_work_minutes + lunch_break_minutes = target_end_time := by
  sorry


end work_day_ends_at_target_time_l3484_348440


namespace cubic_equation_roots_l3484_348451

-- Define the polynomial
def f (x : ℂ) : ℂ := x^3 - x^2 - 1

-- State the theorem
theorem cubic_equation_roots :
  ∃ (a b c : ℂ), 
    (a + b + c = 1) ∧ 
    (a * b + a * c + b * c = 0) ∧ 
    (a * b * c = -1) ∧ 
    (f a = 0) ∧ (f b = 0) ∧ (f c = 0) := by
  sorry

end cubic_equation_roots_l3484_348451


namespace function_equivalence_l3484_348476

-- Define the function f
def f : ℝ → ℝ := fun x ↦ x^2 + 2*x + 2

-- State the theorem
theorem function_equivalence :
  (∀ x ≥ 0, f (Real.sqrt x - 1) = x + 1) →
  (∀ x ≥ -1, f x = x^2 + 2*x + 2) :=
by
  sorry

end function_equivalence_l3484_348476


namespace fifth_polygon_exterior_angles_sum_l3484_348475

/-- Represents a polygon in the sequence -/
structure Polygon where
  sides : ℕ

/-- Generates the next polygon in the sequence -/
def nextPolygon (p : Polygon) : Polygon :=
  { sides := p.sides + 2 }

/-- The sequence of polygons -/
def polygonSequence : ℕ → Polygon
  | 0 => { sides := 4 }  -- Square
  | n + 1 => nextPolygon (polygonSequence n)

/-- Sum of exterior angles of a polygon -/
def sumExteriorAngles (p : Polygon) : ℝ := 360

theorem fifth_polygon_exterior_angles_sum :
  sumExteriorAngles (polygonSequence 4) = 360 := by
  sorry

end fifth_polygon_exterior_angles_sum_l3484_348475
