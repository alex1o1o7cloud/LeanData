import Mathlib

namespace arithmetic_sequence_ratio_l2344_234429

theorem arithmetic_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- arithmetic sequence definition
  q > 1 →  -- common ratio condition
  a 1 + a 4 = 9 →  -- first condition
  a 2 * a 3 = 8 →  -- second condition
  (a 2015 + a 2016) / (a 2013 + a 2014) = 4 := by
  sorry

end arithmetic_sequence_ratio_l2344_234429


namespace fish_tagging_theorem_l2344_234463

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

end fish_tagging_theorem_l2344_234463


namespace quadratic_with_irrational_root_l2344_234457

theorem quadratic_with_irrational_root :
  ∃ (a b c : ℚ), a ≠ 0 ∧
  (∀ x : ℝ, a * x^2 + b * x + c = 0 ↔ x = Real.sqrt 5 - 3 ∨ x = -Real.sqrt 5 - 3) ∧
  a = 1 ∧ b = 6 ∧ c = -4 := by
sorry

end quadratic_with_irrational_root_l2344_234457


namespace farmer_price_l2344_234419

def potato_problem (x : ℝ) : Prop :=
  let andrey_revenue := 60 * (2 * x)
  let boris_revenue := 15 * (1.6 * x) + 45 * (2.24 * x)
  boris_revenue - andrey_revenue = 1200

theorem farmer_price : ∃ x : ℝ, potato_problem x ∧ x = 250 := by
  sorry

end farmer_price_l2344_234419


namespace exists_periodic_functions_with_nonperiodic_difference_l2344_234421

/-- A function is periodic if it takes at least two different values and there exists a positive period. -/
def IsPeriodic (f : ℝ → ℝ) : Prop :=
  (∃ x y, f x ≠ f y) ∧ (∃ p > 0, ∀ x, f (x + p) = f x)

/-- The period of a function. -/
def Period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  p > 0 ∧ ∀ x, f (x + p) = f x

/-- Theorem stating that there exist periodic functions g and h with periods 6 and 2π respectively,
    such that their difference is not periodic. -/
theorem exists_periodic_functions_with_nonperiodic_difference :
  ∃ (g h : ℝ → ℝ),
    IsPeriodic g ∧ Period g 6 ∧
    IsPeriodic h ∧ Period h (2 * Real.pi) ∧
    ¬IsPeriodic (g - h) := by
  sorry

end exists_periodic_functions_with_nonperiodic_difference_l2344_234421


namespace square_difference_l2344_234411

theorem square_difference (a b : ℝ) (h1 : a + b = 5) (h2 : a - b = 2) : a^2 - b^2 = 10 := by
  sorry

end square_difference_l2344_234411


namespace exists_row_or_column_with_sqrt_n_distinct_l2344_234489

/-- Represents a grid with n rows and n columns -/
structure Grid (n : ℕ) where
  entries : Fin n → Fin n → Fin n

/-- A grid is valid if each number from 1 to n appears exactly n times -/
def isValidGrid {n : ℕ} (g : Grid n) : Prop :=
  ∀ k : Fin n, (Finset.sum Finset.univ (λ i => Finset.sum Finset.univ (λ j => if g.entries i j = k then 1 else 0))) = n

/-- The number of distinct elements in a row -/
def distinctInRow {n : ℕ} (g : Grid n) (i : Fin n) : ℕ :=
  Finset.card (Finset.image (g.entries i) Finset.univ)

/-- The number of distinct elements in a column -/
def distinctInColumn {n : ℕ} (g : Grid n) (j : Fin n) : ℕ :=
  Finset.card (Finset.image (λ i => g.entries i j) Finset.univ)

/-- The main theorem -/
theorem exists_row_or_column_with_sqrt_n_distinct {n : ℕ} (g : Grid n) (h : isValidGrid g) :
  (∃ i : Fin n, distinctInRow g i ≥ Int.ceil (Real.sqrt n)) ∨
  (∃ j : Fin n, distinctInColumn g j ≥ Int.ceil (Real.sqrt n)) := by
  sorry

end exists_row_or_column_with_sqrt_n_distinct_l2344_234489


namespace return_flight_is_98_minutes_l2344_234441

/-- Represents the flight scenario between two cities --/
structure FlightScenario where
  outbound_time : ℝ
  total_time : ℝ
  still_air_difference : ℝ

/-- Calculates the return flight time given a flight scenario --/
def return_flight_time (scenario : FlightScenario) : ℝ :=
  scenario.total_time - scenario.outbound_time

/-- Theorem stating that the return flight time is 98 minutes --/
theorem return_flight_is_98_minutes (scenario : FlightScenario) 
  (h1 : scenario.outbound_time = 120)
  (h2 : scenario.total_time = 222)
  (h3 : scenario.still_air_difference = 6) :
  return_flight_time scenario = 98 := by
  sorry

#eval return_flight_time { outbound_time := 120, total_time := 222, still_air_difference := 6 }

end return_flight_is_98_minutes_l2344_234441


namespace smallest_advantageous_discount_l2344_234438

theorem smallest_advantageous_discount : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > 0 → m < n → 
    (1 - m / 100 : ℝ) ≥ (1 - 0.15)^2 ∨ 
    (1 - m / 100 : ℝ) ≥ (1 - 0.10)^3 ∨ 
    (1 - m / 100 : ℝ) ≥ (1 - 0.25) * (1 - 0.05)) ∧
  (1 - n / 100 : ℝ) < (1 - 0.15)^2 ∧
  (1 - n / 100 : ℝ) < (1 - 0.10)^3 ∧
  (1 - n / 100 : ℝ) < (1 - 0.25) * (1 - 0.05) ∧
  n = 29 :=
by sorry

end smallest_advantageous_discount_l2344_234438


namespace multiples_of_four_between_70_and_300_l2344_234472

theorem multiples_of_four_between_70_and_300 : 
  (Finset.filter (fun n => n % 4 = 0 ∧ n > 70 ∧ n < 300) (Finset.range 300)).card = 57 := by
  sorry

end multiples_of_four_between_70_and_300_l2344_234472


namespace custom_mul_one_one_eq_neg_eleven_l2344_234402

/-- Custom multiplication operation -/
def custom_mul (a b : ℝ) (x y : ℝ) : ℝ := a * x + b * y + 1

/-- Theorem: Given the conditions, 1 * 1 = -11 -/
theorem custom_mul_one_one_eq_neg_eleven 
  (a b : ℝ) 
  (h1 : custom_mul a b 3 5 = 15) 
  (h2 : custom_mul a b 4 7 = 28) : 
  custom_mul a b 1 1 = -11 :=
by sorry

end custom_mul_one_one_eq_neg_eleven_l2344_234402


namespace volunteer_group_selection_l2344_234468

def class_size : ℕ := 4
def total_classes : ℕ := 4
def group_size : ℕ := class_size * total_classes
def selection_size : ℕ := 3

def select_committee (n k : ℕ) : ℕ := Nat.choose n k

theorem volunteer_group_selection :
  let with_class3 := select_committee class_size 1 * select_committee (group_size - class_size) (selection_size - 1)
  let without_class3 := select_committee (group_size - class_size) selection_size - 
                        (total_classes - 1) * select_committee class_size selection_size
  with_class3 + without_class3 = 472 := by sorry

end volunteer_group_selection_l2344_234468


namespace fourth_number_proof_l2344_234449

theorem fourth_number_proof : ∃ x : ℕ, 9548 + 7314 = 3362 + x ∧ x = 13500 := by
  sorry

end fourth_number_proof_l2344_234449


namespace solve_quadratic_l2344_234454

theorem solve_quadratic (x : ℝ) (h1 : x^2 - 4*x = 0) (h2 : x ≠ 0) : x = 4 := by
  sorry

end solve_quadratic_l2344_234454


namespace square_difference_l2344_234481

theorem square_difference (a b : ℝ) (h1 : a + b = -2) (h2 : a - b = 4) : a^2 - b^2 = -8 := by
  sorry

end square_difference_l2344_234481


namespace max_value_2sin_l2344_234413

theorem max_value_2sin (x : ℝ) : ∃ (M : ℝ), M = 2 ∧ ∀ y : ℝ, 2 * Real.sin x ≤ M := by
  sorry

end max_value_2sin_l2344_234413


namespace no_answer_paradox_correct_answer_is_no_l2344_234486

/-- Represents the possible answers Alice can give to the Black Queen's question -/
inductive Answer
  | Yes
  | No

/-- Represents the possible outcomes of Alice's exam -/
inductive ExamResult
  | Pass
  | Fail

/-- Represents the Black Queen's judgment based on Alice's answer -/
def blackQueenJudgment (answer : Answer) : ExamResult → Prop :=
  match answer with
  | Answer.Yes => fun result => 
      (result = ExamResult.Pass → False) ∧ 
      (result = ExamResult.Fail → False)
  | Answer.No => fun result => 
      (result = ExamResult.Pass → False) ∧ 
      (result = ExamResult.Fail → False)

/-- Theorem stating that answering "No" creates an unresolvable paradox -/
theorem no_answer_paradox : 
  ∀ (result : ExamResult), blackQueenJudgment Answer.No result → False :=
by
  sorry

/-- Theorem stating that "No" is the correct answer to avoid failing the exam -/
theorem correct_answer_is_no : 
  ∀ (answer : Answer), 
    (∀ (result : ExamResult), blackQueenJudgment answer result → False) → 
    answer = Answer.No :=
by
  sorry

end no_answer_paradox_correct_answer_is_no_l2344_234486


namespace right_triangle_hypotenuse_l2344_234453

theorem right_triangle_hypotenuse : ∀ (a b c : ℝ),
  a = 1 → b = 2 → c^2 = a^2 + b^2 → c = Real.sqrt 5 := by
  sorry

end right_triangle_hypotenuse_l2344_234453


namespace largest_solution_of_equation_l2344_234417

theorem largest_solution_of_equation (x : ℝ) : 
  (3 * (10 * x^2 + 11 * x + 12) = x * (10 * x - 45)) →
  x ≤ (-39 + Real.sqrt 801) / 20 := by
sorry

end largest_solution_of_equation_l2344_234417


namespace cyclist_speed_l2344_234432

/-- The cyclist's problem -/
theorem cyclist_speed (initial_time : ℝ) (faster_time : ℝ) (faster_speed : ℝ) :
  initial_time = 6 →
  faster_time = 3 →
  faster_speed = 14 →
  ∃ (distance : ℝ) (initial_speed : ℝ),
    distance = initial_speed * initial_time ∧
    distance = faster_speed * faster_time ∧
    initial_speed = 7 :=
by sorry

end cyclist_speed_l2344_234432


namespace circle_radius_l2344_234497

theorem circle_radius (x y : ℝ) : (x - 1)^2 + y^2 = 9 → 3 = Real.sqrt ((x - 1)^2 + y^2) := by
  sorry

end circle_radius_l2344_234497


namespace roses_cut_correct_l2344_234499

/-- The number of roses Mary cut from her garden -/
def roses_cut (initial_roses final_roses : ℕ) : ℕ :=
  final_roses - initial_roses

theorem roses_cut_correct (initial_roses final_roses : ℕ) 
  (h : initial_roses ≤ final_roses) : 
  roses_cut initial_roses final_roses = final_roses - initial_roses :=
by
  sorry

#eval roses_cut 6 16  -- Should output 10

end roses_cut_correct_l2344_234499


namespace share_difference_for_given_distribution_l2344_234475

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

end share_difference_for_given_distribution_l2344_234475


namespace right_triangle_rotation_volumes_l2344_234415

theorem right_triangle_rotation_volumes 
  (a b c : ℝ) 
  (h_right_triangle : a^2 + b^2 = c^2) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) : 
  ∃ (K₁ K₂ K₃ : ℝ),
    K₁ = (2/3) * a * b^2 * Real.pi ∧
    K₂ = (2/3) * a^2 * b * Real.pi ∧
    K₃ = (2/3) * (a^2 * b^2) / c * Real.pi :=
by sorry

end right_triangle_rotation_volumes_l2344_234415


namespace min_value_expression_l2344_234407

theorem min_value_expression :
  (∀ x y : ℝ, 3 * x^2 + 3 * x * y + y^2 - 3 * x + 3 * y + 9 ≥ -1.125) ∧
  (∃ x y : ℝ, 3 * x^2 + 3 * x * y + y^2 - 3 * x + 3 * y + 9 = -1.125) := by
  sorry

end min_value_expression_l2344_234407


namespace breakfast_cost_is_30_25_l2344_234492

/-- Represents the menu prices and orders for a breakfast at a cafe. -/
structure BreakfastOrder where
  toast_price : ℝ
  egg_price : ℝ
  coffee_price : ℝ
  juice_price : ℝ
  dale_toast : ℕ
  dale_eggs : ℕ
  dale_coffee : ℕ
  andrew_toast : ℕ
  andrew_eggs : ℕ
  andrew_juice : ℕ
  melanie_toast : ℕ
  melanie_eggs : ℕ
  melanie_juice : ℕ
  service_charge_rate : ℝ

/-- Calculates the total cost of a breakfast order including service charge. -/
def totalCost (order : BreakfastOrder) : ℝ :=
  let subtotal := 
    order.toast_price * (order.dale_toast + order.andrew_toast + order.melanie_toast : ℝ) +
    order.egg_price * (order.dale_eggs + order.andrew_eggs + order.melanie_eggs : ℝ) +
    order.coffee_price * (order.dale_coffee : ℝ) +
    order.juice_price * (order.andrew_juice + order.melanie_juice : ℝ)
  subtotal * (1 + order.service_charge_rate)

/-- Theorem stating that the total cost of the given breakfast order is £30.25. -/
theorem breakfast_cost_is_30_25 : 
  let order : BreakfastOrder := {
    toast_price := 1,
    egg_price := 3,
    coffee_price := 2,
    juice_price := 1.5,
    dale_toast := 2,
    dale_eggs := 2,
    dale_coffee := 1,
    andrew_toast := 1,
    andrew_eggs := 2,
    andrew_juice := 1,
    melanie_toast := 3,
    melanie_eggs := 1,
    melanie_juice := 2,
    service_charge_rate := 0.1
  }
  totalCost order = 30.25 := by sorry

end breakfast_cost_is_30_25_l2344_234492


namespace smallest_absolute_value_of_z_l2344_234416

open Complex

theorem smallest_absolute_value_of_z (z : ℂ) (h : abs (z - 12) + abs (z - 5*I) = 13) :
  ∃ (w : ℂ), abs (z - 12) + abs (z - 5*I) = 13 ∧ abs w ≤ abs z ∧ abs w = 60 / 13 :=
sorry

end smallest_absolute_value_of_z_l2344_234416


namespace pascal_identity_l2344_234448

theorem pascal_identity (n k : ℕ) (h1 : k ≤ n) (h2 : ¬(n = 0 ∧ k = 0)) : 
  Nat.choose n k = Nat.choose (n - 1) k + Nat.choose (n - 1) (k - 1) :=
by sorry

end pascal_identity_l2344_234448


namespace vasyas_premium_will_increase_l2344_234427

/-- Represents a car insurance policy -/
structure CarInsurancePolicy where
  premium : ℝ
  hadAccident : Bool

/-- Represents an insurance company -/
class InsuranceCompany where
  renewPolicy : CarInsurancePolicy → CarInsurancePolicy

/-- Axiom: Insurance companies increase premiums for policies with accidents -/
axiom premium_increase_after_accident (company : InsuranceCompany) (policy : CarInsurancePolicy) :
  policy.hadAccident → (company.renewPolicy policy).premium > policy.premium

/-- Theorem: Vasya's insurance premium will increase after his car accident -/
theorem vasyas_premium_will_increase (company : InsuranceCompany) (vasyas_policy : CarInsurancePolicy) 
    (h_accident : vasyas_policy.hadAccident) : 
  (company.renewPolicy vasyas_policy).premium > vasyas_policy.premium :=
by
  sorry


end vasyas_premium_will_increase_l2344_234427


namespace complex_number_quadrant_l2344_234480

theorem complex_number_quadrant : ∃ (z : ℂ), z = (1 - Complex.I) / Complex.I ∧ z.re < 0 ∧ z.im < 0 := by
  sorry

end complex_number_quadrant_l2344_234480


namespace stock_profit_percentage_l2344_234451

theorem stock_profit_percentage 
  (stock_worth : ℝ) 
  (profit_portion : ℝ) 
  (loss_portion : ℝ) 
  (loss_percentage : ℝ) 
  (overall_loss : ℝ) 
  (h1 : stock_worth = 12499.99)
  (h2 : profit_portion = 0.2)
  (h3 : loss_portion = 0.8)
  (h4 : loss_percentage = 0.1)
  (h5 : overall_loss = 500) :
  ∃ (P : ℝ), 
    (stock_worth * profit_portion * (1 + P / 100) + 
     stock_worth * loss_portion * (1 - loss_percentage) = 
     stock_worth - overall_loss) ∧ 
    (abs (P - 20.04) < 0.01) := by
  sorry

end stock_profit_percentage_l2344_234451


namespace circle_equation_l2344_234484

-- Define the center of the circle
def center : ℝ × ℝ := (3, 1)

-- Define a point on the circle (the origin)
def origin : ℝ × ℝ := (0, 0)

-- Define the equation of a circle
def is_on_circle (x y : ℝ) : Prop :=
  (x - center.1)^2 + (y - center.2)^2 = (origin.1 - center.1)^2 + (origin.2 - center.2)^2

-- Theorem statement
theorem circle_equation : 
  ∀ x y : ℝ, is_on_circle x y ↔ (x - 3)^2 + (y - 1)^2 = 10 :=
by sorry

end circle_equation_l2344_234484


namespace total_children_l2344_234401

theorem total_children (happy sad neutral boys girls happy_boys sad_girls : ℕ) :
  happy = 30 →
  sad = 10 →
  neutral = 20 →
  boys = 18 →
  girls = 42 →
  happy_boys = 6 →
  sad_girls = 4 →
  happy + sad + neutral = boys + girls :=
by sorry

end total_children_l2344_234401


namespace inequality_proof_l2344_234476

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (2 * a) / (a^2 + b * c) + (2 * b) / (b^2 + c * a) + (2 * c) / (c^2 + a * b) ≤ 
  a / (b * c) + b / (c * a) + c / (a * b) := by
sorry

end inequality_proof_l2344_234476


namespace colored_balls_permutations_l2344_234473

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def multiset_permutations (n : ℕ) (counts : List ℕ) : ℕ :=
  factorial n / (counts.map factorial).prod

theorem colored_balls_permutations :
  let total_balls : ℕ := 5
  let color_counts : List ℕ := [1, 1, 2, 1]  -- red, blue, yellow, white
  multiset_permutations total_balls color_counts = 60 := by
  sorry

end colored_balls_permutations_l2344_234473


namespace square_conditions_solutions_l2344_234408

def is_perfect_square (n : ℤ) : Prop := ∃ m : ℤ, n = m^2

theorem square_conditions_solutions :
  ∀ a b : ℕ+,
  (is_perfect_square (a.val^2 - 4*b.val) ∧ is_perfect_square (b.val^2 - 4*a.val)) ↔
  ((a, b) = (⟨4, by norm_num⟩, ⟨4, by norm_num⟩) ∨
   (a, b) = (⟨5, by norm_num⟩, ⟨6, by norm_num⟩) ∨
   (a, b) = (⟨6, by norm_num⟩, ⟨5, by norm_num⟩)) :=
by sorry

end square_conditions_solutions_l2344_234408


namespace angle_with_complement_33_percent_of_supplement_is_45_degrees_l2344_234403

theorem angle_with_complement_33_percent_of_supplement_is_45_degrees (x : ℝ) :
  (90 - x = (1 / 3) * (180 - x)) → x = 45 := by
  sorry

end angle_with_complement_33_percent_of_supplement_is_45_degrees_l2344_234403


namespace natural_number_equality_l2344_234496

def divisible (a b : ℕ) : Prop := ∃ k : ℕ, b = k * a

theorem natural_number_equality (a b : ℕ) 
  (h : ∃ S : Set ℕ, Set.Infinite S ∧ ∀ n ∈ S, divisible (a^(n+1) + b^(n+1)) (a^n + b^n)) :
  a = b :=
sorry

end natural_number_equality_l2344_234496


namespace bus_passenger_count_l2344_234470

/-- Calculates the final number of passengers on a bus after several stops -/
def final_passengers (initial : ℕ) (first_stop : ℕ) (off_other_stops : ℕ) (on_other_stops : ℕ) : ℕ :=
  initial + first_stop - off_other_stops + on_other_stops

/-- Theorem stating that given the specific passenger changes, the final number is 49 -/
theorem bus_passenger_count : final_passengers 50 16 22 5 = 49 := by
  sorry

end bus_passenger_count_l2344_234470


namespace range_of_m_l2344_234434

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, (x^2 - 2*x - 3 > 0) → (x < m - 1 ∨ x > m + 1)) ∧ 
  (∃ x : ℝ, (x < m - 1 ∨ x > m + 1) ∧ ¬(x^2 - 2*x - 3 > 0)) →
  0 ≤ m ∧ m ≤ 2 :=
by sorry

end range_of_m_l2344_234434


namespace long_sleeve_shirts_to_wash_l2344_234447

theorem long_sleeve_shirts_to_wash :
  ∀ (total_shirts short_sleeve_shirts long_sleeve_shirts shirts_washed shirts_not_washed : ℕ),
    total_shirts = short_sleeve_shirts + long_sleeve_shirts →
    shirts_washed = 29 →
    shirts_not_washed = 1 →
    short_sleeve_shirts = 9 →
    long_sleeve_shirts = 19 :=
by
  sorry

end long_sleeve_shirts_to_wash_l2344_234447


namespace quadratic_no_roots_if_geometric_sequence_l2344_234404

/-- A geometric sequence is a sequence where each term after the first is found by 
    multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a b c : ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ b = a * r ∧ c = b * r

/-- A quadratic function f(x) = ax² + bx + c has no real roots if and only if
    its discriminant is negative. -/
def HasNoRealRoots (a b c : ℝ) : Prop :=
  b^2 - 4*a*c < 0

theorem quadratic_no_roots_if_geometric_sequence (a b c : ℝ) (ha : a ≠ 0) :
  IsGeometricSequence a b c → HasNoRealRoots a b c := by
  sorry

end quadratic_no_roots_if_geometric_sequence_l2344_234404


namespace lottery_probability_l2344_234460

/-- The probability of winning in a lottery with 10 balls labeled 1 to 10, 
    where winning occurs if the selected number is not less than 6. -/
theorem lottery_probability : 
  let total_balls : ℕ := 10
  let winning_balls : ℕ := 5
  let probability := winning_balls / total_balls
  probability = 1 / 2 := by
sorry

end lottery_probability_l2344_234460


namespace negation_of_proposition_l2344_234482

theorem negation_of_proposition (p : Prop) : 
  (¬(∃ x₀ : ℝ, x₀ ∈ Set.Icc (-3) 3 ∧ x₀^2 + 2*x₀ + 1 ≤ 0)) ↔ 
  (∀ x : ℝ, x ∈ Set.Icc (-3) 3 → x^2 + 2*x + 1 > 0) :=
by sorry

end negation_of_proposition_l2344_234482


namespace function_divisibility_property_l2344_234435

theorem function_divisibility_property (f : ℤ → ℤ) : 
  (∀ m n : ℤ, (Int.gcd m n : ℤ) ∣ (f m + f n)) → 
  ∃ k : ℤ, ∀ n : ℤ, f n = k * n :=
by sorry

end function_divisibility_property_l2344_234435


namespace square_root_problem_l2344_234409

theorem square_root_problem (x y : ℝ) : 
  (∃ k : ℤ, x + 7 = (3 * k)^2) → 
  ((2*x - y - 13)^(1/3) = -2) → 
  (5*x - 6*y)^(1/2) = 4 := by
sorry

end square_root_problem_l2344_234409


namespace parallelogram_exists_l2344_234443

/-- Represents a cell in the grid -/
structure Cell where
  x : Nat
  y : Nat

/-- Represents the grid and its blue cells -/
structure Grid where
  n : Nat
  blue_cells : Finset Cell

/-- Predicate to check if four cells form a parallelogram -/
def is_parallelogram (c1 c2 c3 c4 : Cell) : Prop :=
  (c2.x - c1.x = c4.x - c3.x) ∧ (c2.y - c1.y = c4.y - c3.y)

/-- Main theorem: In an n x n grid with 2n blue cells, there exist 4 blue cells forming a parallelogram -/
theorem parallelogram_exists (g : Grid) (h1 : g.blue_cells.card = 2 * g.n) :
  ∃ c1 c2 c3 c4 : Cell, c1 ∈ g.blue_cells ∧ c2 ∈ g.blue_cells ∧ c3 ∈ g.blue_cells ∧ c4 ∈ g.blue_cells ∧
    is_parallelogram c1 c2 c3 c4 :=
sorry

end parallelogram_exists_l2344_234443


namespace midpoint_property_l2344_234465

/-- Given two points P and Q in the plane, their midpoint R satisfies 3x - 2y = -15 --/
theorem midpoint_property (P Q R : ℝ × ℝ) : 
  P = (-8, 15) → 
  Q = (6, -3) → 
  R.1 = (P.1 + Q.1) / 2 → 
  R.2 = (P.2 + Q.2) / 2 → 
  3 * R.1 - 2 * R.2 = -15 := by
sorry

end midpoint_property_l2344_234465


namespace smallest_x_value_l2344_234495

/-- Given the equation x|x| = 3x + k and the inequality x + 2 ≤ 3,
    the smallest value of x that satisfies these conditions is -2 when k = 2. -/
theorem smallest_x_value (x : ℝ) (k : ℝ) : 
  (x * abs x = 3 * x + k) → 
  (x + 2 ≤ 3) → 
  (k = 2) →
  (∀ y : ℝ, (y * abs y = 3 * y + k) → (y + 2 ≤ 3) → (x ≤ y)) →
  x = -2 := by
  sorry

end smallest_x_value_l2344_234495


namespace third_number_value_l2344_234420

theorem third_number_value (a b c : ℕ) : 
  a + b + c = 500 → 
  a = 200 → 
  b = 2 * c → 
  c = 100 := by
sorry

end third_number_value_l2344_234420


namespace hyperbola_eccentricity_l2344_234456

/-- The eccentricity of a hyperbola with given properties -/
theorem hyperbola_eccentricity (a b c : ℝ) (ha : a > 0) (hb : b > 0) : 
  let hyperbola := fun (x y : ℝ) => x^2 / a^2 - y^2 / b^2 = 1
  let line := fun (x y : ℝ) => x / a + y / b = 1
  let foci_distance_sum := 4 * c / 5
  let eccentricity := c / a
  (∀ x y, hyperbola x y → line x y) → 
  (foci_distance_sum = 2 * b) →
  eccentricity = 5 * Real.sqrt 21 / 21 :=
by sorry

end hyperbola_eccentricity_l2344_234456


namespace p_adic_valuation_properties_l2344_234464

/-- The p-adic valuation of an integer -/
noncomputable def v_p (p : ℕ) (n : ℤ) : ℕ := sorry

/-- Properties of p-adic valuation for prime p and integers m, n -/
theorem p_adic_valuation_properties (p : ℕ) (m n : ℤ) (hp : Nat.Prime p) :
  (v_p p (m * n) = v_p p m + v_p p n) ∧
  (v_p p (m + n) ≥ min (v_p p m) (v_p p n)) ∧
  (v_p p (Int.gcd m n) = min (v_p p m) (v_p p n)) ∧
  (v_p p (Int.lcm m n) = max (v_p p m) (v_p p n)) :=
by sorry

end p_adic_valuation_properties_l2344_234464


namespace a_6_equals_11_l2344_234478

/-- Given a sequence {aₙ} where Sₙ is the sum of its first n terms -/
def S (n : ℕ) : ℕ := n^2 + 1

/-- The nth term of the sequence -/
def a (n : ℕ) : ℕ := S n - S (n-1)

/-- Proof that the 6th term of the sequence is 11 -/
theorem a_6_equals_11 : a 6 = 11 := by
  sorry

end a_6_equals_11_l2344_234478


namespace sum_of_squares_of_roots_l2344_234461

theorem sum_of_squares_of_roots : ∃ (x₁ x₂ : ℝ),
  (x₁^2 - 2*x₁ + 4)^(x₁^2 - 2*x₁ + 3) = 625 ∧
  (x₂^2 - 2*x₂ + 4)^(x₂^2 - 2*x₂ + 3) = 625 ∧
  x₁ ≠ x₂ ∧
  x₁^2 + x₂^2 = 6 :=
by sorry

end sum_of_squares_of_roots_l2344_234461


namespace quadrilateral_diagonals_l2344_234433

-- Define a convex quadrilateral
structure ConvexQuadrilateral :=
  (perimeter : ℝ)
  (diagonal1 : ℝ)
  (diagonal2 : ℝ)
  (is_convex : perimeter > 0 ∧ diagonal1 > 0 ∧ diagonal2 > 0)

-- Theorem statement
theorem quadrilateral_diagonals 
  (q : ConvexQuadrilateral) 
  (h1 : q.perimeter = 2004) 
  (h2 : q.diagonal1 = 1001) : 
  (q.diagonal2 ≠ 1) ∧ 
  (∃ q' : ConvexQuadrilateral, q'.perimeter = 2004 ∧ q'.diagonal1 = 1001 ∧ q'.diagonal2 = 2) ∧
  (∃ q'' : ConvexQuadrilateral, q''.perimeter = 2004 ∧ q''.diagonal1 = 1001 ∧ q''.diagonal2 = 1001) :=
by sorry

end quadrilateral_diagonals_l2344_234433


namespace decreasing_interval_of_even_f_l2344_234462

def f (k : ℝ) (x : ℝ) : ℝ := (k - 2) * x^2 + (k - 1) * x + 3

theorem decreasing_interval_of_even_f (k : ℝ) :
  (∀ x, f k x = f k (-x)) →
  ∀ x > 0, ∀ y > x, f k y < f k x :=
by sorry

end decreasing_interval_of_even_f_l2344_234462


namespace equal_probabilities_l2344_234450

/-- Represents a box containing balls of different colors -/
structure Box where
  red : ℕ
  green : ℕ

/-- Represents the state of both boxes -/
structure BoxState where
  red_box : Box
  green_box : Box

/-- Initial state of the boxes -/
def initial_state : BoxState :=
  { red_box := { red := 100, green := 0 },
    green_box := { red := 0, green := 100 } }

/-- State after the first transfer -/
def first_transfer (state : BoxState) : BoxState :=
  { red_box := { red := state.red_box.red - 8, green := state.red_box.green },
    green_box := { red := state.green_box.red + 8, green := state.green_box.green } }

/-- State after the second transfer -/
def second_transfer (state : BoxState) : BoxState :=
  { red_box := { red := state.red_box.red, green := state.red_box.green + 1 },
    green_box := { red := state.green_box.red - 1, green := state.green_box.green - 7 } }

/-- Calculate the probability of drawing a specific color from a box -/
def draw_probability (box : Box) (color : String) : ℚ :=
  match color with
  | "red" => (box.red : ℚ) / (box.red + box.green : ℚ)
  | "green" => (box.green : ℚ) / (box.red + box.green : ℚ)
  | _ => 0

/-- The main theorem to prove -/
theorem equal_probabilities :
  let final_state := second_transfer (first_transfer initial_state)
  (draw_probability final_state.red_box "green") = (draw_probability final_state.green_box "red") := by
  sorry

end equal_probabilities_l2344_234450


namespace arithmetic_calculations_l2344_234425

theorem arithmetic_calculations : 
  (2 / 5 - 1 / 5 * (-5) + 3 / 5 = 2) ∧ 
  (-2^2 - (-3)^3 / 3 * (1 / 3) = -1) := by sorry

end arithmetic_calculations_l2344_234425


namespace max_expensive_product_price_is_30900_l2344_234405

/-- Represents a company's product line -/
structure ProductLine where
  total_products : Nat
  average_price : ℝ
  min_price : ℝ
  num_below_threshold : Nat
  threshold : ℝ

/-- Calculates the maximum possible price for the most expensive product -/
def max_expensive_product_price (pl : ProductLine) : ℝ :=
  let total_value := pl.total_products * pl.average_price
  let min_value_below_threshold := pl.num_below_threshold * pl.min_price
  let remaining_products := pl.total_products - pl.num_below_threshold
  let remaining_value := total_value - min_value_below_threshold
  let value_at_threshold := (remaining_products - 1) * pl.threshold
  remaining_value - value_at_threshold

/-- Theorem stating the maximum price of the most expensive product -/
theorem max_expensive_product_price_is_30900 :
  let pl := ProductLine.mk 40 1800 500 15 1400
  max_expensive_product_price pl = 30900 := by
  sorry

end max_expensive_product_price_is_30900_l2344_234405


namespace prism_with_seven_faces_has_fifteen_edges_l2344_234436

/-- A prism is a polyhedron with two congruent and parallel faces (bases) 
    and all other faces are parallelograms (lateral faces). -/
structure Prism where
  faces : ℕ
  bases : ℕ
  lateral_faces : ℕ
  edges_per_base : ℕ
  lateral_edges : ℕ
  total_edges : ℕ

/-- The number of edges in a prism with 7 faces is 15. -/
theorem prism_with_seven_faces_has_fifteen_edges :
  ∀ (p : Prism), p.faces = 7 → p.total_edges = 15 := by
  sorry


end prism_with_seven_faces_has_fifteen_edges_l2344_234436


namespace jersey_profit_calculation_l2344_234418

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

end jersey_profit_calculation_l2344_234418


namespace projection_equality_l2344_234426

def a : Fin 2 → ℚ := ![3, -2]
def b : Fin 2 → ℚ := ![6, -1]
def p : Fin 2 → ℚ := ![9/10, -27/10]

theorem projection_equality (v : Fin 2 → ℚ) (hv : v ≠ 0) :
  (v • a / (v • v)) • v = (v • b / (v • b)) • v → 
  (v • a / (v • v)) • v = p :=
by sorry

end projection_equality_l2344_234426


namespace largest_inscribed_semicircle_area_l2344_234455

theorem largest_inscribed_semicircle_area (r : ℝ) (h : r = 1) : 
  let A := π * (1 / Real.sqrt 3)^2 / 2
  120 * A / π = 20 := by sorry

end largest_inscribed_semicircle_area_l2344_234455


namespace sum_of_digits_of_seven_to_twelve_l2344_234467

/-- The sum of the tens digit and the ones digit of (1+6)^12 is 1 -/
theorem sum_of_digits_of_seven_to_twelve : 
  (((1 + 6)^12 / 10) % 10 + (1 + 6)^12 % 10) = 1 := by
  sorry

end sum_of_digits_of_seven_to_twelve_l2344_234467


namespace min_fraction_sum_l2344_234471

theorem min_fraction_sum (a b c d : ℕ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  1 ≤ a ∧ a ≤ 10 →
  1 ≤ b ∧ b ≤ 10 →
  1 ≤ c ∧ c ≤ 10 →
  1 ≤ d ∧ d ≤ 10 →
  (a : ℚ) / b + (c : ℚ) / d ≥ 14 / 45 := by
  sorry

end min_fraction_sum_l2344_234471


namespace fourth_term_of_sequence_l2344_234430

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n, a (n + 1) = q * a n

theorem fourth_term_of_sequence (a : ℕ → ℝ) :
  a 1 = 1 → (∀ n, a (n + 1) = 2 * a n) → a 4 = 8 := by
sorry

end fourth_term_of_sequence_l2344_234430


namespace floor_ceiling_evaluation_l2344_234469

theorem floor_ceiling_evaluation : ⌊(1.999 : ℝ)⌋ + ⌈(3.001 : ℝ)⌉ - ⌊(0.001 : ℝ)⌋ = 5 := by
  sorry

end floor_ceiling_evaluation_l2344_234469


namespace relative_error_comparison_l2344_234442

theorem relative_error_comparison :
  let line1_length : ℚ := 15
  let line1_error : ℚ := 3 / 100
  let line2_length : ℚ := 125
  let line2_error : ℚ := 1 / 4
  let relative_error1 : ℚ := line1_error / line1_length
  let relative_error2 : ℚ := line2_error / line2_length
  relative_error1 = relative_error2 :=
by sorry

end relative_error_comparison_l2344_234442


namespace exp_13pi_i_div_2_eq_i_l2344_234410

-- Define the complex exponential function
noncomputable def complex_exp (z : ℂ) : ℂ := Real.exp z.re * (Complex.cos z.im + Complex.I * Complex.sin z.im)

-- State the theorem
theorem exp_13pi_i_div_2_eq_i : complex_exp (13 * Real.pi * Complex.I / 2) = Complex.I := by
  sorry

end exp_13pi_i_div_2_eq_i_l2344_234410


namespace december_burger_expenditure_l2344_234445

/-- The daily expenditure on burgers given the total monthly expenditure and number of days -/
def daily_burger_expenditure (total_expenditure : ℚ) (days : ℕ) : ℚ :=
  total_expenditure / days

theorem december_burger_expenditure :
  let total_expenditure : ℚ := 465
  let days : ℕ := 31
  daily_burger_expenditure total_expenditure days = 15 := by
sorry

end december_burger_expenditure_l2344_234445


namespace find_b_value_l2344_234428

theorem find_b_value (x y b : ℝ) 
  (eq1 : 7^(3*x - 1) * b^(4*y - 3) = 49^x * 27^y)
  (eq2 : x + y = 4) : 
  b = 3 := by sorry

end find_b_value_l2344_234428


namespace nancy_gardens_l2344_234493

theorem nancy_gardens (total_seeds : ℕ) (big_garden_seeds : ℕ) (seeds_per_small_garden : ℕ) 
  (h1 : total_seeds = 52)
  (h2 : big_garden_seeds = 28)
  (h3 : seeds_per_small_garden = 4) :
  (total_seeds - big_garden_seeds) / seeds_per_small_garden = 6 :=
by sorry

end nancy_gardens_l2344_234493


namespace smallest_positive_angle_correct_largest_negative_angle_correct_equivalent_angles_in_range_correct_l2344_234423

-- Define the original angle
def original_angle : Int := -2010

-- Define a function to find the smallest positive equivalent angle
def smallest_positive_equivalent (angle : Int) : Int :=
  angle % 360

-- Define a function to find the largest negative equivalent angle
def largest_negative_equivalent (angle : Int) : Int :=
  (angle % 360) - 360

-- Define a function to find equivalent angles within a range
def equivalent_angles_in_range (angle : Int) (lower : Int) (upper : Int) : List Int :=
  let base_angle := angle % 360
  List.filter (fun x => lower ≤ x ∧ x < upper)
    [base_angle - 720, base_angle - 360, base_angle, base_angle + 360]

-- Theorem statements
theorem smallest_positive_angle_correct :
  smallest_positive_equivalent original_angle = 150 := by sorry

theorem largest_negative_angle_correct :
  largest_negative_equivalent original_angle = -210 := by sorry

theorem equivalent_angles_in_range_correct :
  equivalent_angles_in_range original_angle (-720) 720 = [-570, -210, 150, 510] := by sorry

end smallest_positive_angle_correct_largest_negative_angle_correct_equivalent_angles_in_range_correct_l2344_234423


namespace base_conversion_difference_l2344_234431

/-- Converts a number from base 7 to base 10 --/
def base7ToBase10 (n : Nat) : Nat :=
  (n / 100000) * 7^5 +
  ((n / 10000) % 10) * 7^4 +
  ((n / 1000) % 10) * 7^3 +
  ((n / 100) % 10) * 7^2 +
  ((n / 10) % 10) * 7^1 +
  (n % 10) * 7^0

/-- Converts a number from base 8 to base 10 --/
def base8ToBase10 (n : Nat) : Nat :=
  (n / 10000) * 8^4 +
  ((n / 1000) % 10) * 8^3 +
  ((n / 100) % 10) * 8^2 +
  ((n / 10) % 10) * 8^1 +
  (n % 10) * 8^0

theorem base_conversion_difference :
  base7ToBase10 543210 - base8ToBase10 43210 = 76717 := by
  sorry

#eval base7ToBase10 543210 - base8ToBase10 43210

end base_conversion_difference_l2344_234431


namespace geometric_sequence_a8_l2344_234466

/-- A sequence where a_n + 2 forms a geometric sequence -/
def IsGeometricPlus2 (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, (a (n + 1) + 2) = (a n + 2) * q

theorem geometric_sequence_a8 (a : ℕ → ℝ) 
  (h_geom : IsGeometricPlus2 a) 
  (h_a2 : a 2 = -1) 
  (h_a4 : a 4 = 2) : 
  a 8 = 62 := by
sorry

end geometric_sequence_a8_l2344_234466


namespace quadratic_roots_reciprocal_sum_l2344_234458

theorem quadratic_roots_reciprocal_sum (x₁ x₂ : ℝ) : 
  x₁^2 - 5*x₁ - 6 = 0 → 
  x₂^2 - 5*x₂ - 6 = 0 → 
  x₁ ≠ x₂ → 
  (1/x₁) + (1/x₂) = -5/6 := by
sorry

end quadratic_roots_reciprocal_sum_l2344_234458


namespace joan_payment_amount_l2344_234424

-- Define the costs and change as constants
def cat_toy_cost : ℚ := 877 / 100
def cage_cost : ℚ := 1097 / 100
def change_received : ℚ := 26 / 100

-- Define the theorem
theorem joan_payment_amount :
  cat_toy_cost + cage_cost + change_received = 20 := by
  sorry

end joan_payment_amount_l2344_234424


namespace train_speed_calculation_l2344_234498

/-- Proves that a train with given length and time to cross a pole has a specific speed in km/hr -/
theorem train_speed_calculation (train_length : Real) (crossing_time : Real) :
  train_length = 320 →
  crossing_time = 7.999360051195905 →
  ∃ (speed_kmh : Real), 
    abs (speed_kmh - (train_length / crossing_time * 3.6)) < 0.001 ∧ 
    abs (speed_kmh - 144.018) < 0.001 := by
  sorry

end train_speed_calculation_l2344_234498


namespace team_incorrect_answers_contest_result_l2344_234487

theorem team_incorrect_answers 
  (total_questions : Nat) 
  (riley_incorrect : Nat) 
  (ofelia_correct_addition : Nat) : Nat :=
  let riley_correct := total_questions - riley_incorrect
  let ofelia_correct := riley_correct / 2 + ofelia_correct_addition
  let ofelia_incorrect := total_questions - ofelia_correct
  riley_incorrect + ofelia_incorrect

#check @team_incorrect_answers

theorem contest_result : 
  team_incorrect_answers 35 3 5 = 17 := by
  sorry

#check @contest_result

end team_incorrect_answers_contest_result_l2344_234487


namespace evaluate_expression_l2344_234479

theorem evaluate_expression : 11 + Real.sqrt (-4 + 6 * 4 / 3) = 13 := by
  sorry

end evaluate_expression_l2344_234479


namespace pump_operations_proof_l2344_234452

/-- The fraction of air remaining after one pump operation -/
def pump_efficiency : ℝ := 0.5

/-- The target fraction of air remaining -/
def target_fraction : ℝ := 0.001

/-- The minimum number of pump operations needed to reach the target fraction -/
def min_operations : ℕ := 10

theorem pump_operations_proof :
  (∀ n : ℕ, n < min_operations → (pump_efficiency ^ n : ℝ) > target_fraction) ∧
  (pump_efficiency ^ min_operations : ℝ) ≤ target_fraction :=
sorry

end pump_operations_proof_l2344_234452


namespace borrowed_amount_correct_l2344_234422

/-- The amount of money Yoque borrowed -/
def borrowed_amount : ℝ := 150

/-- The number of months for repayment -/
def repayment_months : ℕ := 11

/-- The additional percentage added to the repayment -/
def additional_percentage : ℝ := 0.1

/-- The monthly payment amount -/
def monthly_payment : ℝ := 15

/-- Theorem stating that the borrowed amount satisfies the given conditions -/
theorem borrowed_amount_correct : 
  borrowed_amount * (1 + additional_percentage) = repayment_months * monthly_payment := by
  sorry


end borrowed_amount_correct_l2344_234422


namespace firecrackers_confiscated_l2344_234439

theorem firecrackers_confiscated (initial : ℕ) (remaining : ℕ) : 
  initial = 48 →
  remaining < initial →
  (1 : ℚ) / 6 * remaining = remaining - (2 * 15) →
  initial - remaining = 12 :=
by
  sorry

end firecrackers_confiscated_l2344_234439


namespace parallel_line_through_point_l2344_234483

/-- 
Given a line L1 with equation 2x + 3y - 6 = 0 and a point P(1, -1),
prove that the line L2 with equation 2x + 3y + 1 = 0 is parallel to L1 and passes through P.
-/
theorem parallel_line_through_point (x y : ℝ) : 
  (2 * x + 3 * y - 6 = 0) →  -- Equation of L1
  (2 * 1 + 3 * (-1) + 1 = 0) →  -- L2 passes through P(1, -1)
  (∀ (x y : ℝ), 2 * x + 3 * y + 1 = 0 ↔ 
    (∃ (k : ℝ), 2 * x + 3 * y = 2 * 1 + 3 * (-1) + k * (2 * 1 + 3 * (-1) - (2 * 1 + 3 * (-1))))) :=
by
  sorry

end parallel_line_through_point_l2344_234483


namespace sin_150_minus_sin_30_equals_zero_l2344_234459

theorem sin_150_minus_sin_30_equals_zero :
  Real.sin (150 * π / 180) - Real.sin (30 * π / 180) = 0 := by
  sorry

end sin_150_minus_sin_30_equals_zero_l2344_234459


namespace sum_three_numbers_l2344_234406

theorem sum_three_numbers (a b c M : ℤ) : 
  a + b + c = 75 ∧ 
  a + 4 = M ∧ 
  b - 5 = M ∧ 
  3 * c = M → 
  M = 31 := by
sorry

end sum_three_numbers_l2344_234406


namespace perpendicular_line_proof_l2344_234446

def given_line (x y : ℝ) : Prop := 2 * x + y - 5 = 0

def perpendicular_line (x y : ℝ) : Prop := x - 2 * y + 8 = 0

theorem perpendicular_line_proof :
  (∀ x y : ℝ, perpendicular_line x y → given_line x y → (x + 2) * (y - 3) = 0) ∧
  (∀ x y : ℝ, given_line x y → perpendicular_line x y → (x + 2) * (2 * x + y) + (y - 3) * (x + 2 * y) = 0) := by
  sorry

end perpendicular_line_proof_l2344_234446


namespace total_hamburger_configurations_l2344_234488

/-- The number of different condiments available. -/
def num_condiments : ℕ := 10

/-- The number of options for meat patties. -/
def meat_patty_options : ℕ := 4

/-- Theorem: The total number of different hamburger configurations. -/
theorem total_hamburger_configurations :
  (2 ^ num_condiments) * meat_patty_options = 4096 := by
  sorry

end total_hamburger_configurations_l2344_234488


namespace rational_function_sum_l2344_234414

/-- Given rational functions r and s, prove r(x) + s(x) = -x^3 + 3x under specific conditions -/
theorem rational_function_sum (r s : ℝ → ℝ) : 
  (∃ (a b : ℝ), s x = a * (x - 2) * (x + 2) * x) →  -- s(x) is cubic with roots at 2, -2, and 0
  (∃ (b : ℝ), r x = b * x) →  -- r(x) is linear with a root at 0
  r (-1) = 1 →  -- condition on r
  s 1 = 3 →  -- condition on s
  ∀ x, r x + s x = -x^3 + 3*x := by
  sorry

end rational_function_sum_l2344_234414


namespace smallest_n_for_roots_of_unity_l2344_234494

def polynomial (z : ℂ) : ℂ := z^4 + z^3 + z^2 + z + 1

def is_nth_root_of_unity (z : ℂ) (n : ℕ) : Prop := z^n = 1

theorem smallest_n_for_roots_of_unity : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (z : ℂ), polynomial z = 0 → is_nth_root_of_unity z n) ∧
  (∀ (m : ℕ), m > 0 → m < n → 
    ∃ (w : ℂ), polynomial w = 0 ∧ ¬is_nth_root_of_unity w m) ∧
  n = 5 := by sorry

end smallest_n_for_roots_of_unity_l2344_234494


namespace lamp_distance_in_specific_classroom_l2344_234490

/-- Represents a classroom with two lamps -/
structure Classroom where
  length : ℝ
  ceiling_height : ℝ
  lamp1_position : ℝ
  lamp2_position : ℝ
  lamp1_circle_diameter : ℝ
  lamp2_illumination_length : ℝ

/-- The distance between two lamps in the classroom -/
def lamp_distance (c : Classroom) : ℝ :=
  |c.lamp1_position - c.lamp2_position|

/-- Theorem stating the distance between lamps in the specific classroom setup -/
theorem lamp_distance_in_specific_classroom :
  ∀ (c : Classroom),
    c.length = 10 ∧
    c.lamp1_circle_diameter = 6 ∧
    c.lamp2_illumination_length = 10 ∧
    c.lamp1_position = c.length / 2 ∧
    c.lamp2_position = 1 →
    lamp_distance c = 4 := by
  sorry

#check lamp_distance_in_specific_classroom

end lamp_distance_in_specific_classroom_l2344_234490


namespace f_sum_equals_half_l2344_234437

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def has_period_4 (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 2) = f (x - 2)

def f_condition (f : ℝ → ℝ) : Prop :=
  ∀ x, x > -2 ∧ x < 0 → f x = -2^x

theorem f_sum_equals_half (f : ℝ → ℝ) 
  (h_odd : is_odd_function f)
  (h_period : has_period_4 f)
  (h_condition : f_condition f) :
  f 1 + f 4 = 1/2 := by
sorry

end f_sum_equals_half_l2344_234437


namespace black_squares_in_45th_row_l2344_234477

/-- Represents the number of squares in the nth row of the stair-step pattern -/
def squares_in_row (n : ℕ) : ℕ := 2 * n + 1

/-- Represents the number of black squares in the nth row of the stair-step pattern -/
def black_squares_in_row (n : ℕ) : ℕ := (squares_in_row n - 1) / 2

/-- Theorem stating that the number of black squares in the 45th row is 45 -/
theorem black_squares_in_45th_row :
  black_squares_in_row 45 = 45 := by
  sorry

end black_squares_in_45th_row_l2344_234477


namespace equivalent_operation_l2344_234400

theorem equivalent_operation (x : ℚ) : 
  (x * (2 / 3)) / (4 / 7) = x * (7 / 6) :=
by sorry

end equivalent_operation_l2344_234400


namespace typing_speed_difference_l2344_234491

theorem typing_speed_difference (before_speed after_speed : ℕ) 
  (h1 : before_speed = 10) 
  (h2 : after_speed = 8) 
  (difference : ℕ) 
  (h3 : difference = 10) : 
  ∃ (minutes : ℕ), minutes * before_speed - minutes * after_speed = difference ∧ minutes = 5 :=
sorry

end typing_speed_difference_l2344_234491


namespace sum_of_x_and_y_l2344_234474

theorem sum_of_x_and_y (x y : ℝ) 
  (hx : |x| = 1) 
  (hy : |y| = 2) 
  (hxy : x * y > 0) : 
  x + y = 3 ∨ x + y = -3 := by
  sorry

end sum_of_x_and_y_l2344_234474


namespace man_double_son_age_l2344_234412

/-- The number of years until a man's age is twice his son's age -/
def years_until_double_age (son_age : ℕ) (age_difference : ℕ) : ℕ :=
  2

/-- Theorem stating that the number of years until the man's age is twice his son's age is 2 -/
theorem man_double_son_age 
  (son_age : ℕ) 
  (age_difference : ℕ) 
  (h1 : son_age = 18) 
  (h2 : age_difference = 20) : 
  years_until_double_age son_age age_difference = 2 := by
  sorry

end man_double_son_age_l2344_234412


namespace four_integer_average_l2344_234444

theorem four_integer_average (a b c d : ℕ+) : 
  (a + b : ℚ) / 2 = 35 →
  c ≤ 130 →
  d ≤ 130 →
  (a + b + c + d : ℚ) / 4 = 50.25 :=
by sorry

end four_integer_average_l2344_234444


namespace nancy_antacid_consumption_l2344_234440

/-- Calculates the number of antacids Nancy takes per month based on her eating habits. -/
def antacids_per_month (indian_antacids : ℕ) (mexican_antacids : ℕ) (other_antacids : ℕ)
  (indian_freq : ℕ) (mexican_freq : ℕ) (days_per_week : ℕ) (weeks_per_month : ℕ) : ℕ :=
  let other_days := days_per_week - indian_freq - mexican_freq
  let weekly_antacids := indian_antacids * indian_freq + mexican_antacids * mexican_freq + other_antacids * other_days
  weekly_antacids * weeks_per_month

/-- Theorem stating that Nancy takes 60 antacids per month given her eating habits. -/
theorem nancy_antacid_consumption :
  antacids_per_month 3 2 1 3 2 7 4 = 60 := by
  sorry

#eval antacids_per_month 3 2 1 3 2 7 4

end nancy_antacid_consumption_l2344_234440


namespace album_collection_problem_l2344_234485

/-- The number of albums in either Andrew's or John's collection, but not both -/
def exclusive_albums (shared : ℕ) (andrew_total : ℕ) (john_exclusive : ℕ) : ℕ :=
  (andrew_total - shared) + john_exclusive

theorem album_collection_problem (shared : ℕ) (andrew_total : ℕ) (john_exclusive : ℕ)
  (h1 : shared = 12)
  (h2 : andrew_total = 20)
  (h3 : john_exclusive = 8) :
  exclusive_albums shared andrew_total john_exclusive = 16 := by
  sorry

end album_collection_problem_l2344_234485
