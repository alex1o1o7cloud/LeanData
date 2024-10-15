import Mathlib

namespace NUMINAMATH_CALUDE_unique_solution_condition_l1087_108702

theorem unique_solution_condition (a : ℝ) : 
  (∃! x : ℝ, x^2 - a * abs x + a^2 - 3 = 0) ↔ a = -Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l1087_108702


namespace NUMINAMATH_CALUDE_fourth_month_sales_l1087_108706

def sales_1 : ℕ := 5400
def sales_2 : ℕ := 9000
def sales_3 : ℕ := 6300
def sales_5 : ℕ := 4500
def sales_6 : ℕ := 1200
def average_sale : ℕ := 5600
def num_months : ℕ := 6

theorem fourth_month_sales :
  ∃ (sales_4 : ℕ), 
    (sales_1 + sales_2 + sales_3 + sales_4 + sales_5 + sales_6) / num_months = average_sale ∧
    sales_4 = 8200 := by
  sorry

end NUMINAMATH_CALUDE_fourth_month_sales_l1087_108706


namespace NUMINAMATH_CALUDE_job_completion_time_solution_l1087_108717

/-- Represents the time taken by three machines working together to complete a job -/
def job_completion_time (y : ℝ) : Prop :=
  let machine_a_time := y + 4
  let machine_b_time := y + 3
  let machine_c_time := 3 * y
  (1 / machine_a_time) + (1 / machine_b_time) + (1 / machine_c_time) = 1 / y

/-- Proves that the job completion time satisfies the given equation -/
theorem job_completion_time_solution :
  ∃ y : ℝ, job_completion_time y ∧ y = (-14 + Real.sqrt 296) / 10 := by
  sorry

end NUMINAMATH_CALUDE_job_completion_time_solution_l1087_108717


namespace NUMINAMATH_CALUDE_arrangements_count_l1087_108763

/-- Represents the number of volunteers -/
def num_volunteers : ℕ := 5

/-- Represents the number of venues -/
def num_venues : ℕ := 4

/-- Represents the condition that A is assigned to the badminton venue -/
def a_assigned_to_badminton : Prop := true

/-- Represents the condition that each volunteer goes to only one venue -/
def one_venue_per_volunteer : Prop := true

/-- Represents the condition that each venue has at least one volunteer -/
def at_least_one_volunteer_per_venue : Prop := true

/-- The total number of different arrangements -/
def total_arrangements : ℕ := 60

/-- Theorem stating that the number of arrangements is 60 -/
theorem arrangements_count :
  num_volunteers = 5 ∧
  num_venues = 4 ∧
  a_assigned_to_badminton ∧
  one_venue_per_volunteer ∧
  at_least_one_volunteer_per_venue →
  total_arrangements = 60 :=
by sorry

end NUMINAMATH_CALUDE_arrangements_count_l1087_108763


namespace NUMINAMATH_CALUDE_solve_for_x_l1087_108767

-- Define the € operation
def euro (x y : ℝ) : ℝ := 2 * x * y

-- State the theorem
theorem solve_for_x (x : ℝ) : euro x (euro 4 5) = 480 → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_x_l1087_108767


namespace NUMINAMATH_CALUDE_car_speed_problem_l1087_108724

/-- Given two cars starting from the same point and traveling in opposite directions,
    this theorem proves that if one car travels at 60 mph and after 4.66666666667 hours
    they are 490 miles apart, then the speed of the other car must be 45 mph. -/
theorem car_speed_problem (v : ℝ) : 
  (v * (14/3) + 60 * (14/3) = 490) → v = 45 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l1087_108724


namespace NUMINAMATH_CALUDE_proportional_function_quadrants_l1087_108742

/-- A function passes through the first and third quadrants if for any non-zero x,
    x and f(x) have the same sign. -/
def passes_through_first_and_third_quadrants (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ≠ 0 → (x > 0 ∧ f x > 0) ∨ (x < 0 ∧ f x < 0)

/-- Theorem: If the graph of y = kx passes through the first and third quadrants,
    then k is positive. -/
theorem proportional_function_quadrants (k : ℝ) :
  passes_through_first_and_third_quadrants (λ x => k * x) → k > 0 := by
  sorry

end NUMINAMATH_CALUDE_proportional_function_quadrants_l1087_108742


namespace NUMINAMATH_CALUDE_triangle_area_l1087_108726

theorem triangle_area (a b c : ℝ) (A B C : ℝ) : 
  c^2 = (a - b)^2 + 6 →
  C = π/3 →
  (1/2) * a * b * Real.sin C = (3 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1087_108726


namespace NUMINAMATH_CALUDE_square_sum_equals_sixteen_l1087_108773

theorem square_sum_equals_sixteen (x y : ℝ) 
  (h1 : (x + y)^2 = 36) 
  (h2 : x * y = 10) : 
  x^2 + y^2 = 16 := by
sorry

end NUMINAMATH_CALUDE_square_sum_equals_sixteen_l1087_108773


namespace NUMINAMATH_CALUDE_f_range_on_interval_l1087_108751

noncomputable def f (x : ℝ) : ℝ := (1/2) * Real.exp x * (Real.sin x + Real.cos x)

theorem f_range_on_interval :
  let a := 0
  let b := Real.pi / 2
  ∃ (min max : ℝ), 
    (∀ x ∈ Set.Icc a b, f x ≥ min ∧ f x ≤ max) ∧
    (∃ x₁ ∈ Set.Icc a b, f x₁ = min) ∧
    (∃ x₂ ∈ Set.Icc a b, f x₂ = max) ∧
    min = 1/2 ∧
    max = (1/2) * Real.exp (Real.pi / 2) :=
by sorry

end NUMINAMATH_CALUDE_f_range_on_interval_l1087_108751


namespace NUMINAMATH_CALUDE_largest_power_of_two_dividing_difference_of_fourth_powers_l1087_108770

theorem largest_power_of_two_dividing_difference_of_fourth_powers :
  ∃ k : ℕ, (2^k : ℕ) = 128 ∧ (2^k : ℕ) ∣ (17^4 - 15^4) ∧
  ∀ m : ℕ, 2^m ∣ (17^4 - 15^4) → m ≤ k :=
by sorry

end NUMINAMATH_CALUDE_largest_power_of_two_dividing_difference_of_fourth_powers_l1087_108770


namespace NUMINAMATH_CALUDE_part_one_part_two_l1087_108711

-- Define the sets P and Q
def P (a : ℝ) : Set ℝ := {x | a + 1 ≤ x ∧ x ≤ 2 * a + 1}
def Q : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}

-- Part 1
theorem part_one : (Set.univ \ P 3) ∩ Q = {x | -2 ≤ x ∧ x < 4} := by sorry

-- Part 2
theorem part_two : {a : ℝ | P a ⊂ Q ∧ P a ≠ ∅} = {a : ℝ | 0 ≤ a ∧ a ≤ 2} := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1087_108711


namespace NUMINAMATH_CALUDE_sum_of_digits_0_to_99_l1087_108785

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Sum of digits for a range of natural numbers -/
def sumOfDigitsRange (a b : ℕ) : ℕ := 
  (Finset.range (b - a + 1)).sum (fun i => sumOfDigits (a + i))

theorem sum_of_digits_0_to_99 :
  sumOfDigitsRange 0 99 = 900 :=
by
  sorry

/-- Given condition -/
axiom sum_of_digits_18_to_21 : sumOfDigitsRange 18 21 = 24

#check sum_of_digits_0_to_99
#check sum_of_digits_18_to_21

end NUMINAMATH_CALUDE_sum_of_digits_0_to_99_l1087_108785


namespace NUMINAMATH_CALUDE_greatest_integer_with_gcf_five_l1087_108735

def is_valid (n : ℕ) : Prop :=
  n < 150 ∧ Nat.gcd n 30 = 5

theorem greatest_integer_with_gcf_five : 
  (∀ m, is_valid m → m ≤ 145) ∧ is_valid 145 := by sorry

end NUMINAMATH_CALUDE_greatest_integer_with_gcf_five_l1087_108735


namespace NUMINAMATH_CALUDE_product_equals_eight_l1087_108794

theorem product_equals_eight :
  (1 + 1/1) * (1 + 1/2) * (1 + 1/3) * (1 + 1/4) * (1 + 1/5) * (1 + 1/6) * (1 + 1/7) = 8 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_eight_l1087_108794


namespace NUMINAMATH_CALUDE_fraction_sum_inequality_l1087_108745

theorem fraction_sum_inequality (a b c : ℝ) (h : a * b * c = 1) :
  (1 / (2 * a^2 + b^2 + 3)) + (1 / (2 * b^2 + c^2 + 3)) + (1 / (2 * c^2 + a^2 + 3)) ≤ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_inequality_l1087_108745


namespace NUMINAMATH_CALUDE_unique_solution_quartic_l1087_108734

theorem unique_solution_quartic (n : ℤ) : 
  (∃! x : ℝ, 4 * x^4 + n * x^2 + 4 = 0) ↔ (n = 8 ∨ n = -8) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_quartic_l1087_108734


namespace NUMINAMATH_CALUDE_sandwich_combinations_l1087_108723

/-- Represents the number of different types of bread available. -/
def num_breads : ℕ := 5

/-- Represents the number of different types of meat available. -/
def num_meats : ℕ := 7

/-- Represents the number of different types of cheese available. -/
def num_cheeses : ℕ := 6

/-- Represents the number of sandwiches with turkey and mozzarella combinations. -/
def turkey_mozzarella_combos : ℕ := num_breads

/-- Represents the number of sandwiches with rye bread and salami combinations. -/
def rye_salami_combos : ℕ := num_cheeses

/-- Represents the number of sandwiches with white bread and chicken combinations. -/
def white_chicken_combos : ℕ := num_cheeses

/-- Theorem stating the number of possible sandwich combinations. -/
theorem sandwich_combinations :
  num_breads * num_meats * num_cheeses - 
  (turkey_mozzarella_combos + rye_salami_combos + white_chicken_combos) = 193 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_combinations_l1087_108723


namespace NUMINAMATH_CALUDE_complement_of_union_equals_five_l1087_108779

-- Define the universal set U
def U : Finset Nat := {1, 2, 3, 4, 5}

-- Define set M
def M : Finset Nat := {1, 2}

-- Define set N
def N : Finset Nat := {3, 4}

-- Theorem statement
theorem complement_of_union_equals_five :
  (U \ (M ∪ N)) = {5} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_equals_five_l1087_108779


namespace NUMINAMATH_CALUDE_equilateral_triangle_exists_l1087_108788

-- Define the plane S parallel to x₁,₂ axis
structure Plane :=
  (s₁ : ℝ)
  (s₂ : ℝ)

-- Define a point in 3D space
structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

-- Define the trace lines of the plane
def traceLine1 (S : Plane) : Set Point3D :=
  {p : Point3D | p.y = S.s₁}

def traceLine2 (S : Plane) : Set Point3D :=
  {p : Point3D | p.z = S.s₂}

-- Define an equilateral triangle
structure EquilateralTriangle :=
  (A : Point3D)
  (B : Point3D)
  (C : Point3D)

-- State the theorem
theorem equilateral_triangle_exists (S : Plane) (A : Point3D) 
  (h : A.y = S.s₁ ∧ A.z = S.s₂) : 
  ∃ (t : EquilateralTriangle), 
    t.A = A ∧ 
    t.B ∈ traceLine1 S ∧ 
    t.C ∈ traceLine2 S ∧
    (t.A.x - t.B.x)^2 + (t.A.y - t.B.y)^2 + (t.A.z - t.B.z)^2 = 
    (t.B.x - t.C.x)^2 + (t.B.y - t.C.y)^2 + (t.B.z - t.C.z)^2 ∧
    (t.A.x - t.B.x)^2 + (t.A.y - t.B.y)^2 + (t.A.z - t.B.z)^2 = 
    (t.A.x - t.C.x)^2 + (t.A.y - t.C.y)^2 + (t.A.z - t.C.z)^2 := by
  sorry


end NUMINAMATH_CALUDE_equilateral_triangle_exists_l1087_108788


namespace NUMINAMATH_CALUDE_original_number_proof_l1087_108787

theorem original_number_proof (x : ℝ) : 
  x - 25 = 0.75 * x + 25 → x = 200 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l1087_108787


namespace NUMINAMATH_CALUDE_greatest_odd_integer_below_sqrt_50_l1087_108712

theorem greatest_odd_integer_below_sqrt_50 :
  ∀ x : ℕ, x % 2 = 1 → x^2 < 50 → x ≤ 7 :=
by sorry

end NUMINAMATH_CALUDE_greatest_odd_integer_below_sqrt_50_l1087_108712


namespace NUMINAMATH_CALUDE_real_square_properties_l1087_108747

theorem real_square_properties (a b : ℝ) : 
  (a^2 ≠ b^2 → a ≠ b) ∧ (a > |b| → a^2 > b^2) := by
  sorry

end NUMINAMATH_CALUDE_real_square_properties_l1087_108747


namespace NUMINAMATH_CALUDE_unique_element_condition_l1087_108701

def A (a : ℝ) : Set ℝ := {x | a * x^2 - 2 * x - 1 = 0}

theorem unique_element_condition (a : ℝ) : (∃! x, x ∈ A a) ↔ (a = 0 ∨ a = -1) := by
  sorry

end NUMINAMATH_CALUDE_unique_element_condition_l1087_108701


namespace NUMINAMATH_CALUDE_abs_sum_greater_than_abs_l1087_108700

theorem abs_sum_greater_than_abs (a b c : ℝ) 
  (h1 : a < b) (h2 : b < c) 
  (h3 : a * b + b * c + a * c = 0) 
  (h4 : a * b * c = 1) : 
  |a + b| > |c| := by
sorry

end NUMINAMATH_CALUDE_abs_sum_greater_than_abs_l1087_108700


namespace NUMINAMATH_CALUDE_fibonacci_sum_l1087_108795

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Sum of F_n / 10^n from n = 0 to infinity -/
noncomputable def fibSum : ℝ := ∑' n, (fib n : ℝ) / (10 : ℝ) ^ n

/-- Theorem: The sum of F_n / 10^n from n = 0 to infinity equals 10/89 -/
theorem fibonacci_sum : fibSum = 10 / 89 := by sorry

end NUMINAMATH_CALUDE_fibonacci_sum_l1087_108795


namespace NUMINAMATH_CALUDE_sin_range_on_interval_l1087_108736

theorem sin_range_on_interval :
  let f : ℝ → ℝ := λ x ↦ Real.sin x
  let S : Set ℝ := { x | -π/4 ≤ x ∧ x ≤ 3*π/4 }
  f '' S = { y | -Real.sqrt 2 / 2 ≤ y ∧ y ≤ 1 } := by
  sorry

end NUMINAMATH_CALUDE_sin_range_on_interval_l1087_108736


namespace NUMINAMATH_CALUDE_max_notebooks_purchase_l1087_108749

theorem max_notebooks_purchase (notebook_price : ℕ) (available_money : ℚ) : 
  notebook_price = 45 → available_money = 40.5 → 
  ∃ max_notebooks : ℕ, max_notebooks = 90 ∧ 
  (max_notebooks : ℚ) * (notebook_price : ℚ) / 100 ≤ available_money ∧
  ∀ n : ℕ, (n : ℚ) * (notebook_price : ℚ) / 100 ≤ available_money → n ≤ max_notebooks :=
by sorry

end NUMINAMATH_CALUDE_max_notebooks_purchase_l1087_108749


namespace NUMINAMATH_CALUDE_total_hotdogs_sold_l1087_108783

theorem total_hotdogs_sold (small_hotdogs large_hotdogs : ℕ) 
  (h1 : small_hotdogs = 58) 
  (h2 : large_hotdogs = 21) : 
  small_hotdogs + large_hotdogs = 79 := by
  sorry

end NUMINAMATH_CALUDE_total_hotdogs_sold_l1087_108783


namespace NUMINAMATH_CALUDE_intersection_A_B_union_complements_l1087_108753

-- Define the sets A and B
def A : Set ℝ := {x | x < -4 ∨ x > 1}
def B : Set ℝ := {x | -3 ≤ x - 1 ∧ x - 1 ≤ 2}

-- Theorem for A ∩ B
theorem intersection_A_B : A ∩ B = {x | 1 < x ∧ x ≤ 3} := by sorry

-- Theorem for (C_U A) ∪ (C_U B)
theorem union_complements : (Set.univ \ A) ∪ (Set.univ \ B) = {x | x ≤ 1 ∨ x > 3} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_union_complements_l1087_108753


namespace NUMINAMATH_CALUDE_candy_shop_ratio_l1087_108705

/-- Proves that the ratio of cherry sours to lemon sours is 4:5 given the conditions of the candy shop problem -/
theorem candy_shop_ratio :
  ∀ (total cherry orange lemon : ℕ),
  total = 96 →
  cherry = 32 →
  orange = total / 4 →
  total = cherry + orange + lemon →
  (cherry : ℚ) / lemon = 4 / 5 := by
sorry

end NUMINAMATH_CALUDE_candy_shop_ratio_l1087_108705


namespace NUMINAMATH_CALUDE_unique_remainder_mod_nine_l1087_108758

theorem unique_remainder_mod_nine : 
  ∃! n : ℤ, 0 ≤ n ∧ n < 9 ∧ -1111 ≡ n [ZMOD 9] := by
  sorry

end NUMINAMATH_CALUDE_unique_remainder_mod_nine_l1087_108758


namespace NUMINAMATH_CALUDE_circular_plate_arrangement_l1087_108781

def arrangement_count (blue red green yellow : ℕ) : ℕ :=
  sorry

theorem circular_plate_arrangement :
  arrangement_count 6 3 2 1 = 22680 :=
sorry

end NUMINAMATH_CALUDE_circular_plate_arrangement_l1087_108781


namespace NUMINAMATH_CALUDE_intersection_A_B_l1087_108762

def f (x : ℝ) : ℝ := x^2 - 12*x + 36

def A : Set ℕ := {a | 1 ≤ a ∧ a ≤ 10}

def B : Set ℕ := {b | ∃ a ∈ A, f a = b}

theorem intersection_A_B : A ∩ B = {1, 4, 9} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l1087_108762


namespace NUMINAMATH_CALUDE_eight_power_ten_sum_equals_two_power_y_l1087_108704

theorem eight_power_ten_sum_equals_two_power_y (y : ℕ) :
  8^10 + 8^10 + 8^10 + 8^10 + 8^10 + 8^10 + 8^10 + 8^10 = 2^y → y = 33 := by
  sorry

end NUMINAMATH_CALUDE_eight_power_ten_sum_equals_two_power_y_l1087_108704


namespace NUMINAMATH_CALUDE_count_is_thirty_l1087_108731

/-- 
Counts the number of non-negative integers n less than 120 for which 
there exists an integer m divisible by 4 such that the roots of 
x^2 - nx + m = 0 are consecutive non-negative integers.
-/
def count_valid_n : ℕ := by
  sorry

/-- The main theorem stating that the count is equal to 30 -/
theorem count_is_thirty : count_valid_n = 30 := by
  sorry

end NUMINAMATH_CALUDE_count_is_thirty_l1087_108731


namespace NUMINAMATH_CALUDE_weight_10_moles_CaH2_l1087_108737

/-- The molecular weight of CaH2 in g/mol -/
def molecular_weight_CaH2 : ℝ := 40.08 + 2 * 1.008

/-- The total weight of a given number of moles of CaH2 in grams -/
def total_weight_CaH2 (moles : ℝ) : ℝ := moles * molecular_weight_CaH2

/-- Theorem stating that 10 moles of CaH2 weigh 420.96 grams -/
theorem weight_10_moles_CaH2 : total_weight_CaH2 10 = 420.96 := by sorry

end NUMINAMATH_CALUDE_weight_10_moles_CaH2_l1087_108737


namespace NUMINAMATH_CALUDE_min_value_theorem_l1087_108732

theorem min_value_theorem (x y : ℝ) (h : x^2 * y^2 + y^4 = 1) :
  ∃ (m : ℝ), m = 2 * Real.sqrt 2 ∧ ∀ (z w : ℝ), z^2 * w^2 + w^4 = 1 → x^2 + 3 * y^2 ≤ z^2 + 3 * w^2 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1087_108732


namespace NUMINAMATH_CALUDE_sin_cos_difference_equals_negative_half_l1087_108756

theorem sin_cos_difference_equals_negative_half :
  Real.sin (119 * π / 180) * Real.cos (91 * π / 180) - 
  Real.sin (91 * π / 180) * Real.sin (29 * π / 180) = -1/2 :=
by sorry

end NUMINAMATH_CALUDE_sin_cos_difference_equals_negative_half_l1087_108756


namespace NUMINAMATH_CALUDE_jellybean_theorem_l1087_108748

/-- The number of jellybeans each person has -/
structure JellyBeans where
  arnold : ℕ
  lee : ℕ
  tino : ℕ
  joshua : ℕ

/-- The conditions of the jellybean distribution -/
def jellybean_conditions (j : JellyBeans) : Prop :=
  j.arnold = 5 ∧
  j.lee = 2 * j.arnold ∧
  j.tino = j.lee + 24 ∧
  j.joshua = 3 * j.arnold

/-- The theorem to prove -/
theorem jellybean_theorem (j : JellyBeans) 
  (h : jellybean_conditions j) : 
  j.tino = 34 ∧ j.arnold + j.lee + j.tino + j.joshua = 64 := by
  sorry

end NUMINAMATH_CALUDE_jellybean_theorem_l1087_108748


namespace NUMINAMATH_CALUDE_no_solution_condition_l1087_108784

theorem no_solution_condition (a : ℝ) : 
  (∀ x : ℝ, x ≠ 2 → (1 / (x - 2) + a / (2 - x) ≠ 2 * a)) ↔ (a = 0 ∨ a = 1) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_condition_l1087_108784


namespace NUMINAMATH_CALUDE_total_tickets_sold_l1087_108789

/-- Proves that the total number of tickets sold is 350 --/
theorem total_tickets_sold (orchestra_price balcony_price : ℕ)
  (total_cost : ℕ) (balcony_excess : ℕ) :
  orchestra_price = 12 →
  balcony_price = 8 →
  total_cost = 3320 →
  balcony_excess = 90 →
  ∃ (orchestra_tickets balcony_tickets : ℕ),
    orchestra_tickets * orchestra_price + balcony_tickets * balcony_price = total_cost ∧
    balcony_tickets = orchestra_tickets + balcony_excess ∧
    orchestra_tickets + balcony_tickets = 350 :=
by sorry

end NUMINAMATH_CALUDE_total_tickets_sold_l1087_108789


namespace NUMINAMATH_CALUDE_grid_sum_theorem_l1087_108750

/-- A 3x3 grid represented as a function from (Fin 3 × Fin 3) to ℕ -/
def Grid := Fin 3 → Fin 3 → ℕ

/-- The sum of numbers on the main diagonal of the grid -/
def mainDiagonalSum (g : Grid) : ℕ :=
  g 0 0 + g 1 1 + g 2 2

/-- The sum of numbers on the other diagonal of the grid -/
def otherDiagonalSum (g : Grid) : ℕ :=
  g 0 2 + g 1 1 + g 2 0

/-- The sum of numbers not on either diagonal -/
def nonDiagonalSum (g : Grid) : ℕ :=
  g 0 1 + g 1 0 + g 1 2 + g 2 1 + g 1 1

/-- The theorem statement -/
theorem grid_sum_theorem (g : Grid) :
  (∀ i j, g i j ∈ Finset.range 10) →
  (mainDiagonalSum g = 7) →
  (otherDiagonalSum g = 21) →
  (nonDiagonalSum g = 25) := by
  sorry

end NUMINAMATH_CALUDE_grid_sum_theorem_l1087_108750


namespace NUMINAMATH_CALUDE_cubic_roots_sum_product_l1087_108741

theorem cubic_roots_sum_product (α β γ : ℂ) (u v w : ℂ) : 
  (∀ x : ℂ, x^3 + 5*x^2 + 7*x - 13 = (x - α) * (x - β) * (x - γ)) →
  (∀ x : ℂ, x^3 + u*x^2 + v*x + w = (x - (α + β)) * (x - (β + γ)) * (x - (γ + α))) →
  w = 48 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_product_l1087_108741


namespace NUMINAMATH_CALUDE_photo_arrangement_l1087_108752

theorem photo_arrangement (n_male : ℕ) (n_female : ℕ) : 
  n_male = 4 → n_female = 2 → (
    (3 : ℕ) *           -- ways to place "甲" in middle positions
    (4 : ℕ).factorial * -- ways to arrange remaining units
    (2 : ℕ).factorial   -- ways to arrange female students within their unit
  ) = 144 := by
  sorry

end NUMINAMATH_CALUDE_photo_arrangement_l1087_108752


namespace NUMINAMATH_CALUDE_largest_k_for_inequality_l1087_108757

theorem largest_k_for_inequality (a b c : ℝ) 
  (h1 : a ≤ b) (h2 : b ≤ c) 
  (h3 : a * b + b * c + c * a = 0) 
  (h4 : a * b * c = 1) :
  (∀ k : ℝ, (∀ a b c : ℝ, a ≤ b → b ≤ c → a * b + b * c + c * a = 0 → a * b * c = 1 → 
    |a + b| ≥ k * |c|) → k ≤ 4) ∧
  (∀ a b c : ℝ, a ≤ b → b ≤ c → a * b + b * c + c * a = 0 → a * b * c = 1 → 
    |a + b| ≥ 4 * |c|) :=
by sorry

end NUMINAMATH_CALUDE_largest_k_for_inequality_l1087_108757


namespace NUMINAMATH_CALUDE_earth_habitable_fraction_l1087_108780

theorem earth_habitable_fraction :
  (earth_land_fraction : ℚ) →
  (land_habitable_fraction : ℚ) →
  earth_land_fraction = 1/3 →
  land_habitable_fraction = 1/4 →
  earth_land_fraction * land_habitable_fraction = 1/12 :=
by sorry

end NUMINAMATH_CALUDE_earth_habitable_fraction_l1087_108780


namespace NUMINAMATH_CALUDE_inequality_proof_l1087_108796

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : c < 1) : (a - b) * (c - 1) < 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1087_108796


namespace NUMINAMATH_CALUDE_modulus_of_complex_quotient_l1087_108786

theorem modulus_of_complex_quotient : 
  ∀ (z₁ z₂ : ℂ), 
    z₁ = Complex.mk 0 2 → 
    z₂ = Complex.mk 1 (-1) → 
    Complex.abs (z₁ / z₂) = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_modulus_of_complex_quotient_l1087_108786


namespace NUMINAMATH_CALUDE_shorter_segment_length_l1087_108716

-- Define the triangle ABC
def Triangle (a b c : ℝ) := a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

-- Define the angle bisector
def AngleBisector (a b c ae ec : ℝ) := ae / ec = a / b

theorem shorter_segment_length 
  (a b c : ℝ) 
  (h_triangle : Triangle a b c)
  (h_ratio : ∃ (k : ℝ), a = 3*k ∧ b = 4*k ∧ c = 5*k)
  (h_ab_length : c = 24)
  (ae ec : ℝ)
  (h_bisector : AngleBisector a b c ae ec)
  (h_sum : ae + ec = c)
  (h_ae_shorter : ae ≤ ec) :
  ae = 72/7 :=
sorry

end NUMINAMATH_CALUDE_shorter_segment_length_l1087_108716


namespace NUMINAMATH_CALUDE_animal_count_l1087_108777

theorem animal_count (num_cats : ℕ) : 
  (1 : ℕ) +                   -- 1 dog
  num_cats +                  -- cats
  2 * num_cats +              -- rabbits (2 per cat)
  3 * (2 * num_cats) = 37 →   -- hares (3 per rabbit)
  num_cats = 4 := by
sorry

end NUMINAMATH_CALUDE_animal_count_l1087_108777


namespace NUMINAMATH_CALUDE_at_op_difference_l1087_108710

-- Define the @ operation
def at_op (x y : ℤ) : ℤ := x * y - 3 * x + y

-- Theorem statement
theorem at_op_difference : (at_op 5 6) - (at_op 6 5) = 4 := by
  sorry

end NUMINAMATH_CALUDE_at_op_difference_l1087_108710


namespace NUMINAMATH_CALUDE_friends_total_distance_l1087_108774

/-- Represents the distance walked by each friend -/
structure FriendDistances where
  lionel : ℕ  -- miles
  esther : ℕ  -- yards
  niklaus : ℕ  -- feet

/-- Converts miles to feet -/
def milesToFeet (miles : ℕ) : ℕ := miles * 5280

/-- Converts yards to feet -/
def yardsToFeet (yards : ℕ) : ℕ := yards * 3

/-- Calculates the total distance walked by all friends in feet -/
def totalDistanceInFeet (distances : FriendDistances) : ℕ :=
  milesToFeet distances.lionel + yardsToFeet distances.esther + distances.niklaus

/-- Theorem stating that the total distance walked by the friends is 26332 feet -/
theorem friends_total_distance (distances : FriendDistances) 
  (h1 : distances.lionel = 4)
  (h2 : distances.esther = 975)
  (h3 : distances.niklaus = 1287) :
  totalDistanceInFeet distances = 26332 := by
  sorry


end NUMINAMATH_CALUDE_friends_total_distance_l1087_108774


namespace NUMINAMATH_CALUDE_rotation_result_l1087_108791

-- Define the shapes
inductive Shape
  | Triangle
  | SmallCircle
  | Square
  | InvertedTriangle

-- Define the initial configuration
def initial_config : List Shape :=
  [Shape.Triangle, Shape.SmallCircle, Shape.Square, Shape.InvertedTriangle]

-- Define the rotation function
def rotate (angle : ℕ) (config : List Shape) : List Shape :=
  let shift := angle / 30  -- 150° = 5 * 30°
  config.rotateLeft shift

-- Theorem statement
theorem rotation_result :
  rotate 150 initial_config = [Shape.Square, Shape.InvertedTriangle, Shape.Triangle, Shape.SmallCircle] :=
by sorry

end NUMINAMATH_CALUDE_rotation_result_l1087_108791


namespace NUMINAMATH_CALUDE_divisor_power_difference_l1087_108782

theorem divisor_power_difference (k : ℕ) : 
  (15 ^ k ∣ 823435) → 5 ^ k - k ^ 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_divisor_power_difference_l1087_108782


namespace NUMINAMATH_CALUDE_floor_plus_twice_eq_33_l1087_108727

theorem floor_plus_twice_eq_33 :
  ∃! x : ℝ, (⌊x⌋ : ℝ) + 2 * x = 33 :=
by sorry

end NUMINAMATH_CALUDE_floor_plus_twice_eq_33_l1087_108727


namespace NUMINAMATH_CALUDE_equation_solution_l1087_108775

theorem equation_solution (y : ℝ) : 
  y = (13/2)^4 ↔ 3 * y^(1/4) - (3 * y^(1/2)) / y^(1/4) = 13 - 2 * y^(1/4) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1087_108775


namespace NUMINAMATH_CALUDE_boys_to_girls_ratio_l1087_108778

/-- Proves that the ratio of boys to girls in a school with 90 students, of which 60 are girls, is 1:2 -/
theorem boys_to_girls_ratio (total_students : Nat) (girls : Nat) (h1 : total_students = 90) (h2 : girls = 60) :
  (total_students - girls) / girls = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_boys_to_girls_ratio_l1087_108778


namespace NUMINAMATH_CALUDE_quadratic_function_determination_l1087_108743

open Real

/-- Given real numbers a, b, c, and functions f and g,
    if the maximum value of g(x) is 2 when -1 ≤ x ≤ 1,
    then f(x) = 2x^2 - 1 -/
theorem quadratic_function_determination
  (a b c : ℝ)
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
  (h_f : ∀ x, f x = a * x^2 + b * x + c)
  (h_g : ∀ x, g x = a * x + b)
  (h_max : ∀ x ∈ Set.Icc (-1) 1, g x ≤ 2)
  (h_reaches_max : ∃ x ∈ Set.Icc (-1) 1, g x = 2) :
  ∀ x, f x = 2 * x^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_determination_l1087_108743


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_sum_l1087_108798

theorem partial_fraction_decomposition_sum (p q r A B C : ℝ) : 
  (p ≠ q ∧ q ≠ r ∧ p ≠ r) →
  (∀ (x : ℝ), x ≠ p ∧ x ≠ q ∧ x ≠ r → 
    1 / (x^3 - 15*x^2 + 50*x - 56) = A / (x - p) + B / (x - q) + C / (x - r)) →
  (x^3 - 15*x^2 + 50*x - 56 = (x - p) * (x - q) * (x - r)) →
  1 / A + 1 / B + 1 / C = 225 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_sum_l1087_108798


namespace NUMINAMATH_CALUDE_product_base_8_units_digit_l1087_108793

def base_10_to_8_units_digit (n : ℕ) : ℕ :=
  n % 8

theorem product_base_8_units_digit :
  base_10_to_8_units_digit (348 * 27) = 4 := by
sorry

end NUMINAMATH_CALUDE_product_base_8_units_digit_l1087_108793


namespace NUMINAMATH_CALUDE_product_49_sum_0_l1087_108739

theorem product_49_sum_0 (a b c d : ℤ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d → 
  a * b * c * d = 49 → 
  a + b + c + d = 0 := by
sorry

end NUMINAMATH_CALUDE_product_49_sum_0_l1087_108739


namespace NUMINAMATH_CALUDE_problem_statement_l1087_108799

theorem problem_statement (a b : ℝ) (h1 : a * b = -3) (h2 : a + b = 2) :
  a^2 * b + a * b^2 = -6 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1087_108799


namespace NUMINAMATH_CALUDE_unique_permutation_with_difference_one_l1087_108776

theorem unique_permutation_with_difference_one (n : ℕ+) :
  ∃! (x : Fin (2 * n) → Fin (2 * n)), 
    Function.Bijective x ∧ 
    (∀ i : Fin (2 * n), |x i - i.val| = 1) := by
  sorry

end NUMINAMATH_CALUDE_unique_permutation_with_difference_one_l1087_108776


namespace NUMINAMATH_CALUDE_law_school_students_l1087_108718

/-- The number of students in the business school -/
def business_students : ℕ := 500

/-- The number of sibling pairs -/
def sibling_pairs : ℕ := 30

/-- The probability of selecting a sibling pair -/
def sibling_pair_probability : ℚ := 7500000000000001 / 100000000000000000

/-- Theorem stating the number of law students -/
theorem law_school_students (L : ℕ) : 
  (sibling_pairs : ℚ) / (business_students * L) = sibling_pair_probability → 
  L = 8000 := by
  sorry

end NUMINAMATH_CALUDE_law_school_students_l1087_108718


namespace NUMINAMATH_CALUDE_square_difference_l1087_108740

theorem square_difference (x y : ℝ) (h1 : (x + y)^2 = 81) (h2 : x * y = 6) :
  (x - y)^2 = 57 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l1087_108740


namespace NUMINAMATH_CALUDE_four_tire_repair_cost_l1087_108730

/-- The total cost for repairing a given number of tires -/
def total_cost (repair_cost : ℚ) (sales_tax : ℚ) (num_tires : ℕ) : ℚ :=
  (repair_cost + sales_tax) * num_tires

/-- Theorem: The total cost for repairing 4 tires is $30 -/
theorem four_tire_repair_cost :
  total_cost 7 0.5 4 = 30 := by
  sorry

end NUMINAMATH_CALUDE_four_tire_repair_cost_l1087_108730


namespace NUMINAMATH_CALUDE_greatest_divisible_by_cubes_l1087_108761

theorem greatest_divisible_by_cubes : ∃ (n : ℕ), n = 60 ∧ 
  (∀ (m : ℕ), m^3 ≤ n → n % m = 0) ∧
  (∀ (k : ℕ), k > n → ∃ (m : ℕ), m^3 ≤ k ∧ k % m ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_greatest_divisible_by_cubes_l1087_108761


namespace NUMINAMATH_CALUDE_solve_equation_l1087_108765

theorem solve_equation (X : ℝ) : 
  (X^3).sqrt = 81 * (81^(1/12)) → X = 3^(14/9) := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1087_108765


namespace NUMINAMATH_CALUDE_rectangle_area_perimeter_sum_l1087_108764

theorem rectangle_area_perimeter_sum (a b : ℕ+) :
  ∃ (x y : ℕ+), (x + 2) * (y + 2) - 2 = 114 ∧
  ∃ (x y : ℕ+), (x + 2) * (y + 2) - 2 = 116 ∧
  ∃ (x y : ℕ+), (x + 2) * (y + 2) - 2 = 120 ∧
  ∃ (x y : ℕ+), (x + 2) * (y + 2) - 2 = 122 ∧
  ¬∃ (x y : ℕ+), (x + 2) * (y + 2) - 2 = 118 :=
by sorry


end NUMINAMATH_CALUDE_rectangle_area_perimeter_sum_l1087_108764


namespace NUMINAMATH_CALUDE_bucket_capacities_solution_l1087_108728

/-- Represents the capacities of three buckets A, B, and C. -/
structure BucketCapacities where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if the given capacities satisfy the problem conditions. -/
def satisfiesConditions (caps : BucketCapacities) : Prop :=
  caps.a + caps.b + caps.c = 1440 ∧
  caps.a + (1/5) * caps.b = caps.c ∧
  caps.b + (1/3) * caps.a = caps.c

/-- Theorem stating that the unique solution satisfying the conditions is (480, 400, 560). -/
theorem bucket_capacities_solution :
  ∃! (caps : BucketCapacities), satisfiesConditions caps ∧ 
    caps.a = 480 ∧ caps.b = 400 ∧ caps.c = 560 := by
  sorry

end NUMINAMATH_CALUDE_bucket_capacities_solution_l1087_108728


namespace NUMINAMATH_CALUDE_sum_of_solutions_eq_sixteen_l1087_108715

theorem sum_of_solutions_eq_sixteen : 
  ∃ (x₁ x₂ : ℝ), (x₁ - 8)^2 = 16 ∧ (x₂ - 8)^2 = 16 ∧ x₁ + x₂ = 16 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_eq_sixteen_l1087_108715


namespace NUMINAMATH_CALUDE_log_inequality_l1087_108714

theorem log_inequality (x y a b : ℝ) 
  (hx : 0 < x) (hy : x < y) (hy1 : y < 1) 
  (hb : 1 < b) (ha : b < a) : 
  Real.log x / b < Real.log y / a := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l1087_108714


namespace NUMINAMATH_CALUDE_weight_of_aluminum_oxide_l1087_108738

/-- The atomic weight of aluminum in g/mol -/
def atomic_weight_Al : ℝ := 26.98

/-- The atomic weight of oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The number of aluminum atoms in one molecule of aluminum oxide -/
def Al_count : ℕ := 2

/-- The number of oxygen atoms in one molecule of aluminum oxide -/
def O_count : ℕ := 3

/-- The number of moles of aluminum oxide -/
def moles_Al2O3 : ℝ := 5

/-- The molecular weight of aluminum oxide in g/mol -/
def molecular_weight_Al2O3 : ℝ := Al_count * atomic_weight_Al + O_count * atomic_weight_O

/-- The total weight of the given amount of aluminum oxide in grams -/
def total_weight_Al2O3 : ℝ := moles_Al2O3 * molecular_weight_Al2O3

theorem weight_of_aluminum_oxide :
  total_weight_Al2O3 = 509.8 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_aluminum_oxide_l1087_108738


namespace NUMINAMATH_CALUDE_removed_number_is_34_l1087_108744

/-- Given n consecutive natural numbers starting from 1, if one number x is removed
    and the average of the remaining numbers is 152/7, then x = 34. -/
theorem removed_number_is_34 (n : ℕ) (x : ℕ) :
  (x ≥ 1 ∧ x ≤ n) →
  (n * (n + 1) / 2 - x) / (n - 1) = 152 / 7 →
  x = 34 := by
  sorry

end NUMINAMATH_CALUDE_removed_number_is_34_l1087_108744


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l1087_108746

theorem inequality_and_equality_condition (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x + y^2 / x ≥ 2 * y ∧ (x + y^2 / x = 2 * y ↔ x = y) := by
  sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l1087_108746


namespace NUMINAMATH_CALUDE_unique_square_divisible_by_three_in_range_l1087_108703

theorem unique_square_divisible_by_three_in_range : ∃! y : ℕ, 
  (∃ x : ℕ, y = x^2) ∧ 
  (∃ k : ℕ, y = 3 * k) ∧ 
  50 < y ∧ y < 120 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_square_divisible_by_three_in_range_l1087_108703


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_and_product_l1087_108768

theorem quadratic_roots_sum_and_product (α β : ℝ) : 
  α ≠ β →
  α^2 - 5*α - 2 = 0 →
  β^2 - 5*β - 2 = 0 →
  α + β + α*β = 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_and_product_l1087_108768


namespace NUMINAMATH_CALUDE_no_odd_integer_solution_l1087_108790

theorem no_odd_integer_solution (n : ℕ+) (x y z : ℤ) 
  (hx : Odd x) (hy : Odd y) (hz : Odd z) : 
  (x + y)^n.val + (y + z)^n.val ≠ (x + z)^n.val := by
  sorry

end NUMINAMATH_CALUDE_no_odd_integer_solution_l1087_108790


namespace NUMINAMATH_CALUDE_equation_proof_l1087_108719

theorem equation_proof : 529 - 2 * 23 * 8 + 64 = 225 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l1087_108719


namespace NUMINAMATH_CALUDE_sum_first_10_even_integers_l1087_108771

/-- The sum of the first n positive even integers -/
def sum_first_n_even_integers (n : ℕ) : ℕ :=
  2 * n * (n + 1)

/-- Theorem: The sum of the first 10 positive even integers is 110 -/
theorem sum_first_10_even_integers :
  sum_first_n_even_integers 10 = 110 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_10_even_integers_l1087_108771


namespace NUMINAMATH_CALUDE_triangle_to_hexagon_area_ratio_l1087_108797

/-- A regular hexagon with an inscribed equilateral triangle -/
structure RegularHexagonWithTriangle where
  -- The area of the regular hexagon
  hexagon_area : ℝ
  -- The area of the inscribed equilateral triangle
  triangle_area : ℝ

/-- The ratio of the inscribed triangle's area to the hexagon's area is 1/6 -/
theorem triangle_to_hexagon_area_ratio 
  (hex : RegularHexagonWithTriangle) : 
  hex.triangle_area / hex.hexagon_area = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_to_hexagon_area_ratio_l1087_108797


namespace NUMINAMATH_CALUDE_cylinder_side_surface_diagonal_l1087_108772

/-- Given a cylinder with height 8 feet and base perimeter 6 feet,
    prove that the diagonal of the rectangular plate forming its side surface is 10 feet. -/
theorem cylinder_side_surface_diagonal (h : ℝ) (p : ℝ) (d : ℝ) :
  h = 8 →
  p = 6 →
  d = (h^2 + p^2)^(1/2) →
  d = 10 :=
by sorry

end NUMINAMATH_CALUDE_cylinder_side_surface_diagonal_l1087_108772


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l1087_108755

theorem polynomial_division_remainder (k : ℝ) : 
  (∀ x : ℝ, ∃ q : ℝ, 3 * x^3 - k * x^2 + 4 = (3 * x - 1) * q + 5) → k = -8 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l1087_108755


namespace NUMINAMATH_CALUDE_boarding_students_count_l1087_108769

theorem boarding_students_count (x : ℕ) (students : ℕ) : 
  (students = 4 * x + 10) →  -- If each dormitory houses 4 people with 10 left over
  (6 * (x - 1) + 1 ≤ students) →  -- Lower bound when housing 6 per dormitory
  (students ≤ 6 * (x - 1) + 5) →  -- Upper bound when housing 6 per dormitory
  (students = 34 ∨ students = 38) :=
by sorry

end NUMINAMATH_CALUDE_boarding_students_count_l1087_108769


namespace NUMINAMATH_CALUDE_social_media_ratio_l1087_108766

/-- Represents the daily phone usage in hours -/
def daily_phone_usage : ℝ := 16

/-- Represents the weekly social media usage in hours -/
def weekly_social_media_usage : ℝ := 56

/-- Represents the number of days in a week -/
def days_in_week : ℝ := 7

/-- Theorem: The ratio of daily time spent on social media to total daily time spent on phone is 1:2 -/
theorem social_media_ratio : 
  (weekly_social_media_usage / days_in_week) / daily_phone_usage = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_social_media_ratio_l1087_108766


namespace NUMINAMATH_CALUDE_gcd_12345_54321_l1087_108707

theorem gcd_12345_54321 : Nat.gcd 12345 54321 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_12345_54321_l1087_108707


namespace NUMINAMATH_CALUDE_sin_cos_sum_bound_l1087_108720

theorem sin_cos_sum_bound (θ : Real) (h1 : π/2 < θ) (h2 : θ < π) (h3 : Real.sin (θ/2) < Real.cos (θ/2)) :
  -Real.sqrt 2 < Real.sin (θ/2) + Real.cos (θ/2) ∧ Real.sin (θ/2) + Real.cos (θ/2) < -1 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_bound_l1087_108720


namespace NUMINAMATH_CALUDE_negative_fraction_comparison_l1087_108733

theorem negative_fraction_comparison : -3/5 < -1/5 := by
  sorry

end NUMINAMATH_CALUDE_negative_fraction_comparison_l1087_108733


namespace NUMINAMATH_CALUDE_parabola_equation_l1087_108722

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  equation : ℝ → ℝ → Prop
  focus : ℝ × ℝ
  h_positive : p > 0
  h_equation : ∀ x y, equation x y ↔ x^2 = 2*p*y
  h_focus : focus = (0, p/2)

/-- Theorem: If there exists a point M on parabola C such that |OM| = |MF| = 3,
    then the equation of parabola C is x^2 = 8y -/
theorem parabola_equation (C : Parabola) :
  (∃ M : ℝ × ℝ, C.equation M.1 M.2 ∧ 
    Real.sqrt (M.1^2 + M.2^2) = 3 ∧
    Real.sqrt ((M.1 - C.focus.1)^2 + (M.2 - C.focus.2)^2) = 3) →
  C.p = 4 ∧ ∀ x y, C.equation x y ↔ x^2 = 8*y :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_l1087_108722


namespace NUMINAMATH_CALUDE_f_min_value_existence_l1087_108708

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < a then -a * x + 3 else (x - 3) * Real.exp x + Real.exp 2

theorem f_min_value_existence (a : ℝ) :
  (∃ (m : ℝ), ∀ (x : ℝ), f a x ≥ m) →
  (∃ (a' : ℝ), 0 ≤ a' ∧ a' ≤ Real.sqrt 3 ∧
    ∀ (x : ℝ), f a' x ≥ f a' (2 : ℝ)) ∧
  (∀ (a' : ℝ), a' > Real.sqrt 3 →
    ¬∃ (m : ℝ), ∀ (x : ℝ), f a' x ≥ m) :=
sorry

end NUMINAMATH_CALUDE_f_min_value_existence_l1087_108708


namespace NUMINAMATH_CALUDE_correct_practice_times_l1087_108721

/-- Represents the practice schedule and time spent on instruments in a month -/
structure PracticeSchedule where
  piano_daily_minutes : ℕ
  violin_daily_minutes : ℕ
  flute_daily_minutes : ℕ
  violin_days_per_week : ℕ
  flute_days_per_week : ℕ
  weeks_with_6_days : ℕ
  weeks_with_7_days : ℕ

/-- Calculates the total practice time for each instrument in the given month -/
def calculate_practice_time (schedule : PracticeSchedule) :
  ℕ × ℕ × ℕ :=
  let total_days := schedule.weeks_with_6_days * 6 + schedule.weeks_with_7_days * 7
  let violin_total_days := schedule.violin_days_per_week * (schedule.weeks_with_6_days + schedule.weeks_with_7_days)
  let flute_total_days := schedule.flute_days_per_week * (schedule.weeks_with_6_days + schedule.weeks_with_7_days)
  (schedule.piano_daily_minutes * total_days,
   schedule.violin_daily_minutes * violin_total_days,
   schedule.flute_daily_minutes * flute_total_days)

/-- Theorem stating the correct practice times for each instrument -/
theorem correct_practice_times (schedule : PracticeSchedule)
  (h1 : schedule.piano_daily_minutes = 25)
  (h2 : schedule.violin_daily_minutes = 3 * schedule.piano_daily_minutes)
  (h3 : schedule.flute_daily_minutes = schedule.violin_daily_minutes / 2)
  (h4 : schedule.violin_days_per_week = 5)
  (h5 : schedule.flute_days_per_week = 4)
  (h6 : schedule.weeks_with_6_days = 2)
  (h7 : schedule.weeks_with_7_days = 2) :
  calculate_practice_time schedule = (650, 1500, 600) :=
sorry


end NUMINAMATH_CALUDE_correct_practice_times_l1087_108721


namespace NUMINAMATH_CALUDE_essay_writing_rate_l1087_108759

/-- Proves that the writing rate for the first two hours must be 400 words per hour 
    given the conditions of the essay writing problem. -/
theorem essay_writing_rate (total_words : ℕ) (total_hours : ℕ) (later_rate : ℕ) 
    (h1 : total_words = 1200)
    (h2 : total_hours = 4)
    (h3 : later_rate = 200) : 
  ∃ (initial_rate : ℕ), 
    initial_rate * 2 + later_rate * (total_hours - 2) = total_words ∧ 
    initial_rate = 400 := by
  sorry

end NUMINAMATH_CALUDE_essay_writing_rate_l1087_108759


namespace NUMINAMATH_CALUDE_orientation_count_equals_product_of_combinations_l1087_108725

/-- The number of ways to orient 40 unit segments for zero sum --/
def orientationCount : ℕ := sorry

/-- The total number of unit segments --/
def totalSegments : ℕ := 40

/-- The number of horizontal (or vertical) segments --/
def segmentsPerDirection : ℕ := 20

/-- The number of segments that need to be positive in each direction for zero sum --/
def positiveSegmentsPerDirection : ℕ := 10

theorem orientation_count_equals_product_of_combinations : 
  orientationCount = Nat.choose segmentsPerDirection positiveSegmentsPerDirection * 
                     Nat.choose segmentsPerDirection positiveSegmentsPerDirection := by sorry

end NUMINAMATH_CALUDE_orientation_count_equals_product_of_combinations_l1087_108725


namespace NUMINAMATH_CALUDE_examination_statements_l1087_108729

/-- Represents a statistical population -/
structure Population where
  size : ℕ

/-- Represents a sample from a population -/
structure Sample (pop : Population) where
  size : ℕ
  h_size_le : size ≤ pop.size

/-- The given examination scenario -/
def examination_scenario : Prop :=
  ∃ (pop : Population) (sample : Sample pop),
    pop.size = 70000 ∧
    sample.size = 1000 ∧
    (sample.size = 1000 → sample.size = 1000) ∧
    (pop.size = 70000 → pop.size = 70000)

/-- The statements to be proved -/
theorem examination_statements (h : examination_scenario) :
  ∃ (pop : Population) (sample : Sample pop),
    pop.size = 70000 ∧
    sample.size = 1000 ∧
    (Sample pop → True) ∧  -- Statement 1
    (pop.size = 70000 → True) ∧  -- Statement 3
    (sample.size = 1000 → True)  -- Statement 4
    := by sorry

end NUMINAMATH_CALUDE_examination_statements_l1087_108729


namespace NUMINAMATH_CALUDE_min_value_fraction_l1087_108754

theorem min_value_fraction (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y - 3 = 0) :
  (x + 2 * y) / (x * y) ≥ 3 ∧ ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ 2 * x₀ + y₀ - 3 = 0 ∧ (x₀ + 2 * y₀) / (x₀ * y₀) = 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_fraction_l1087_108754


namespace NUMINAMATH_CALUDE_pi_over_three_irrational_l1087_108713

theorem pi_over_three_irrational : Irrational (π / 3) :=
by
  sorry

end NUMINAMATH_CALUDE_pi_over_three_irrational_l1087_108713


namespace NUMINAMATH_CALUDE_cubic_factorization_l1087_108760

theorem cubic_factorization (x : ℝ) : x^3 - x = x * (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l1087_108760


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l1087_108709

/-- Given that x and y are inversely proportional, prove that y = -56.25 when x = -12,
    given that x = 3y when x + y = 60 -/
theorem inverse_proportion_problem (x y : ℝ) (k : ℝ) : 
  (∀ x' y', x' * y' = k) →  -- x and y are inversely proportional
  (∃ x₀ y₀, x₀ = 3 * y₀ ∧ x₀ + y₀ = 60) →  -- when their sum is 60, x is three times y
  (x = -12 → y = -56.25) :=  -- y = -56.25 when x = -12
by sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l1087_108709


namespace NUMINAMATH_CALUDE_convergence_implies_cluster_sets_l1087_108792

open Set
open Filter
open Topology

/-- A sequence converges to a limit -/
def SequenceConvergesTo (x : ℕ → ℝ) (a : ℝ) :=
  Tendsto x atTop (𝓝 a)

/-- An interval is a cluster set for a sequence if it contains infinitely many terms of the sequence -/
def IsClusterSet (x : ℕ → ℝ) (s : Set ℝ) :=
  ∀ n : ℕ, ∃ m ≥ n, x m ∈ s

theorem convergence_implies_cluster_sets (x : ℕ → ℝ) (a : ℝ) :
  SequenceConvergesTo x a →
  (∀ ε > 0, IsClusterSet x (Ioo (a - ε) (a + ε))) ∧
  (∀ s : Set ℝ, IsOpen s → a ∉ s → ¬IsClusterSet x s) :=
sorry

end NUMINAMATH_CALUDE_convergence_implies_cluster_sets_l1087_108792
