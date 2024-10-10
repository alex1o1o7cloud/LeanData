import Mathlib

namespace lara_chips_count_l3415_341545

theorem lara_chips_count :
  ∀ (total_chips : ℕ),
  (total_chips / 6 : ℚ) + 34 + 16 = total_chips →
  total_chips = 60 := by
sorry

end lara_chips_count_l3415_341545


namespace ashutosh_completion_time_l3415_341571

theorem ashutosh_completion_time 
  (suresh_completion_time : ℝ) 
  (suresh_work_time : ℝ) 
  (ashutosh_remaining_time : ℝ) 
  (h1 : suresh_completion_time = 15)
  (h2 : suresh_work_time = 9)
  (h3 : ashutosh_remaining_time = 8)
  : ∃ (ashutosh_alone_time : ℝ), ashutosh_alone_time = 20 := by
  sorry

end ashutosh_completion_time_l3415_341571


namespace divisibility_condition_l3415_341513

theorem divisibility_condition (n : ℕ+) :
  (∃ m : ℤ, (2^n.val - 1) ∣ (m^2 + 9)) ↔ ∃ x : ℕ, n = 2^x :=
sorry

end divisibility_condition_l3415_341513


namespace carlotta_performance_length_l3415_341533

/-- Represents the length of Carlotta's final stage performance in minutes -/
def performance_length : ℝ := 6

/-- For every minute of singing, Carlotta spends 3 minutes practicing -/
def practice_ratio : ℝ := 3

/-- For every minute of singing, Carlotta spends 5 minutes throwing tantrums -/
def tantrum_ratio : ℝ := 5

/-- The total combined time of singing, practicing, and throwing tantrums in minutes -/
def total_time : ℝ := 54

theorem carlotta_performance_length :
  performance_length * (1 + practice_ratio + tantrum_ratio) = total_time :=
sorry

end carlotta_performance_length_l3415_341533


namespace characterization_of_k_l3415_341515

/-- The greatest odd divisor of a natural number -/
def greatestOddDivisor (m : ℕ) : ℕ := sorry

/-- The property that n does not divide the greatest odd divisor of k^n + 1 -/
def noDivide (k n : ℕ) : Prop :=
  ¬(n ∣ greatestOddDivisor ((k^n + 1) : ℕ))

/-- The main theorem -/
theorem characterization_of_k (k : ℕ) (h : k ≥ 2) :
  (∃ l : ℕ, l ≥ 2 ∧ k = 2^l - 1) ↔ (∀ n : ℕ, n ≥ 2 → noDivide k n) := by
  sorry

end characterization_of_k_l3415_341515


namespace function_property_l3415_341548

/-- Given a function f(x) = ax² - bx where a and b are positive constants,
    if f(f(1)) = -1 and √(ab) = 3, then a = 1 or a = 2. -/
theorem function_property (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  let f : ℝ → ℝ := λ x => a * x^2 - b * x
  (f (f 1) = -1) ∧ (Real.sqrt (a * b) = 3) → a = 1 ∨ a = 2 := by
  sorry

end function_property_l3415_341548


namespace pi_estimation_l3415_341598

theorem pi_estimation (total_points : ℕ) (obtuse_points : ℕ) : 
  total_points = 120 → obtuse_points = 34 → 
  (obtuse_points : ℝ) / (total_points : ℝ) = π / 4 - 1 / 2 → 
  π = 47 / 15 := by
sorry

end pi_estimation_l3415_341598


namespace coefficient_x4_in_expansion_l3415_341514

theorem coefficient_x4_in_expansion (x : ℝ) : 
  ∃ (a b c d e f : ℝ), (2*x + 1) * (x - 1)^5 = a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + f ∧ b = 15 :=
sorry

end coefficient_x4_in_expansion_l3415_341514


namespace document_word_count_l3415_341593

/-- Given Barbara's typing speeds and time, calculate the number of words in the document -/
theorem document_word_count 
  (original_speed : ℕ) 
  (speed_reduction : ℕ) 
  (typing_time : ℕ) 
  (h1 : original_speed = 212)
  (h2 : speed_reduction = 40)
  (h3 : typing_time = 20) : 
  (original_speed - speed_reduction) * typing_time = 3440 :=
by sorry

end document_word_count_l3415_341593


namespace isabella_hair_growth_l3415_341516

/-- Calculates hair growth given initial and final hair lengths -/
def hair_growth (initial_length final_length : ℝ) : ℝ :=
  final_length - initial_length

theorem isabella_hair_growth :
  let initial_length : ℝ := 18
  let final_length : ℝ := 24
  hair_growth initial_length final_length = 6 := by
  sorry

end isabella_hair_growth_l3415_341516


namespace min_value_f_l3415_341540

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 4*x + (2 - a)*Real.log x

-- Define the derivative of f
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 2*x - 4 - (2 - a)/x

-- Theorem statement
theorem min_value_f (a : ℝ) :
  ∃ (min_val : ℝ), ∀ x ∈ Set.Icc (Real.exp 1) (Real.exp 2), f a x ≥ min_val ∧
  (min_val = f a (Real.exp 1) ∨
   min_val = f a (Real.exp 2) ∨
   (∃ y ∈ Set.Ioo (Real.exp 1) (Real.exp 2), min_val = f a y ∧ f_deriv a y = 0)) :=
sorry

end

end min_value_f_l3415_341540


namespace arithmetic_sequence_ratio_l3415_341581

theorem arithmetic_sequence_ratio (a b d₁ d₂ : ℝ) : 
  (a + 4 * d₁ = b) → (a + 5 * d₂ = b) → d₁ / d₂ = 5 / 4 := by
  sorry

end arithmetic_sequence_ratio_l3415_341581


namespace tripled_base_and_exponent_l3415_341502

theorem tripled_base_and_exponent (a b x : ℝ) (ha : a > 0) (hb : b > 0) (hx : x > 0) :
  (3 * a) ^ (3 * b) = a ^ b * x ^ b → x = 27 * a ^ 2 := by
  sorry

end tripled_base_and_exponent_l3415_341502


namespace remaining_money_proof_l3415_341522

def salary : ℚ := 190000

def food_fraction : ℚ := 1/5
def rent_fraction : ℚ := 1/10
def clothes_fraction : ℚ := 3/5

def remaining_amount : ℚ := salary * (1 - (food_fraction + rent_fraction + clothes_fraction))

theorem remaining_money_proof :
  remaining_amount = 19000 := by sorry

end remaining_money_proof_l3415_341522


namespace perfect_square_trinomial_condition_l3415_341590

/-- A perfect square trinomial in x and y -/
def isPerfectSquareTrinomial (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (k : ℝ), ∀ x y, f x y = (x + k*y)^2 ∨ f x y = (x - k*y)^2

/-- The theorem stating that if x^2 + axy + y^2 is a perfect square trinomial, then a = 2 or a = -2 -/
theorem perfect_square_trinomial_condition (a : ℝ) :
  isPerfectSquareTrinomial (fun x y => x^2 + a*x*y + y^2) → a = 2 ∨ a = -2 := by
  sorry


end perfect_square_trinomial_condition_l3415_341590


namespace range_of_a_l3415_341566

-- Define the propositions p and q
def p (x : ℝ) : Prop := 1/2 ≤ x ∧ x ≤ 1
def q (x a : ℝ) : Prop := (x - a) * (x - a - 1) > 0

-- Define the sufficient but not necessary condition
def sufficient_not_necessary (a : ℝ) : Prop :=
  (∀ x, p x → q x a) ∧ ∃ x, ¬p x ∧ q x a

-- State the theorem
theorem range_of_a : 
  ∀ a : ℝ, sufficient_not_necessary a ↔ 0 ≤ a ∧ a ≤ 1/2 :=
sorry

end range_of_a_l3415_341566


namespace first_concert_attendance_l3415_341564

theorem first_concert_attendance (second_concert : ℕ) (difference : ℕ) : 
  second_concert = 66018 → difference = 119 → second_concert - difference = 65899 := by
  sorry

end first_concert_attendance_l3415_341564


namespace distance_proof_l3415_341509

/-- Proves that the distance between two points is 2 km given specific travel conditions -/
theorem distance_proof (T : ℝ) : 
  (4 * (T + 7/60) = 8 * (T - 8/60)) → 
  (4 * (T + 7/60) = 2) := by
  sorry

end distance_proof_l3415_341509


namespace max_real_part_sum_l3415_341544

theorem max_real_part_sum (z : Fin 18 → ℂ) (w : Fin 18 → ℂ) : 
  (∀ j : Fin 18, z j ^ 18 = (2 : ℂ) ^ 54) →
  (∀ j : Fin 18, w j = z j ∨ w j = Complex.I * z j ∨ w j = -z j) →
  (∃ w_choice : Fin 18 → ℂ, 
    (∀ j : Fin 18, w_choice j = z j ∨ w_choice j = Complex.I * z j ∨ w_choice j = -z j) ∧
    (Finset.sum Finset.univ (λ j => (w_choice j).re) = 
      8 + 8 * (2 * (1 + Real.sqrt 3 + Real.sqrt 2 + 
        Real.cos (π / 9) + Real.cos (2 * π / 9) + Real.cos (4 * π / 9) + 
        Real.cos (5 * π / 9) + Real.cos (7 * π / 9) + Real.cos (8 * π / 9))))) ∧
  (∀ w_alt : Fin 18 → ℂ, 
    (∀ j : Fin 18, w_alt j = z j ∨ w_alt j = Complex.I * z j ∨ w_alt j = -z j) →
    Finset.sum Finset.univ (λ j => (w_alt j).re) ≤ 
      8 + 8 * (2 * (1 + Real.sqrt 3 + Real.sqrt 2 + 
        Real.cos (π / 9) + Real.cos (2 * π / 9) + Real.cos (4 * π / 9) + 
        Real.cos (5 * π / 9) + Real.cos (7 * π / 9) + Real.cos (8 * π / 9)))) := by
  sorry

end max_real_part_sum_l3415_341544


namespace fraction_equality_l3415_341520

theorem fraction_equality (a b c d : ℝ) 
  (h : (a - b) * (c - d) / ((b - c) * (d - a)) = 3 / 4) : 
  (a - c) * (b - d) / ((a - b) * (c - d)) = 1 := by sorry

end fraction_equality_l3415_341520


namespace jogger_train_distance_l3415_341518

/-- Calculates the distance a jogger is ahead of a train given their speeds and the time it takes for the train to pass the jogger. -/
theorem jogger_train_distance (jogger_speed : ℝ) (train_speed : ℝ) (train_length : ℝ) (passing_time : ℝ) : 
  jogger_speed = 10 * (5/18) → 
  train_speed = 46 * (5/18) → 
  train_length = 120 → 
  passing_time = 46 → 
  (train_speed - jogger_speed) * passing_time - train_length = 340 := by
  sorry

#check jogger_train_distance

end jogger_train_distance_l3415_341518


namespace existence_of_prime_1021_n_l3415_341583

theorem existence_of_prime_1021_n : ∃ n : ℕ, n ≥ 3 ∧ Nat.Prime (n^3 + 2*n + 1) := by
  sorry

end existence_of_prime_1021_n_l3415_341583


namespace function_coefficient_sum_l3415_341578

/-- Given a function f : ℝ → ℝ satisfying certain conditions, prove that a + b + c = 3 -/
theorem function_coefficient_sum (f : ℝ → ℝ) (a b c : ℝ) :
  (∀ x, f (x + 3) = 4 * x^2 + 9 * x + 5) →
  (∀ x, f x = a * x^2 + b * x + c) →
  a + b + c = 3 := by
sorry

end function_coefficient_sum_l3415_341578


namespace abs_five_minus_sqrt_two_l3415_341530

theorem abs_five_minus_sqrt_two : |5 - Real.sqrt 2| = 5 - Real.sqrt 2 := by
  sorry

end abs_five_minus_sqrt_two_l3415_341530


namespace employee_count_sum_l3415_341547

theorem employee_count_sum : 
  (Finset.sum (Finset.filter (fun s => 200 ≤ s ∧ s ≤ 300 ∧ (s - 1) % 7 = 0) (Finset.range 301)) id) = 3493 :=
by sorry

end employee_count_sum_l3415_341547


namespace inequality_solution_set_l3415_341574

theorem inequality_solution_set (x : ℝ) :
  (x^2 - 2*x - 3) * (x^2 + 1) < 0 ↔ -1 < x ∧ x < 3 := by sorry

end inequality_solution_set_l3415_341574


namespace intersection_of_sets_l3415_341573

theorem intersection_of_sets : 
  let M : Set ℤ := {0, 1, 2, 3}
  let P : Set ℤ := {-1, 1, -2, 2}
  M ∩ P = {1, 2} := by
  sorry

end intersection_of_sets_l3415_341573


namespace largest_class_size_l3415_341517

/-- Proves that in a school with 5 classes, where each class has 2 students less than the previous class,
    and the total number of students is 115, the largest class has 27 students. -/
theorem largest_class_size (total_students : ℕ) (num_classes : ℕ) (diff : ℕ) :
  total_students = 115 →
  num_classes = 5 →
  diff = 2 →
  ∃ (x : ℕ), x = 27 ∧ 
    (x + (x - diff) + (x - 2*diff) + (x - 3*diff) + (x - 4*diff) = total_students) :=
by sorry

end largest_class_size_l3415_341517


namespace badminton_medals_count_l3415_341577

/-- Proves that the number of badminton medals is 5 --/
theorem badminton_medals_count :
  ∀ (total_medals track_medals swimming_medals badminton_medals : ℕ),
  total_medals = 20 →
  track_medals = 5 →
  swimming_medals = 2 * track_medals →
  badminton_medals = total_medals - (track_medals + swimming_medals) →
  badminton_medals = 5 := by
  sorry

end badminton_medals_count_l3415_341577


namespace job_selection_probability_l3415_341567

theorem job_selection_probability 
  (carol_prob : ℚ) 
  (bernie_prob : ℚ) 
  (h1 : carol_prob = 4 / 5) 
  (h2 : bernie_prob = 3 / 5) : 
  carol_prob * bernie_prob = 12 / 25 := by
  sorry

end job_selection_probability_l3415_341567


namespace javiers_dogs_l3415_341557

theorem javiers_dogs (total_legs : ℕ) (human_count : ℕ) (human_legs : ℕ) (dog_legs : ℕ) :
  total_legs = 22 →
  human_count = 5 →
  human_legs = 2 →
  dog_legs = 4 →
  (human_count * human_legs + (total_legs - human_count * human_legs) / dog_legs : ℕ) = 3 := by
  sorry

end javiers_dogs_l3415_341557


namespace trapezoid_point_distance_l3415_341563

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a line in 2D space -/
structure Line :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)

/-- Represents a trapezoid ABCD -/
structure Trapezoid :=
  (A : Point)
  (B : Point)
  (C : Point)
  (D : Point)

/-- Returns the intersection point of two lines -/
def intersectionPoint (l1 l2 : Line) : Point :=
  sorry

/-- Returns the line passing through two points -/
def lineThroughPoints (p1 p2 : Point) : Line :=
  sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ :=
  sorry

/-- Recursively defines points A_n and B_n -/
def definePoints (trap : Trapezoid) (E : Point) (n : ℕ) : Point × Point :=
  match n with
  | 0 => (trap.A, trap.B)
  | n+1 =>
    let (A_n, _) := definePoints trap E n
    let B_next := intersectionPoint (lineThroughPoints A_n trap.C) (lineThroughPoints trap.B trap.D)
    let A_next := intersectionPoint (lineThroughPoints E B_next) (lineThroughPoints trap.A trap.B)
    (A_next, B_next)

/-- The main theorem to be proved -/
theorem trapezoid_point_distance (trap : Trapezoid) (E : Point) (n : ℕ) :
  let (A_n, _) := definePoints trap E n
  distance A_n trap.B = distance trap.A trap.B / (n + 1) :=
sorry

end trapezoid_point_distance_l3415_341563


namespace system_solution_l3415_341596

theorem system_solution : 
  ∀ (x y z t : ℕ), 
    x + y = z * t ∧ z + t = x * y → 
      ((x = 1 ∧ y = 5 ∧ z = 2 ∧ t = 3) ∨ 
       (x = 5 ∧ y = 1 ∧ z = 3 ∧ t = 2) ∨ 
       (x = 2 ∧ y = 2 ∧ z = 2 ∧ t = 2)) := by
  sorry

#check system_solution

end system_solution_l3415_341596


namespace instantaneous_velocity_at_1_2_l3415_341587

/-- Equation of motion for an object -/
def s (t : ℝ) : ℝ := 2 * (1 - t^2)

/-- Instantaneous velocity at time t -/
def v (t : ℝ) : ℝ := -4 * t

theorem instantaneous_velocity_at_1_2 : v 1.2 = -4.8 := by
  sorry

end instantaneous_velocity_at_1_2_l3415_341587


namespace haley_zoo_pictures_l3415_341588

/-- The number of pictures Haley took at the zoo -/
def zoo_pictures : ℕ := sorry

/-- The total number of pictures Haley took before deleting any -/
def total_pictures : ℕ := zoo_pictures + 8

/-- The number of pictures Haley had after deleting some -/
def remaining_pictures : ℕ := total_pictures - 38

theorem haley_zoo_pictures :
  zoo_pictures = 50 ∧ remaining_pictures = 20 :=
sorry

end haley_zoo_pictures_l3415_341588


namespace sum_of_roots_quadratic_l3415_341507

theorem sum_of_roots_quadratic (b : ℝ) (x₁ x₂ : ℝ) : 
  x₁^2 - 2*x₁ + b = 0 → x₂^2 - 2*x₂ + b = 0 → x₁ + x₂ = 2 := by
  sorry

end sum_of_roots_quadratic_l3415_341507


namespace unique_a_value_l3415_341560

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x : ℝ | a - 1 < x ∧ x < a + 1}
def B : Set ℝ := {x : ℝ | x^2 - 4*x + 3 ≥ 0}

-- Define the proposition p and q
def p (a : ℝ) : Prop := ∃ x, x ∈ A a
def q : Prop := ∃ x, x ∈ B

-- Define the negation of q
def not_q : Prop := ∃ x, x ∉ B

-- Theorem statement
theorem unique_a_value : 
  ∃! a : ℝ, (∀ x : ℝ, not_q → p a) ∧ a = 2 := by sorry

end unique_a_value_l3415_341560


namespace polynomial_factor_implies_coefficients_l3415_341506

theorem polynomial_factor_implies_coefficients 
  (p q : ℝ) 
  (h : ∃ (a b c : ℝ), px^4 + qx^3 + 40*x^2 - 24*x + 9 = (4*x^2 - 3*x + 2) * (a*x^2 + b*x + c)) :
  p = 12.5 ∧ q = -30.375 := by
sorry

end polynomial_factor_implies_coefficients_l3415_341506


namespace problem_solution_l3415_341576

theorem problem_solution (t : ℚ) (x y : ℚ) 
  (h1 : x = 3 - 2 * t) 
  (h2 : y = 5 * t + 9) 
  (h3 : x = 0) : 
  y = 33 / 2 := by
  sorry

end problem_solution_l3415_341576


namespace quadratic_equation_roots_l3415_341519

theorem quadratic_equation_roots (m : ℝ) : 
  (∃ x : ℝ, x^2 - 6*x + m = 0 ∧ x = 2) → 
  (∃ y : ℝ, y^2 - 6*y + m = 0 ∧ y = 4) := by
sorry

end quadratic_equation_roots_l3415_341519


namespace square_to_rectangle_perimeter_l3415_341549

theorem square_to_rectangle_perimeter (n : ℕ) (a : ℝ) : 
  a > 0 →
  n > 0 →
  ∃ k : ℕ, k > 0 ∧ k < n ∧
  (k : ℝ) * 6 * a = (n - 2 * k : ℝ) * 4 * a ∧
  4 * n * a - (4 * n * a - 40) = 40 →
  4 * n * a = 280 :=
by sorry

end square_to_rectangle_perimeter_l3415_341549


namespace necessary_and_sufficient_condition_l3415_341510

theorem necessary_and_sufficient_condition (a : ℝ) :
  let f := fun x => x * (x - a) * (x - 2)
  let f' := fun x => 3 * x^2 - 2 * (a + 2) * x + 2 * a
  (0 < a ∧ a < 2) ↔ f' a < 0 := by
  sorry

end necessary_and_sufficient_condition_l3415_341510


namespace mouse_seeds_count_l3415_341534

/-- Represents the number of seeds per burrow for the mouse -/
def mouse_seeds_per_burrow : ℕ := 4

/-- Represents the number of seeds per burrow for the rabbit -/
def rabbit_seeds_per_burrow : ℕ := 7

/-- Represents the difference in number of burrows between mouse and rabbit -/
def burrow_difference : ℕ := 3

theorem mouse_seeds_count (mouse_burrows rabbit_burrows : ℕ) 
  (h1 : mouse_burrows = rabbit_burrows + burrow_difference)
  (h2 : mouse_seeds_per_burrow * mouse_burrows = rabbit_seeds_per_burrow * rabbit_burrows) :
  mouse_seeds_per_burrow * mouse_burrows = 28 := by
  sorry

end mouse_seeds_count_l3415_341534


namespace numerator_proof_l3415_341595

theorem numerator_proof (x y : ℝ) (h1 : x / y = 7 / 3) :
  ∃ (k N : ℝ), x = 7 * k ∧ y = 3 * k ∧ N / (x - y) = 2.5 ∧ N = 10 * k := by
  sorry

end numerator_proof_l3415_341595


namespace car_speeds_l3415_341504

theorem car_speeds (distance : ℝ) (time_difference : ℝ) (arrival_difference : ℝ) 
  (speed_ratio_small : ℝ) (speed_ratio_large : ℝ) 
  (h1 : distance = 135)
  (h2 : time_difference = 4)
  (h3 : arrival_difference = 1/2)
  (h4 : speed_ratio_small = 5)
  (h5 : speed_ratio_large = 2) :
  ∃ (speed_small : ℝ) (speed_large : ℝ),
    speed_small = 45 ∧ 
    speed_large = 18 ∧
    speed_small / speed_large = speed_ratio_small / speed_ratio_large ∧
    distance / speed_small = distance / speed_large - time_difference - arrival_difference :=
by
  sorry

end car_speeds_l3415_341504


namespace sqrt_x_minus_one_defined_l3415_341538

theorem sqrt_x_minus_one_defined (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 1) ↔ x ≥ 1 := by sorry

end sqrt_x_minus_one_defined_l3415_341538


namespace no_solutions_absolute_value_equation_l3415_341550

theorem no_solutions_absolute_value_equation :
  ¬ ∃ x : ℝ, |x - 2| = |x - 1| + |x - 5| := by
sorry

end no_solutions_absolute_value_equation_l3415_341550


namespace square_difference_minus_sum_squares_product_l3415_341565

theorem square_difference_minus_sum_squares_product (a b : ℝ) :
  (a - b)^2 - (b^2 + a^2 - 2*a*b) = 0 := by
  sorry

end square_difference_minus_sum_squares_product_l3415_341565


namespace poly_expansions_general_poly_expansion_possible_m_values_l3415_341511

-- Define the polynomial expressions
def poly1 (x : ℝ) := (x + 2) * (x + 3)
def poly2 (x : ℝ) := (x + 2) * (x - 3)
def poly3 (x : ℝ) := (x - 2) * (x + 3)
def poly4 (x : ℝ) := (x - 2) * (x - 3)

-- Define the general polynomial expression
def polyGeneral (x a b : ℝ) := (x + a) * (x + b)

-- Theorem statements
theorem poly_expansions :
  (∀ x : ℝ, poly1 x = x^2 + 5*x + 6) ∧
  (∀ x : ℝ, poly2 x = x^2 - x - 6) ∧
  (∀ x : ℝ, poly3 x = x^2 + x - 6) ∧
  (∀ x : ℝ, poly4 x = x^2 - 5*x + 6) :=
sorry

theorem general_poly_expansion :
  ∀ x a b : ℝ, polyGeneral x a b = x^2 + (a + b)*x + a*b :=
sorry

theorem possible_m_values :
  ∀ a b m : ℤ, (∀ x : ℝ, polyGeneral x (a : ℝ) (b : ℝ) = x^2 + m*x + 5) →
  (m = 6 ∨ m = -6) :=
sorry

end poly_expansions_general_poly_expansion_possible_m_values_l3415_341511


namespace johns_allowance_l3415_341526

theorem johns_allowance (allowance : ℝ) : 
  (allowance > 0) →
  (2 / 3 * (2 / 5 * allowance) = 1.28) →
  allowance = 4.80 := by
sorry

end johns_allowance_l3415_341526


namespace arrangement_count_is_36_l3415_341570

/-- The number of ways to arrange 5 students in a row with specific conditions -/
def arrangement_count : ℕ :=
  let n : ℕ := 5  -- Total number of students
  let special_pair : ℕ := 2  -- Number of students that must be adjacent (A and B)
  let non_end_student : ℕ := 1  -- Number of students that can't be at the ends (A)
  -- The actual count calculation would go here
  36

/-- Theorem stating that the number of arrangements under given conditions is 36 -/
theorem arrangement_count_is_36 : arrangement_count = 36 := by
  sorry

end arrangement_count_is_36_l3415_341570


namespace cube_triangle_areas_sum_l3415_341524

/-- Represents a 2x2x2 cube -/
structure Cube :=
  (side_length : ℝ)
  (is_2x2x2 : side_length = 2)

/-- Represents a triangle with vertices from the cube -/
structure CubeTriangle :=
  (vertices : Fin 3 → Fin 8)

/-- The area of a triangle given its side lengths -/
noncomputable def triangle_area (a b c : ℝ) : ℝ := sorry

/-- The sum of areas of all triangles in the cube -/
noncomputable def sum_of_triangle_areas (cube : Cube) : ℝ := sorry

/-- The main theorem -/
theorem cube_triangle_areas_sum (cube : Cube) :
  ∃ (m n p : ℕ), 
    sum_of_triangle_areas cube = m + Real.sqrt n + Real.sqrt p ∧
    m + n + p = 121 := by sorry

end cube_triangle_areas_sum_l3415_341524


namespace unique_number_with_three_prime_factors_l3415_341551

theorem unique_number_with_three_prime_factors (x n : ℕ) : 
  x = 7^n + 1 →
  Odd n →
  (∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ p ≠ 11 ∧ q ≠ 11 ∧ x = 2 * 11 * p * q) →
  x = 16808 :=
by sorry

end unique_number_with_three_prime_factors_l3415_341551


namespace correct_num_cats_l3415_341542

/-- Represents the number of cats on the ship -/
def num_cats : ℕ := 5

/-- Represents the number of sailors on the ship -/
def num_sailors : ℕ := 14 - num_cats

/-- The total number of heads on the ship -/
def total_heads : ℕ := 16

/-- The total number of legs on the ship -/
def total_legs : ℕ := 41

/-- Theorem stating that the number of cats is correct given the conditions -/
theorem correct_num_cats : 
  num_cats + num_sailors + 2 = total_heads ∧ 
  4 * num_cats + 2 * num_sailors + 3 = total_legs :=
by sorry

end correct_num_cats_l3415_341542


namespace cos_m_eq_sin_318_l3415_341580

theorem cos_m_eq_sin_318 (m : ℤ) (h1 : -180 ≤ m) (h2 : m ≤ 180) (h3 : Real.cos (m * π / 180) = Real.sin (318 * π / 180)) :
  m = 132 ∨ m = -132 := by
sorry

end cos_m_eq_sin_318_l3415_341580


namespace average_candy_count_l3415_341527

def candy_counts : List Nat := [5, 7, 9, 12, 12, 15, 15, 18, 25]

theorem average_candy_count (num_bags : Nat) (counts : List Nat) 
  (h1 : num_bags = 9)
  (h2 : counts = candy_counts)
  (h3 : counts.length = num_bags) :
  Int.floor ((counts.sum : ℝ) / num_bags + 0.5) = 13 :=
by sorry

end average_candy_count_l3415_341527


namespace paityn_blue_hats_l3415_341546

theorem paityn_blue_hats (paityn_red : ℕ) (paityn_blue : ℕ) (zola_red : ℕ) (zola_blue : ℕ) 
  (h1 : paityn_red = 20)
  (h2 : zola_red = (4 : ℕ) * paityn_red / 5)
  (h3 : zola_blue = 2 * paityn_blue)
  (h4 : paityn_red + paityn_blue + zola_red + zola_blue = 2 * 54) :
  paityn_blue = 24 := by
  sorry

end paityn_blue_hats_l3415_341546


namespace doris_earnings_l3415_341525

/-- Calculates the number of weeks needed to earn a target amount given an hourly rate and weekly work schedule. -/
def weeks_to_earn (hourly_rate : ℕ) (weekday_hours : ℕ) (saturday_hours : ℕ) (target_amount : ℕ) : ℕ :=
  let weekly_hours := weekday_hours * 5 + saturday_hours
  let weekly_earnings := hourly_rate * weekly_hours
  (target_amount + weekly_earnings - 1) / weekly_earnings

/-- Theorem stating that Doris needs 3 weeks to earn at least $1200 given her work schedule. -/
theorem doris_earnings : weeks_to_earn 20 3 5 1200 = 3 := by
  sorry

end doris_earnings_l3415_341525


namespace repeating_decimal_equals_fraction_l3415_341532

/-- The repeating decimal 0.363636... expressed as a real number -/
def repeating_decimal : ℚ := 0.363636

/-- Theorem stating that the repeating decimal 0.363636... is equal to 4/11 -/
theorem repeating_decimal_equals_fraction : repeating_decimal = 4 / 11 := by
  sorry

end repeating_decimal_equals_fraction_l3415_341532


namespace tangent_circle_radius_l3415_341599

/-- Represents a right triangle with angles 30°, 60°, and 90° -/
structure Triangle30_60_90 where
  shortSide : ℝ
  longSide : ℝ
  hypotenuse : ℝ
  angle30 : Real
  angle60 : Real
  angle90 : Real

/-- Represents a circle tangent to coordinate axes and triangle hypotenuse -/
structure TangentCircle where
  center : ℝ × ℝ
  radius : ℝ

/-- Given a 30-60-90 triangle with shortest side 2, 
    the radius of a circle tangent to coordinate axes and hypotenuse is 1 + 2√3 -/
theorem tangent_circle_radius 
  (t : Triangle30_60_90) 
  (c : TangentCircle) 
  (h1 : t.shortSide = 2) 
  (h2 : c.center.1 > 0 ∧ c.center.2 > 0) 
  (h3 : c.radius = c.center.1 ∧ c.radius = c.center.2) 
  (h4 : ∃ (x y : ℝ), x^2 + y^2 = t.hypotenuse^2 ∧ 
                     (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2) :
  c.radius = 1 + 2 * Real.sqrt 3 := by
  sorry

end tangent_circle_radius_l3415_341599


namespace arithmetic_sequence_length_l3415_341559

-- Define the arithmetic sequence
def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

-- Theorem statement
theorem arithmetic_sequence_length :
  ∃ n : ℕ, n > 0 ∧ arithmetic_sequence 220 (-5) n = 35 ∧ n = 38 := by
  sorry

end arithmetic_sequence_length_l3415_341559


namespace train_speed_l3415_341508

/-- The speed of a train given its length and time to cross a fixed point. -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 280) (h2 : time = 20) :
  length / time = 14 := by
  sorry

end train_speed_l3415_341508


namespace number_of_girls_in_school_l3415_341539

/-- Represents the number of students in a section -/
def SectionSize : ℕ := 24

/-- Represents the total number of boys in the school -/
def TotalBoys : ℕ := 408

/-- Represents the total number of sections -/
def TotalSections : ℕ := 26

/-- Represents the number of sections for boys -/
def BoySections : ℕ := 17

/-- Represents the number of sections for girls -/
def GirlSections : ℕ := 9

/-- Theorem stating the number of girls in the school -/
theorem number_of_girls_in_school : 
  TotalBoys / BoySections = SectionSize ∧ 
  BoySections + GirlSections = TotalSections → 
  GirlSections * SectionSize = 216 :=
by sorry

end number_of_girls_in_school_l3415_341539


namespace quadratic_equation_solution_l3415_341536

theorem quadratic_equation_solution : 
  ∃ x₁ x₂ : ℝ, (x₁^2 - 4 = 0 ∧ x₂^2 - 4 = 0) ∧ x₁ = 2 ∧ x₂ = -2 := by
  sorry

end quadratic_equation_solution_l3415_341536


namespace four_fours_l3415_341523

def four_digit_expr : ℕ → Prop :=
  fun n => ∃ (e : ℕ → ℕ → ℕ → ℕ → ℕ),
    (e 4 4 4 4 = n) ∧
    (∀ x y z w, e x y z w = n → x = 4 ∧ y = 4 ∧ z = 4 ∧ w = 4)

theorem four_fours :
  four_digit_expr 3 ∧
  four_digit_expr 4 ∧
  four_digit_expr 5 ∧
  four_digit_expr 6 := by sorry

end four_fours_l3415_341523


namespace product_eleven_reciprocal_squares_sum_l3415_341505

theorem product_eleven_reciprocal_squares_sum (a b : ℕ+) :
  a * b = 11 → (1 : ℚ) / a^2 + (1 : ℚ) / b^2 = 122 / 121 := by
  sorry

end product_eleven_reciprocal_squares_sum_l3415_341505


namespace simplify_fraction_with_sqrt_three_l3415_341589

theorem simplify_fraction_with_sqrt_three : 
  (1 / (1 + Real.sqrt 3)) * (1 / (1 - Real.sqrt 3)) = -1/2 := by sorry

end simplify_fraction_with_sqrt_three_l3415_341589


namespace rectangle_ratio_l3415_341528

theorem rectangle_ratio (w : ℚ) : 
  w > 0 ∧ 2 * w + 2 * 10 = 30 → w / 10 = 1 / 2 := by sorry

end rectangle_ratio_l3415_341528


namespace circle_sum_zero_l3415_341561

theorem circle_sum_zero (a : Fin 55 → ℤ) 
  (h : ∀ i : Fin 55, a i = a (i - 1) + a (i + 1)) : 
  ∀ i : Fin 55, a i = 0 := by
  sorry

end circle_sum_zero_l3415_341561


namespace files_per_folder_l3415_341529

theorem files_per_folder (initial_files : ℕ) (deleted_files : ℕ) (num_folders : ℕ) 
  (h1 : initial_files = 27)
  (h2 : deleted_files = 9)
  (h3 : num_folders = 3)
  (h4 : num_folders > 0) :
  (initial_files - deleted_files) / num_folders = 6 := by
  sorry

end files_per_folder_l3415_341529


namespace land_allocation_equations_l3415_341500

/-- Represents the land allocation problem for tea gardens and grain fields. -/
theorem land_allocation_equations (total_area : ℝ) (vegetable_percentage : ℝ) 
  (tea_grain_area : ℝ) (tea_area : ℝ) (grain_area : ℝ) : 
  total_area = 60 ∧ 
  vegetable_percentage = 0.1 ∧ 
  tea_grain_area = total_area - vegetable_percentage * total_area ∧
  tea_area = 2 * grain_area - 3 →
  tea_area + grain_area = 54 ∧ tea_area = 2 * grain_area - 3 :=
by sorry

end land_allocation_equations_l3415_341500


namespace base5_division_proof_l3415_341543

-- Define a function to convert from base 5 to decimal
def base5ToDecimal (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

-- Define a function to convert from decimal to base 5
def decimalToBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
  aux n []

-- Theorem statement
theorem base5_division_proof :
  let dividend := [4, 0, 1, 2]  -- 2104₅ in reverse order
  let divisor := [3, 2]         -- 23₅ in reverse order
  let quotient := [1, 4]        -- 41₅ in reverse order
  (base5ToDecimal dividend) / (base5ToDecimal divisor) = base5ToDecimal quotient :=
by sorry

end base5_division_proof_l3415_341543


namespace inscribed_parallelepiped_volume_l3415_341558

/-- The volume of a rectangular parallelepiped inscribed in a pyramid -/
theorem inscribed_parallelepiped_volume
  (a : ℝ) -- Side length of the square base of the pyramid
  (α β : ℝ) -- Angles α and β as described in the problem
  (h1 : 0 < a)
  (h2 : 0 < α ∧ α < π / 2)
  (h3 : 0 < β ∧ β < π / 2)
  (h4 : α + β < π / 2) :
  ∃ V : ℝ, -- Volume of the parallelepiped
    V = (a^3 * Real.sqrt 2 * Real.sin α * Real.cos α^2 * Real.sin β^3) /
        Real.sin (α + β)^3 :=
by sorry

end inscribed_parallelepiped_volume_l3415_341558


namespace burgers_per_day_l3415_341556

/-- The number of days in June -/
def june_days : ℕ := 30

/-- The cost of each burger in dollars -/
def burger_cost : ℕ := 13

/-- The total amount Alice spent on burgers in June in dollars -/
def total_spent : ℕ := 1560

/-- Alice bought burgers every day in June -/
axiom bought_daily : ∀ d : ℕ, d ≤ june_days → ∃ b : ℕ, b > 0

/-- Theorem: Alice purchased 4 burgers per day in June -/
theorem burgers_per_day : 
  (total_spent / burger_cost) / june_days = 4 := by sorry

end burgers_per_day_l3415_341556


namespace sqrt_transformation_l3415_341531

theorem sqrt_transformation (n : ℕ) (h : n ≥ 1) : 
  Real.sqrt ((1 : ℝ) / n * ((1 : ℝ) / (n + 1) - (1 : ℝ) / (n + 2))) = 
  (1 : ℝ) / (n + 1) * Real.sqrt ((n + 1 : ℝ) / (n * (n + 2))) := by
  sorry

end sqrt_transformation_l3415_341531


namespace cylinder_volume_from_rectangle_l3415_341554

/-- The volume of a cylinder formed by rotating a rectangle about its longer side -/
theorem cylinder_volume_from_rectangle (length width : ℝ) (length_ge_width : length ≥ width) :
  let radius := length / 2
  let height := width
  let volume := π * radius^2 * height
  length = 20 ∧ width = 10 → volume = 1000 * π := by
  sorry

end cylinder_volume_from_rectangle_l3415_341554


namespace converse_opposite_numbers_correct_l3415_341572

theorem converse_opposite_numbers_correct :
  (∀ x y : ℝ, x = -y → x + y = 0) := by sorry

end converse_opposite_numbers_correct_l3415_341572


namespace investment_problem_l3415_341597

def first_investment_value (x : ℝ) : Prop :=
  let second_investment : ℝ := 1500
  let combined_return_rate : ℝ := 0.085
  let first_return_rate : ℝ := 0.07
  let second_return_rate : ℝ := 0.09
  (first_return_rate * x + second_return_rate * second_investment = 
   combined_return_rate * (x + second_investment)) ∧
  x = 500

theorem investment_problem : ∃ x : ℝ, first_investment_value x := by
  sorry

end investment_problem_l3415_341597


namespace student_distribution_l3415_341552

theorem student_distribution (total : ℕ) (schemes : ℕ) : 
  total = 7 → 
  schemes = 108 → 
  (∃ (boys girls : ℕ), 
    boys + girls = total ∧ 
    boys * Nat.choose girls 2 * 6 = schemes ∧
    boys = 3 ∧ 
    girls = 4) :=
by sorry

end student_distribution_l3415_341552


namespace min_both_like_problem_l3415_341586

def min_both_like (total surveyed beethoven_fans chopin_fans both_and_vivaldi : ℕ) : ℕ :=
  max (beethoven_fans + chopin_fans - total) both_and_vivaldi

theorem min_both_like_problem :
  let total := 200
  let beethoven_fans := 150
  let chopin_fans := 120
  let both_and_vivaldi := 80
  min_both_like total beethoven_fans chopin_fans both_and_vivaldi = 80 := by
sorry

end min_both_like_problem_l3415_341586


namespace circle_radius_from_sum_of_circumference_and_area_l3415_341569

theorem circle_radius_from_sum_of_circumference_and_area :
  ∀ r : ℝ, r > 0 →
    2 * Real.pi * r + Real.pi * r^2 = 530.929158456675 →
    r = Real.sqrt 170 := by
  sorry

end circle_radius_from_sum_of_circumference_and_area_l3415_341569


namespace perp_condition_relationship_l3415_341562

/-- A structure representing a line in 3D space -/
structure Line3D where
  -- Add necessary fields here
  mk :: -- Constructor

/-- A structure representing a plane in 3D space -/
structure Plane3D where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Predicate indicating if a line is perpendicular to a plane -/
def perp_to_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Predicate indicating if a line is perpendicular to countless lines in a plane -/
def perp_to_countless_lines (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Theorem stating the relationship between the two conditions -/
theorem perp_condition_relationship :
  (∀ (l : Line3D) (α : Plane3D), perp_to_plane l α → perp_to_countless_lines l α) ∧
  (∃ (l : Line3D) (α : Plane3D), perp_to_countless_lines l α ∧ ¬perp_to_plane l α) :=
sorry

end perp_condition_relationship_l3415_341562


namespace base_of_first_term_l3415_341592

theorem base_of_first_term (base x y : ℕ) : 
  base ^ x * 4 ^ y = 19683 → 
  x - y = 9 → 
  x = 9 → 
  base = 3 := by
sorry

end base_of_first_term_l3415_341592


namespace sufficient_not_necessary_l3415_341501

theorem sufficient_not_necessary :
  (∀ x : ℝ, (x + 1) * (x - 3) < 0 → x > -1) ∧
  (∃ x : ℝ, x > -1 ∧ (x + 1) * (x - 3) ≥ 0) :=
by sorry

end sufficient_not_necessary_l3415_341501


namespace pyramid_inequality_l3415_341585

/-- A triangular pyramid with vertex O and base ABC -/
structure TriangularPyramid (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] where
  O : V
  A : V
  B : V
  C : V

/-- The area of a triangle -/
def triangleArea (A B C : V) [NormedAddCommGroup V] [InnerProductSpace ℝ V] : ℝ :=
  sorry

/-- Statement of the theorem -/
theorem pyramid_inequality (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (pyramid : TriangularPyramid V) (M : V) :
  let S_ABC := triangleArea pyramid.A pyramid.B pyramid.C
  let S_MBC := triangleArea M pyramid.B pyramid.C
  let S_MAC := triangleArea M pyramid.A pyramid.C
  let S_MAB := triangleArea M pyramid.A pyramid.B
  ‖pyramid.O - M‖ * S_ABC ≤ 
    ‖pyramid.O - pyramid.A‖ * S_MBC + 
    ‖pyramid.O - pyramid.B‖ * S_MAC + 
    ‖pyramid.O - pyramid.C‖ * S_MAB :=
by
  sorry

end pyramid_inequality_l3415_341585


namespace square_of_difference_of_square_roots_l3415_341541

theorem square_of_difference_of_square_roots : 
  (Real.sqrt (5 + 4 * Real.sqrt 3) - Real.sqrt (5 - 4 * Real.sqrt 3))^2 = 10 + 2 * Complex.I * Real.sqrt 23 :=
by sorry

end square_of_difference_of_square_roots_l3415_341541


namespace dennis_floors_above_charlie_l3415_341521

/-- The floor number on which Frank lives -/
def frank_floor : ℕ := 16

/-- The floor number on which Charlie lives -/
def charlie_floor : ℕ := frank_floor / 4

/-- The floor number on which Dennis lives -/
def dennis_floor : ℕ := 6

/-- The number of floors Dennis lives above Charlie -/
def floors_above : ℕ := dennis_floor - charlie_floor

theorem dennis_floors_above_charlie : floors_above = 2 := by
  sorry

end dennis_floors_above_charlie_l3415_341521


namespace pet_store_cages_l3415_341512

def number_of_cages (initial_puppies : ℕ) (sold_puppies : ℕ) (puppies_per_cage : ℕ) : ℕ :=
  (initial_puppies - sold_puppies) / puppies_per_cage

theorem pet_store_cages : number_of_cages 18 3 5 = 3 := by
  sorry

end pet_store_cages_l3415_341512


namespace correct_weight_calculation_l3415_341582

/-- Given a class of boys with incorrect and correct average weights, calculate the correct weight that was misread. -/
theorem correct_weight_calculation (n : ℕ) (incorrect_avg correct_avg misread_weight : ℚ) 
  (h1 : n = 20)
  (h2 : incorrect_avg = 584/10)
  (h3 : correct_avg = 59)
  (h4 : misread_weight = 56) :
  let incorrect_total := n * incorrect_avg
  let correct_total := n * correct_avg
  let weight_difference := correct_total - incorrect_total
  misread_weight + weight_difference = 68 := by sorry

end correct_weight_calculation_l3415_341582


namespace compound_interest_rate_calculation_l3415_341535

/-- Compound interest rate calculation -/
theorem compound_interest_rate_calculation
  (P : ℝ) (A : ℝ) (t : ℝ) (n : ℝ)
  (h_P : P = 12000)
  (h_A : A = 15200)
  (h_t : t = 7)
  (h_n : n = 1)
  : ∃ r : ℝ, (A = P * (1 + r / n) ^ (n * t)) ∧ (abs (r - 0.0332) < 0.0001) :=
sorry

end compound_interest_rate_calculation_l3415_341535


namespace no_valid_permutation_1986_l3415_341591

/-- Represents a permutation of the sequence 1,1,2,2,...,n,n -/
def Permutation (n : ℕ) := Fin (2*n) → Fin n

/-- The separation between pairs in a permutation -/
def separation (n : ℕ) (p : Permutation n) (i : Fin n) : ℕ := sorry

/-- A permutation satisfies the separation condition if for each i,
    there are exactly i numbers between the two occurrences of i -/
def satisfies_separation (n : ℕ) (p : Permutation n) : Prop :=
  ∀ i : Fin n, separation n p i = i.val

/-- The main theorem: there is no permutation of 1,1,2,2,...,1986,1986
    that satisfies the separation condition -/
theorem no_valid_permutation_1986 :
  ¬ ∃ (p : Permutation 1986), satisfies_separation 1986 p :=
sorry

end no_valid_permutation_1986_l3415_341591


namespace solution_set_of_inequality_l3415_341584

theorem solution_set_of_inequality (x : ℝ) :
  (2 * x + 4 > 0) ↔ (x > -2) := by
  sorry

end solution_set_of_inequality_l3415_341584


namespace power_sum_l3415_341553

theorem power_sum (a m n : ℝ) (h1 : a^m = 2) (h2 : a^n = 3) : a^(m+n) = 6 := by
  sorry

end power_sum_l3415_341553


namespace abs_neg_two_thirds_l3415_341575

theorem abs_neg_two_thirds : |(-2 : ℚ) / 3| = 2 / 3 := by
  sorry

end abs_neg_two_thirds_l3415_341575


namespace cookies_per_bag_l3415_341503

theorem cookies_per_bag (chocolate_chip : ℕ) (oatmeal : ℕ) (baggies : ℕ) :
  chocolate_chip = 13 →
  oatmeal = 41 →
  baggies = 6 →
  (chocolate_chip + oatmeal) / baggies = 9 := by
sorry

end cookies_per_bag_l3415_341503


namespace total_cost_after_discounts_l3415_341568

-- Define the original costs and discount percentages
def laptop_original_cost : ℚ := 800
def accessories_original_cost : ℚ := 200
def laptop_discount_percent : ℚ := 15
def accessories_discount_percent : ℚ := 10

-- Define the function to calculate the discounted price
def discounted_price (original_cost : ℚ) (discount_percent : ℚ) : ℚ :=
  original_cost * (1 - discount_percent / 100)

-- Theorem statement
theorem total_cost_after_discounts :
  discounted_price laptop_original_cost laptop_discount_percent +
  discounted_price accessories_original_cost accessories_discount_percent = 860 := by
  sorry

end total_cost_after_discounts_l3415_341568


namespace largest_solution_of_equation_l3415_341537

theorem largest_solution_of_equation (x : ℝ) :
  (3 * (9 * x^2 + 10 * x + 11) = x * (9 * x - 45)) →
  x ≤ (-1 / 2 : ℝ) :=
by sorry

end largest_solution_of_equation_l3415_341537


namespace inequality_implies_theta_range_l3415_341594

open Real

theorem inequality_implies_theta_range (θ : ℝ) :
  θ ∈ Set.Icc 0 (2 * π) →
  3 * (sin θ ^ 5 + cos (2 * θ) ^ 5) > 5 * (sin θ ^ 3 + cos (2 * θ) ^ 3) →
  θ ∈ Set.Ioo (7 * π / 6) (11 * π / 6) :=
by sorry

end inequality_implies_theta_range_l3415_341594


namespace jack_minimum_cars_per_hour_l3415_341579

/-- The minimum number of cars Jack can change oil in per hour -/
def jack_cars_per_hour : ℝ := 3

/-- The number of hours worked per day -/
def hours_per_day : ℝ := 8

/-- The number of cars Paul can change oil in per hour -/
def paul_cars_per_hour : ℝ := 2

/-- The minimum number of cars both mechanics can finish per day -/
def min_cars_per_day : ℝ := 40

theorem jack_minimum_cars_per_hour :
  jack_cars_per_hour * hours_per_day + paul_cars_per_hour * hours_per_day ≥ min_cars_per_day ∧
  ∀ x : ℝ, x * hours_per_day + paul_cars_per_hour * hours_per_day ≥ min_cars_per_day → x ≥ jack_cars_per_hour :=
by sorry

end jack_minimum_cars_per_hour_l3415_341579


namespace min_c_value_l3415_341555

theorem min_c_value (a b c : ℕ+) (h1 : a ≤ b) (h2 : b < c)
  (h3 : ∃! p : ℝ × ℝ, (2 * p.1 + p.2 = 2023) ∧
    (p.2 = |p.1 - a.val| + |p.1 - b.val| + |p.1 - c.val|)) :
  c.val ≥ 2022 ∧ ∃ a b : ℕ+, a ≤ b ∧ b < 2022 ∧
    ∃! p : ℝ × ℝ, (2 * p.1 + p.2 = 2023) ∧
      (p.2 = |p.1 - a.val| + |p.1 - b.val| + |p.1 - 2022|) := by
  sorry

end min_c_value_l3415_341555
