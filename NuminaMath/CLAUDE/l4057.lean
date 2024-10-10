import Mathlib

namespace second_year_sample_size_l4057_405734

/-- Represents the number of students to be sampled from each year group -/
structure SampleSize where
  first_year : ℕ
  second_year : ℕ
  third_year : ℕ
  fourth_year : ℕ

/-- Calculates the sample size for stratified sampling -/
def stratified_sample (total_students : ℕ) (sample_size : ℕ) (ratio : List ℕ) : SampleSize :=
  sorry

/-- Theorem stating the correct number of second-year students to be sampled -/
theorem second_year_sample_size :
  let total_students : ℕ := 5000
  let sample_size : ℕ := 260
  let ratio : List ℕ := [5, 4, 3, 1]
  let result := stratified_sample total_students sample_size ratio
  result.second_year = 80 := by sorry

end second_year_sample_size_l4057_405734


namespace smallest_divisible_k_l4057_405791

/-- The polynomial p(z) = z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1 -/
def p (z : ℂ) : ℂ := z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1

/-- The function f(k) = z^k - 1 -/
def f (k : ℕ) (z : ℂ) : ℂ := z^k - 1

/-- Theorem stating that 120 is the smallest positive integer k such that p(z) divides f(k)(z) -/
theorem smallest_divisible_k : 
  (∀ z : ℂ, p z ∣ f 120 z) ∧ 
  (∀ k : ℕ, k < 120 → ∃ z : ℂ, ¬(p z ∣ f k z)) :=
sorry

end smallest_divisible_k_l4057_405791


namespace problem_solution_l4057_405766

theorem problem_solution (X : ℝ) : 
  (213 * 16 = 3408) → 
  ((213 * 16) + (1.6 * 2.13) = X) → 
  (X - (5/2) * 1.25 = 3408.283) := by
sorry

end problem_solution_l4057_405766


namespace recurring_larger_than_finite_l4057_405770

def recurring_decimal : ℚ := 1 + 3/10 + 5/100 + 42/10000 + 5/1000 * (1/9)
def finite_decimal : ℚ := 1 + 3/10 + 5/100 + 4/1000 + 2/10000

theorem recurring_larger_than_finite : recurring_decimal > finite_decimal := by
  sorry

end recurring_larger_than_finite_l4057_405770


namespace equal_goldfish_theorem_l4057_405760

/-- Number of months for Brent and Gretel to have the same number of goldfish -/
def equal_goldfish_months : ℕ := 8

/-- Brent's initial number of goldfish -/
def brent_initial : ℕ := 3

/-- Gretel's initial number of goldfish -/
def gretel_initial : ℕ := 243

/-- Brent's goldfish growth rate per month -/
def brent_growth_rate : ℝ := 3

/-- Gretel's goldfish growth rate per month -/
def gretel_growth_rate : ℝ := 1.5

/-- Brent's number of goldfish after n months -/
def brent_goldfish (n : ℕ) : ℝ := brent_initial * brent_growth_rate ^ n

/-- Gretel's number of goldfish after n months -/
def gretel_goldfish (n : ℕ) : ℝ := gretel_initial * gretel_growth_rate ^ n

/-- Theorem stating that Brent and Gretel have the same number of goldfish after equal_goldfish_months -/
theorem equal_goldfish_theorem : 
  brent_goldfish equal_goldfish_months = gretel_goldfish equal_goldfish_months :=
sorry

end equal_goldfish_theorem_l4057_405760


namespace baseball_league_games_l4057_405751

theorem baseball_league_games (N M : ℕ) : 
  (N > 2 * M) → 
  (M > 4) → 
  (4 * N + 5 * M = 94) → 
  (4 * N = 64) := by
sorry

end baseball_league_games_l4057_405751


namespace treasure_in_blown_out_dunes_l4057_405796

/-- The probability that a sand dune remains after being formed -/
def prob_remain : ℚ := 1 / 3

/-- The probability that a sand dune has a lucky coupon -/
def prob_lucky_coupon : ℚ := 2 / 3

/-- The probability that a blown-out sand dune contains both treasure and lucky coupon -/
def prob_both : ℚ := 8888888888888889 / 100000000000000000

/-- The number of blown-out sand dunes considered to find the one with treasure -/
def num_blown_out_dunes : ℕ := 8

theorem treasure_in_blown_out_dunes :
  ∃ (n : ℕ), n = num_blown_out_dunes ∧ 
  (1 : ℚ) / n * prob_lucky_coupon = prob_both ∧
  n = ⌈(1 : ℚ) / (prob_both / prob_lucky_coupon)⌉ :=
sorry

end treasure_in_blown_out_dunes_l4057_405796


namespace class_average_weight_l4057_405776

/-- Given two sections A and B in a class, prove that the average weight of the whole class is 38 kg -/
theorem class_average_weight (students_A : ℕ) (students_B : ℕ) (avg_weight_A : ℝ) (avg_weight_B : ℝ)
  (h1 : students_A = 30)
  (h2 : students_B = 20)
  (h3 : avg_weight_A = 40)
  (h4 : avg_weight_B = 35) :
  (students_A * avg_weight_A + students_B * avg_weight_B) / (students_A + students_B) = 38 := by
  sorry

end class_average_weight_l4057_405776


namespace pure_imaginary_condition_l4057_405781

theorem pure_imaginary_condition (a : ℝ) : 
  (Complex.I + 1) * (Complex.I * a + 2) = Complex.I * (Complex.I * b + c) → a = 2 := by
  sorry

end pure_imaginary_condition_l4057_405781


namespace investment_value_change_l4057_405700

theorem investment_value_change (k m : ℝ) : 
  let increase_factor := 1 + k / 100
  let decrease_factor := 1 - m / 100
  let overall_factor := increase_factor * decrease_factor
  overall_factor = 1 + (k - m - k * m / 100) / 100 :=
by sorry

end investment_value_change_l4057_405700


namespace book_pages_theorem_l4057_405740

theorem book_pages_theorem (total_pages : ℚ) (read_pages : ℚ) 
  (h1 : read_pages = 3 / 7 * total_pages) : 
  ∃ (remaining_pages : ℚ),
    remaining_pages = 4 / 7 * total_pages ∧ 
    read_pages = 3 / 4 * remaining_pages := by
  sorry

end book_pages_theorem_l4057_405740


namespace smallest_primes_satisfying_conditions_l4057_405719

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem smallest_primes_satisfying_conditions (p q : ℕ) :
  is_prime p ∧ is_prime q ∧ is_prime (p * q + 1) ∧ p - q > 40 →
  p = 53 ∧ q = 2 :=
sorry

end smallest_primes_satisfying_conditions_l4057_405719


namespace pizza_solution_l4057_405746

/-- Represents the number of pizza slices with different topping combinations -/
structure PizzaToppings where
  total : ℕ
  ham : ℕ
  pineapple : ℕ
  jalapeno : ℕ
  all_three : ℕ
  ham_only : ℕ
  pineapple_only : ℕ
  jalapeno_only : ℕ
  ham_pineapple : ℕ
  ham_jalapeno : ℕ
  pineapple_jalapeno : ℕ

/-- The pizza topping problem -/
def pizza_problem (p : PizzaToppings) : Prop :=
  p.total = 24 ∧
  p.ham = 15 ∧
  p.pineapple = 10 ∧
  p.jalapeno = 14 ∧
  p.all_three = p.jalapeno_only ∧
  p.total = p.ham_only + p.pineapple_only + p.jalapeno_only + 
            p.ham_pineapple + p.ham_jalapeno + p.pineapple_jalapeno + p.all_three ∧
  p.ham = p.ham_only + p.ham_pineapple + p.ham_jalapeno + p.all_three ∧
  p.pineapple = p.pineapple_only + p.ham_pineapple + p.pineapple_jalapeno + p.all_three ∧
  p.jalapeno = p.jalapeno_only + p.ham_jalapeno + p.pineapple_jalapeno + p.all_three

theorem pizza_solution (p : PizzaToppings) (h : pizza_problem p) : p.all_three = 5 := by
  sorry

end pizza_solution_l4057_405746


namespace tan_alpha_values_l4057_405786

theorem tan_alpha_values (α : Real) (h : 2 * Real.sin (2 * α) = 1 - Real.cos (2 * α)) :
  Real.tan α = 2 ∨ Real.tan α = 0 := by
  sorry

end tan_alpha_values_l4057_405786


namespace miles_driven_l4057_405743

/-- Calculates the number of miles driven given car rental costs and total expenses --/
theorem miles_driven (rental_cost gas_needed gas_price per_mile_charge total_cost : ℚ) : 
  rental_cost = 150 →
  gas_needed = 8 →
  gas_price = 3.5 →
  per_mile_charge = 0.5 →
  total_cost = 338 →
  (total_cost - (rental_cost + gas_needed * gas_price)) / per_mile_charge = 320 := by
  sorry


end miles_driven_l4057_405743


namespace sum_squares_equality_l4057_405735

theorem sum_squares_equality (N : ℕ) : 
  (1^2 + 2^2 + 3^2 + 4^2) / 4 = (2000^2 + 2001^2 + 2002^2 + 2003^2) / N → N = 2134 := by
  sorry

end sum_squares_equality_l4057_405735


namespace shirt_cost_theorem_l4057_405748

theorem shirt_cost_theorem (cost_first : ℕ) (cost_difference : ℕ) : 
  cost_first = 15 → cost_difference = 6 → cost_first + (cost_first - cost_difference) = 24 := by
  sorry

end shirt_cost_theorem_l4057_405748


namespace water_tower_problem_l4057_405771

theorem water_tower_problem (total_capacity : ℕ) (n1 n2 n3 n4 n5 : ℕ) :
  total_capacity = 2700 →
  n1 = 300 →
  n2 = 2 * n1 →
  n3 = n2 + 100 →
  n4 = 3 * n1 →
  n5 = n3 / 2 →
  n1 + n2 + n3 + n4 + n5 > total_capacity :=
by sorry

end water_tower_problem_l4057_405771


namespace solve_ticket_problem_l4057_405753

/-- Represents the cost of tickets and number of students for two teachers. -/
structure TicketInfo where
  student_price : ℕ
  adult_price : ℕ
  kadrnozka_students : ℕ
  hnizdo_students : ℕ

/-- Checks if the given TicketInfo satisfies all the problem conditions. -/
def satisfies_conditions (info : TicketInfo) : Prop :=
  info.adult_price > info.student_price ∧
  info.adult_price ≤ 2 * info.student_price ∧
  info.student_price * info.kadrnozka_students + info.adult_price = 994 ∧
  info.hnizdo_students = info.kadrnozka_students + 3 ∧
  info.student_price * info.hnizdo_students + info.adult_price = 1120

/-- Theorem stating the solution to the problem. -/
theorem solve_ticket_problem :
  ∃ (info : TicketInfo), satisfies_conditions info ∧ 
    info.hnizdo_students = 25 ∧ info.adult_price = 70 :=
by
  sorry


end solve_ticket_problem_l4057_405753


namespace first_term_of_geometric_series_l4057_405799

/-- Given an infinite geometric series with first term a and common ratio r -/
def InfiniteGeometricSeries (a : ℝ) (r : ℝ) : Prop :=
  |r| < 1

theorem first_term_of_geometric_series
  (a : ℝ) (r : ℝ)
  (h_series : InfiniteGeometricSeries a r)
  (h_sum : a / (1 - r) = 30)
  (h_sum_squares : a^2 / (1 - r^2) = 180) :
  a = 10 := by
sorry

end first_term_of_geometric_series_l4057_405799


namespace largest_angle_in_special_triangle_l4057_405718

theorem largest_angle_in_special_triangle : ∀ (a b c : ℝ),
  -- Two angles sum to 4/3 of a right angle
  a + b = 4/3 * 90 →
  -- One angle is 40° larger than the other
  b = a + 40 →
  -- All angles are non-negative
  0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c →
  -- Sum of angles in a triangle is 180°
  a + b + c = 180 →
  -- The largest angle is 80°
  max a (max b c) = 80 := by
  sorry

end largest_angle_in_special_triangle_l4057_405718


namespace mango_rate_per_kg_l4057_405779

/-- The rate per kg of mangoes given the purchase details -/
theorem mango_rate_per_kg
  (grape_kg : ℕ)
  (grape_rate : ℕ)
  (mango_kg : ℕ)
  (total_paid : ℕ)
  (h1 : grape_kg = 8)
  (h2 : grape_rate = 70)
  (h3 : mango_kg = 9)
  (h4 : total_paid = 1055)
  : (total_paid - grape_kg * grape_rate) / mango_kg = 55 := by
  sorry

end mango_rate_per_kg_l4057_405779


namespace exists_n_no_rational_roots_l4057_405723

/-- A quadratic trinomial with real coefficients -/
structure QuadraticTrinomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The value of a quadratic trinomial at a given x -/
def QuadraticTrinomial.eval (p : QuadraticTrinomial) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

/-- Theorem: For any quadratic trinomial with real coefficients, 
    there exists a positive integer n such that p(x) = 1/n has no rational roots -/
theorem exists_n_no_rational_roots (p : QuadraticTrinomial) : 
  ∃ n : ℕ+, ¬∃ q : ℚ, p.eval q = (1 : ℝ) / n := by
  sorry

end exists_n_no_rational_roots_l4057_405723


namespace cube_root_four_solves_equation_l4057_405705

theorem cube_root_four_solves_equation :
  let x : ℝ := (4 : ℝ) ^ (1/3)
  x^3 - ⌊x⌋ = 3 := by sorry

end cube_root_four_solves_equation_l4057_405705


namespace race_time_difference_l4057_405714

def race_length : ℝ := 15
def malcolm_speed : ℝ := 6
def joshua_speed : ℝ := 7

theorem race_time_difference : 
  let malcolm_time := race_length * malcolm_speed
  let joshua_time := race_length * joshua_speed
  joshua_time - malcolm_time = 15 := by
sorry

end race_time_difference_l4057_405714


namespace candy_problem_l4057_405757

/-- The number of candies left in Shelly's bowl before her friend came over -/
def initial_candies : ℕ := 63

/-- The number of candies Shelly's friend brought -/
def friend_candies : ℕ := 2 * initial_candies

/-- The total number of candies after the friend's contribution -/
def total_candies : ℕ := initial_candies + friend_candies

/-- The number of candies Shelly's friend had after eating 10 -/
def friend_final_candies : ℕ := 85

theorem candy_problem :
  initial_candies = 63 ∧
  friend_candies = 2 * initial_candies ∧
  total_candies = initial_candies + friend_candies ∧
  friend_final_candies + 10 = total_candies / 2 :=
sorry

end candy_problem_l4057_405757


namespace binomial_probability_problem_l4057_405768

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  hp : 0 ≤ p ∧ p ≤ 1

/-- Probability of a binomial random variable being greater than or equal to k -/
noncomputable def prob_ge (X : BinomialRV) (k : ℕ) : ℝ := sorry

theorem binomial_probability_problem (ξ η : BinomialRV)
  (hξ : ξ.n = 2)
  (hη : η.n = 4)
  (hp : ξ.p = η.p)
  (hprob : prob_ge ξ 1 = 5/9) :
  prob_ge η 2 = 11/27 := by sorry

end binomial_probability_problem_l4057_405768


namespace michelle_oranges_l4057_405731

theorem michelle_oranges :
  ∀ (total : ℕ),
  (total / 3 : ℚ) + 5 + 7 = total →
  total = 18 :=
by
  sorry

end michelle_oranges_l4057_405731


namespace middle_number_values_l4057_405762

/-- Represents a three-layer product pyramid --/
structure ProductPyramid where
  bottom_left : ℕ+
  bottom_middle : ℕ+
  bottom_right : ℕ+

/-- Calculates the top number of the pyramid --/
def top_number (p : ProductPyramid) : ℕ :=
  (p.bottom_left * p.bottom_middle) * (p.bottom_middle * p.bottom_right)

/-- Theorem stating the possible values for the middle number --/
theorem middle_number_values (p : ProductPyramid) :
  top_number p = 90 → p.bottom_middle = 1 ∨ p.bottom_middle = 3 := by
  sorry

#check middle_number_values

end middle_number_values_l4057_405762


namespace min_floor_sum_l4057_405755

theorem min_floor_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ (a₀ b₀ c₀ : ℝ) (ha₀ : a₀ > 0) (hb₀ : b₀ > 0) (hc₀ : c₀ > 0),
    (⌊(2*a₀+b₀)/c₀⌋ + ⌊(b₀+2*c₀)/a₀⌋ + ⌊(c₀+2*a₀)/b₀⌋ = 9) ∧
    (∀ (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0),
      ⌊(2*x+y)/z⌋ + ⌊(y+2*z)/x⌋ + ⌊(z+2*x)/y⌋ ≥ 9) :=
by sorry

end min_floor_sum_l4057_405755


namespace absolute_value_not_always_greater_than_negative_l4057_405756

theorem absolute_value_not_always_greater_than_negative : ∃ a : ℝ, |a| ≤ -a := by
  sorry

end absolute_value_not_always_greater_than_negative_l4057_405756


namespace solve_equation_l4057_405702

theorem solve_equation (x : ℚ) : (4/7 : ℚ) * (1/5 : ℚ) * x = 12 → x = 105 := by
  sorry

end solve_equation_l4057_405702


namespace percentage_of_female_dogs_l4057_405708

theorem percentage_of_female_dogs (total_dogs : ℕ) (birth_ratio : ℚ) (puppies_per_birth : ℕ) (total_puppies : ℕ) :
  total_dogs = 40 →
  birth_ratio = 3 / 4 →
  puppies_per_birth = 10 →
  total_puppies = 180 →
  (↑total_puppies : ℚ) = (birth_ratio * puppies_per_birth * (60 / 100 * total_dogs)) →
  60 = (100 * (total_puppies / (birth_ratio * puppies_per_birth * total_dogs))) := by
  sorry

end percentage_of_female_dogs_l4057_405708


namespace f_inequality_l4057_405728

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x + 1 else 2^x

theorem f_inequality (x : ℝ) : f x + f (x - 1/2) > 1 ↔ x > -1/4 := by
  sorry

end f_inequality_l4057_405728


namespace nth_root_inequality_l4057_405722

theorem nth_root_inequality (n : ℕ) (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x / y) ^ (1 / (n + 1 : ℝ)) ≤ (x + n * y) / ((n + 1) * y) := by
  sorry

end nth_root_inequality_l4057_405722


namespace smallest_digit_sum_of_sum_l4057_405716

/-- A three-digit positive integer -/
def ThreeDigitInt := {n : ℕ // 100 ≤ n ∧ n < 1000}

/-- Extracts digits from a natural number -/
def digits (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec extract (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else extract (m / 10) ((m % 10) :: acc)
  extract n []

/-- Sum of digits of a natural number -/
def digitSum (n : ℕ) : ℕ := (digits n).sum

/-- All digits in two numbers are different -/
def allDigitsDifferent (a b : ThreeDigitInt) : Prop :=
  (digits a.val ++ digits b.val).Nodup

theorem smallest_digit_sum_of_sum (a b : ThreeDigitInt) 
  (h : allDigitsDifferent a b) : 
  ∃ (S : ℕ), S = a.val + b.val ∧ 1000 ≤ S ∧ S < 10000 ∧ 
  (∀ (T : ℕ), T = a.val + b.val → 1000 ≤ T ∧ T < 10000 → digitSum S ≤ digitSum T) ∧
  digitSum S = 8 :=
sorry

end smallest_digit_sum_of_sum_l4057_405716


namespace correct_oranges_to_put_back_l4057_405774

/-- The number of oranges Mary must put back to achieve the desired average price -/
def oranges_to_put_back (apple_price orange_price : ℚ) (total_fruit : ℕ) (initial_avg_price desired_avg_price : ℚ) : ℕ :=
  sorry

theorem correct_oranges_to_put_back :
  oranges_to_put_back (40/100) (60/100) 10 (54/100) (50/100) = 4 := by
  sorry

end correct_oranges_to_put_back_l4057_405774


namespace cube_net_opposite_face_l4057_405733

-- Define the faces of the cube
inductive Face : Type
  | W | X | Y | Z | V | z

-- Define the concept of opposite faces
def opposite (f1 f2 : Face) : Prop := sorry

-- Define the concept of adjacent faces in the net
def adjacent_in_net (f1 f2 : Face) : Prop := sorry

-- Define the concept of a valid cube net
def valid_cube_net (net : List Face) : Prop := sorry

-- Theorem statement
theorem cube_net_opposite_face (net : List Face) 
  (h_valid : valid_cube_net net)
  (h_z_central : adjacent_in_net Face.z Face.W ∧ 
                 adjacent_in_net Face.z Face.X ∧ 
                 adjacent_in_net Face.z Face.Y)
  (h_v_not_adjacent : ¬adjacent_in_net Face.z Face.V) :
  opposite Face.z Face.V := by sorry

end cube_net_opposite_face_l4057_405733


namespace range_of_m_l4057_405772

/-- The set A -/
def A : Set ℝ := {x | |x - 2| ≤ 4}

/-- The set B parameterized by m -/
def B (m : ℝ) : Set ℝ := {x | (x - 1 - m) * (x - 1 + m) ≤ 0}

/-- The proposition that ¬p is a necessary but not sufficient condition for ¬q -/
def not_p_necessary_not_sufficient_for_not_q (m : ℝ) : Prop :=
  (∀ x, x ∉ B m → x ∉ A) ∧ ∃ x, x ∉ B m ∧ x ∈ A

/-- The theorem stating the range of m -/
theorem range_of_m :
  ∀ m : ℝ, m > 0 ∧ not_p_necessary_not_sufficient_for_not_q m ↔ m ≥ 5 :=
sorry

end range_of_m_l4057_405772


namespace dress_discount_price_l4057_405754

theorem dress_discount_price (original_price discount_percentage : ℝ) 
  (h1 : original_price = 50)
  (h2 : discount_percentage = 30) : 
  original_price * (1 - discount_percentage / 100) = 35 := by
  sorry

end dress_discount_price_l4057_405754


namespace bike_ride_time_l4057_405784

theorem bike_ride_time (distance_to_julia : ℝ) (time_to_julia : ℝ) (distance_to_bernard : ℝ) :
  distance_to_julia = 2 →
  time_to_julia = 8 →
  distance_to_bernard = 5 →
  (distance_to_bernard / distance_to_julia) * time_to_julia = 20 :=
by sorry

end bike_ride_time_l4057_405784


namespace no_solution_condition_l4057_405777

theorem no_solution_condition (a : ℝ) : 
  (∀ x : ℝ, ¬((a^2 * x + 2*a) / (a*x - 2 + a^2) ≥ 0 ∧ a*x + a > 5/4)) ↔ 
  (a ≤ -1/2 ∨ a = 0) :=
sorry

end no_solution_condition_l4057_405777


namespace luke_money_calculation_l4057_405765

theorem luke_money_calculation (initial_amount spent_amount received_amount : ℕ) : 
  initial_amount = 48 → spent_amount = 11 → received_amount = 21 →
  initial_amount - spent_amount + received_amount = 58 := by
sorry

end luke_money_calculation_l4057_405765


namespace two_cubic_feet_to_cubic_inches_l4057_405764

-- Define the conversion factor
def inches_per_foot : ℕ := 12

-- Define the volume conversion
def cubic_inches_per_cubic_foot : ℕ := inches_per_foot ^ 3

-- Theorem statement
theorem two_cubic_feet_to_cubic_inches :
  2 * cubic_inches_per_cubic_foot = 3456 := by
  sorry

end two_cubic_feet_to_cubic_inches_l4057_405764


namespace total_yellow_balloons_l4057_405793

/-- The total number of yellow balloons given the number of balloons each person has -/
def total_balloons (fred_balloons sam_balloons mary_balloons : ℕ) : ℕ :=
  fred_balloons + sam_balloons + mary_balloons

/-- Theorem stating that the total number of yellow balloons is 18 -/
theorem total_yellow_balloons :
  total_balloons 5 6 7 = 18 := by
  sorry

end total_yellow_balloons_l4057_405793


namespace problem_solution_l4057_405792

theorem problem_solution (x n : ℝ) (h1 : x = 40) (h2 : ((x / 4) * 5) + n - 12 = 48) : n = 10 := by
  sorry

end problem_solution_l4057_405792


namespace library_avg_megabytes_per_hour_l4057_405730

/-- Calculates the average megabytes per hour of music in a digital library, rounded to the nearest whole number -/
def avgMegabytesPerHour (days : ℕ) (totalMB : ℕ) : ℕ :=
  let hoursPerDay : ℕ := 24
  let totalHours : ℕ := days * hoursPerDay
  let exactAvg : ℚ := totalMB / totalHours
  (exactAvg + 1/2).floor.toNat

/-- Theorem stating that for a 15-day library with 20,000 MB, the average is 56 MB/hour -/
theorem library_avg_megabytes_per_hour :
  avgMegabytesPerHour 15 20000 = 56 := by
  sorry

end library_avg_megabytes_per_hour_l4057_405730


namespace seventh_term_of_geometric_sequence_l4057_405707

/-- Given a geometric sequence of 9 terms where the first term is 4 and the last term is 2097152,
    prove that the 7th term is 1048576 -/
theorem seventh_term_of_geometric_sequence (a : ℕ → ℝ) :
  (∀ n, 1 ≤ n → n < 9 → a (n + 1) / a n = a 2 / a 1) →  -- geometric sequence condition
  a 1 = 4 →                                            -- first term
  a 9 = 2097152 →                                      -- last term
  a 7 = 1048576 :=                                     -- seventh term to prove
by sorry

end seventh_term_of_geometric_sequence_l4057_405707


namespace paper_string_area_l4057_405752

/-- The area of a paper string made from overlapping square sheets -/
theorem paper_string_area
  (num_sheets : ℕ)
  (sheet_side : ℝ)
  (overlap : ℝ)
  (h_num_sheets : num_sheets = 6)
  (h_sheet_side : sheet_side = 30)
  (h_overlap : overlap = 7) :
  (sheet_side + (num_sheets - 1) * (sheet_side - overlap)) * sheet_side = 4350 :=
sorry

end paper_string_area_l4057_405752


namespace geometric_sequence_common_ratio_l4057_405706

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h1 : a 2 = 1/2) 
  (h2 : a 5 = 4) 
  (h_geom : ∀ n : ℕ, n ≥ 1 → ∃ q : ℝ, a (n + 1) = a n * q) :
  ∃ q : ℝ, (∀ n : ℕ, n ≥ 1 → a (n + 1) = a n * q) ∧ q = 2 :=
by sorry

end geometric_sequence_common_ratio_l4057_405706


namespace probability_no_player_wins_all_is_11_16_l4057_405713

def num_players : Nat := 5

def num_games : Nat := (num_players * (num_players - 1)) / 2

def probability_no_player_wins_all : Rat :=
  1 - (num_players * (1 / 2 ^ (num_players - 1))) / (2 ^ num_games)

theorem probability_no_player_wins_all_is_11_16 :
  probability_no_player_wins_all = 11 / 16 := by
  sorry

end probability_no_player_wins_all_is_11_16_l4057_405713


namespace book_purchase_problem_l4057_405717

/-- Proves that given the conditions of the book purchase problem, the number of math books is 53. -/
theorem book_purchase_problem (total_books : ℕ) (math_cost history_cost total_price : ℚ) 
  (h_total : total_books = 90)
  (h_math_cost : math_cost = 4)
  (h_history_cost : history_cost = 5)
  (h_total_price : total_price = 397) :
  ∃ (math_books : ℕ), 
    math_books = 53 ∧ 
    math_books ≤ total_books ∧
    ∃ (history_books : ℕ),
      history_books = total_books - math_books ∧
      math_cost * math_books + history_cost * history_books = total_price := by
sorry

end book_purchase_problem_l4057_405717


namespace min_value_expression_l4057_405711

theorem min_value_expression (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : a + b = 2) :
  2 / (a + 3 * b) + 1 / (a - b) ≥ (3 + 2 * Real.sqrt 2) / 4 := by
  sorry

end min_value_expression_l4057_405711


namespace sphere_radius_relation_l4057_405783

/-- Given two spheres, one with radius 5 cm and another with 3 times its volume,
    prove that the radius of the larger sphere is 5 * (3^(1/3)) cm. -/
theorem sphere_radius_relation :
  ∀ (r : ℝ),
  (4 / 3 * Real.pi * r^3 = 3 * (4 / 3 * Real.pi * 5^3)) →
  r = 5 * (3^(1 / 3)) :=
by sorry

end sphere_radius_relation_l4057_405783


namespace delta_implies_sigma_l4057_405758

-- Define the type for pairs of real numbers
def Pair := ℝ × ℝ

-- Define equality for pairs
def pair_eq (p q : Pair) : Prop := p.1 = q.1 ∧ p.2 = q.2

-- Define the Ä operation
def op_delta (p q : Pair) : Pair :=
  (p.1 * q.1 + p.2 * q.2, p.2 * q.1 - p.1 * q.2)

-- Define the Å operation
def op_sigma (p q : Pair) : Pair :=
  (p.1 + q.1, p.2 + q.2)

-- State the theorem
theorem delta_implies_sigma :
  ∀ x y : ℝ, pair_eq (op_delta (3, 4) (x, y)) (11, -2) →
  pair_eq (op_sigma (3, 4) (x, y)) (4, 6) := by
  sorry

end delta_implies_sigma_l4057_405758


namespace event_attendance_l4057_405780

/-- Given an event with a total of 42 people where the number of children is twice the number of adults,
    prove that the number of children is 28. -/
theorem event_attendance (total : ℕ) (adults : ℕ) (children : ℕ)
    (h1 : total = 42)
    (h2 : total = adults + children)
    (h3 : children = 2 * adults) :
    children = 28 := by
  sorry

end event_attendance_l4057_405780


namespace fraction_inequality_range_l4057_405720

theorem fraction_inequality_range (m : ℝ) : 
  (∀ x : ℝ, (3 * x^2 + 2 * x + 2) / (x^2 + x + 1) ≥ m) ↔ m ≤ 2 := by
  sorry

end fraction_inequality_range_l4057_405720


namespace max_smaller_cuboids_l4057_405741

/-- Represents the dimensions of a cuboid -/
structure CuboidDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a cuboid given its dimensions -/
def cuboidVolume (d : CuboidDimensions) : ℕ :=
  d.length * d.width * d.height

/-- The dimensions of the smaller cuboid -/
def smallCuboid : CuboidDimensions :=
  { length := 6, width := 4, height := 3 }

/-- The dimensions of the larger cuboid -/
def largeCuboid : CuboidDimensions :=
  { length := 18, width := 15, height := 2 }

/-- Theorem stating the maximum number of whole smaller cuboids that can be formed -/
theorem max_smaller_cuboids :
  (cuboidVolume largeCuboid) / (cuboidVolume smallCuboid) = 7 :=
sorry

end max_smaller_cuboids_l4057_405741


namespace ellipse_and_segment_length_l4057_405721

noncomputable section

-- Define the circles and ellipse
def F₁ (x y : ℝ) : Prop := (x + Real.sqrt 3)^2 + y^2 = 9
def F₂ (x y : ℝ) : Prop := (x - Real.sqrt 3)^2 + y^2 = 1
def C (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the centers of the circles
def center_F₁ : ℝ × ℝ := (-Real.sqrt 3, 0)
def center_F₂ : ℝ × ℝ := (Real.sqrt 3, 0)

-- Define the line x = 2√3
def line (y : ℝ) : ℝ × ℝ := (2 * Real.sqrt 3, y)

-- Define the theorem
theorem ellipse_and_segment_length 
  (a b : ℝ) 
  (h_ab : a > b ∧ b > 0) 
  (h_foci : C (center_F₁.1) (center_F₁.2) a b ∧ C (center_F₂.1) (center_F₂.2) a b)
  (h_intersection : ∀ x y, F₁ x y ∧ F₂ x y → C x y a b)
  (M N : ℝ × ℝ) 
  (h_M : M.1 = 2 * Real.sqrt 3 ∧ M.2 > 0)
  (h_N : N.1 = 2 * Real.sqrt 3)
  (h_orthogonal : (M.1 - center_F₁.1) * (N.1 - center_F₂.1) + 
                  (M.2 - center_F₁.2) * (N.2 - center_F₂.2) = 0)
  (Q : ℝ × ℝ)
  (h_Q : ∃ t₁ t₂ : ℝ, 
    Q.1 = center_F₁.1 + t₁ * (M.1 - center_F₁.1) ∧
    Q.2 = center_F₁.2 + t₁ * (M.2 - center_F₁.2) ∧
    Q.1 = center_F₂.1 + t₂ * (N.1 - center_F₂.1) ∧
    Q.2 = center_F₂.2 + t₂ * (N.2 - center_F₂.2))
  (h_min : ∀ M' N' : ℝ × ℝ, M'.1 = 2 * Real.sqrt 3 ∧ N'.1 = 2 * Real.sqrt 3 → 
    (M'.1 - center_F₁.1) * (N'.1 - center_F₂.1) + 
    (M'.2 - center_F₁.2) * (N'.2 - center_F₂.2) = 0 → 
    (M.2 - N.2)^2 ≤ (M'.2 - N'.2)^2) :
  (∀ x y, C x y a b ↔ x^2 / 2 + y^2 = 1) ∧
  ((M.1 - Q.1)^2 + (M.2 - Q.2)^2 = 9) :=
sorry

end ellipse_and_segment_length_l4057_405721


namespace fruits_problem_solution_l4057_405709

def fruits_problem (x : ℕ) : Prop :=
  let last_night_apples : ℕ := 3
  let last_night_bananas : ℕ := 1
  let last_night_oranges : ℕ := 4
  let today_apples : ℕ := last_night_apples + 4
  let today_bananas : ℕ := x * last_night_bananas
  let today_oranges : ℕ := 2 * today_apples
  let total_fruits : ℕ := 39
  (last_night_apples + last_night_bananas + last_night_oranges + 
   today_apples + today_bananas + today_oranges) = total_fruits

theorem fruits_problem_solution : fruits_problem 10 := by
  sorry

end fruits_problem_solution_l4057_405709


namespace line_ellipse_intersection_l4057_405710

/-- The line y = k(x-1) + 1 intersects the ellipse (x^2 / 9) + (y^2 / 4) = 1 for any real k -/
theorem line_ellipse_intersection (k : ℝ) :
  ∃ (x y : ℝ), y = k * (x - 1) + 1 ∧ (x^2 / 9) + (y^2 / 4) = 1 := by
  sorry

end line_ellipse_intersection_l4057_405710


namespace biggest_number_in_ratio_l4057_405785

theorem biggest_number_in_ratio (a b c d : ℕ) : 
  a + b + c + d = 1344 →
  2 * b = 3 * a →
  4 * a = 2 * c →
  5 * a = 2 * d →
  d ≤ 480 ∧ (∃ (x : ℕ), d = 480) :=
by sorry

end biggest_number_in_ratio_l4057_405785


namespace exists_noncommuting_matrix_exp_l4057_405759

open Matrix

/-- Definition of matrix exponential -/
def matrix_exp (M : Matrix (Fin 2) (Fin 2) ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  1 + M + (1/2) • (M * M) + (1/6) • (M * M * M) + sorry

/-- Theorem: There exist 2x2 matrices A and B such that exp(A+B) ≠ exp(A)exp(B) -/
theorem exists_noncommuting_matrix_exp :
  ∃ (A B : Matrix (Fin 2) (Fin 2) ℝ), matrix_exp (A + B) ≠ matrix_exp A * matrix_exp B :=
sorry

end exists_noncommuting_matrix_exp_l4057_405759


namespace negation_of_conditional_l4057_405701

theorem negation_of_conditional (a : ℝ) :
  ¬(a > 0 → a^2 > 0) ↔ (a ≤ 0 → a^2 ≤ 0) :=
by sorry

end negation_of_conditional_l4057_405701


namespace angle_measures_l4057_405703

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the properties of the triangle
def valid_triangle (t : Triangle) : Prop :=
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧ t.A + t.B + t.C = 180

-- Define the ratio condition
def ratio_condition (t : Triangle) : Prop :=
  t.A / t.B = 1 / 2 ∧ t.B / t.C = 2 / 3

-- Theorem statement
theorem angle_measures (t : Triangle) 
  (h1 : valid_triangle t) (h2 : ratio_condition t) : 
  t.A = 30 ∧ t.B = 60 ∧ t.C = 90 :=
sorry

end angle_measures_l4057_405703


namespace conditions_necessary_not_sufficient_l4057_405712

theorem conditions_necessary_not_sufficient :
  (∀ x y : ℝ, (2 < x ∧ x < 3 ∧ 0 < y ∧ y < 1) → (2 < x + y ∧ x + y < 4 ∧ 0 < x * y ∧ x * y < 3)) ∧
  (∃ x y : ℝ, (2 < x + y ∧ x + y < 4 ∧ 0 < x * y ∧ x * y < 3) ∧ ¬(2 < x ∧ x < 3 ∧ 0 < y ∧ y < 1)) :=
by sorry

end conditions_necessary_not_sufficient_l4057_405712


namespace rectangle_area_l4057_405729

theorem rectangle_area (L W : ℝ) (h1 : L + W = 7) (h2 : L^2 + W^2 = 25) : L * W = 12 := by
  sorry

#check rectangle_area

end rectangle_area_l4057_405729


namespace total_guitars_count_l4057_405744

/-- The number of guitars owned by Davey -/
def daveys_guitars : ℕ := 18

/-- The number of guitars owned by Barbeck -/
def barbecks_guitars : ℕ := daveys_guitars / 3

/-- The number of guitars owned by Steve -/
def steves_guitars : ℕ := barbecks_guitars / 2

/-- The total number of guitars -/
def total_guitars : ℕ := daveys_guitars + barbecks_guitars + steves_guitars

theorem total_guitars_count : total_guitars = 27 := by
  sorry

end total_guitars_count_l4057_405744


namespace production_line_b_units_l4057_405732

theorem production_line_b_units (total : ℕ) (a b c : ℕ) : 
  total = 16800 →
  total = a + b + c →
  b - a = c - b →
  b = 5600 := by
  sorry

end production_line_b_units_l4057_405732


namespace carpool_commute_days_l4057_405769

/-- Proves that the number of commuting days per week is 5 given the carpool conditions --/
theorem carpool_commute_days : 
  let total_commute : ℝ := 21 -- miles one way
  let gas_cost : ℝ := 2.5 -- $/gallon
  let car_efficiency : ℝ := 30 -- miles/gallon
  let weeks_per_month : ℕ := 4
  let individual_payment : ℝ := 14 -- $ per month
  let num_friends : ℕ := 5
  
  -- Calculate the number of commuting days per week
  let commute_days : ℝ := 
    (individual_payment * num_friends) / 
    (gas_cost * (2 * total_commute / car_efficiency) * weeks_per_month)
  
  commute_days = 5 := by
  sorry

end carpool_commute_days_l4057_405769


namespace circle_area_from_diameter_endpoints_l4057_405738

/-- Given two points C and D as endpoints of a diameter of a circle,
    calculate the area of the circle. -/
theorem circle_area_from_diameter_endpoints
  (C D : ℝ × ℝ) -- C and D are points in the real plane
  (h : C = (-2, 3) ∧ D = (4, -1)) -- C and D have specific coordinates
  : (π * ((C.1 - D.1)^2 + (C.2 - D.2)^2) / 4) = 13 * π := by
  sorry


end circle_area_from_diameter_endpoints_l4057_405738


namespace exponent_division_l4057_405704

theorem exponent_division (a : ℝ) : a^7 / a^4 = a^3 := by
  sorry

end exponent_division_l4057_405704


namespace stratified_sampling_high_school_l4057_405788

theorem stratified_sampling_high_school
  (total_students : ℕ)
  (freshmen : ℕ)
  (sophomores : ℕ)
  (sample_size : ℕ)
  (h_total : total_students = 950)
  (h_freshmen : freshmen = 350)
  (h_sophomores : sophomores = 400)
  (h_sample : sample_size = 190) :
  let juniors := total_students - freshmen - sophomores
  let sample_ratio := sample_size / total_students
  let freshmen_sample := (sample_ratio * freshmen : ℚ).num
  let sophomores_sample := (sample_ratio * sophomores : ℚ).num
  let juniors_sample := (sample_ratio * juniors : ℚ).num
  (freshmen_sample, sophomores_sample, juniors_sample) = (70, 80, 40) := by
sorry

end stratified_sampling_high_school_l4057_405788


namespace average_monthly_growth_rate_l4057_405737

theorem average_monthly_growth_rate 
  (initial_sales : ℝ) 
  (final_sales : ℝ) 
  (months : ℕ) 
  (h1 : initial_sales = 5000)
  (h2 : final_sales = 7200)
  (h3 : months = 2) :
  ∃ (rate : ℝ), 
    rate = 1/5 ∧ 
    initial_sales * (1 + rate) ^ months = final_sales :=
by sorry

end average_monthly_growth_rate_l4057_405737


namespace bird_percentage_problem_l4057_405782

theorem bird_percentage_problem :
  ∀ (total : ℝ) (sparrows pigeons crows parrots : ℝ),
    sparrows = 0.4 * total →
    pigeons = 0.2 * total →
    crows = 0.15 * total →
    parrots = total - (sparrows + pigeons + crows) →
    (crows / (total - pigeons)) * 100 = 18.75 := by
  sorry

end bird_percentage_problem_l4057_405782


namespace range_of_H_l4057_405778

-- Define the function H
def H (x : ℝ) : ℝ := |x + 2| - |x - 2|

-- State the theorem about the range of H
theorem range_of_H :
  ∀ y : ℝ, (∃ x : ℝ, H x = y) ↔ -4 ≤ y ∧ y ≤ 4 :=
sorry

end range_of_H_l4057_405778


namespace trapezoid_crop_distribution_l4057_405749

theorem trapezoid_crop_distribution (a b h : ℝ) (angle : ℝ) :
  a > 0 → b > 0 → h > 0 →
  angle > 0 → angle < π / 2 →
  a = 100 → b = 200 → h = 50 * Real.sqrt 3 → angle = π / 3 →
  let total_area := (a + b) * h / 2
  let closest_to_longest_side := (b + (b - a) / 4) * h / 2
  closest_to_longest_side / total_area = 5 / 12 := by
  sorry

end trapezoid_crop_distribution_l4057_405749


namespace sqrt_4_equals_plus_minus_2_cube_root_negative_8_over_27_equals_negative_2_over_3_sqrt_diff_equals_point_1_abs_sqrt_2_minus_1_equals_sqrt_2_minus_1_l4057_405789

-- 1. Prove that ±√4 = ±2
theorem sqrt_4_equals_plus_minus_2 : ∀ x : ℝ, x^2 = 4 ↔ x = 2 ∨ x = -2 := by sorry

-- 2. Prove that ∛(-8/27) = -2/3
theorem cube_root_negative_8_over_27_equals_negative_2_over_3 : 
  ((-8/27 : ℝ) ^ (1/3 : ℝ)) = -2/3 := by sorry

-- 3. Prove that √0.09 - √0.04 = 0.1
theorem sqrt_diff_equals_point_1 : 
  Real.sqrt 0.09 - Real.sqrt 0.04 = 0.1 := by sorry

-- 4. Prove that |√2 - 1| = √2 - 1
theorem abs_sqrt_2_minus_1_equals_sqrt_2_minus_1 : 
  |Real.sqrt 2 - 1| = Real.sqrt 2 - 1 := by sorry

end sqrt_4_equals_plus_minus_2_cube_root_negative_8_over_27_equals_negative_2_over_3_sqrt_diff_equals_point_1_abs_sqrt_2_minus_1_equals_sqrt_2_minus_1_l4057_405789


namespace jamie_father_burns_500_calories_l4057_405790

/-- The number of calories in a pound of body fat -/
def calories_per_pound : ℕ := 3500

/-- The number of pounds Jamie's father wants to lose -/
def pounds_to_lose : ℕ := 5

/-- The number of days it takes Jamie's father to burn off the weight -/
def days_to_burn : ℕ := 35

/-- The number of calories Jamie's father eats per day -/
def calories_eaten_daily : ℕ := 2000

/-- The number of calories Jamie's father burns daily through light exercise -/
def calories_burned_daily : ℕ := (pounds_to_lose * calories_per_pound) / days_to_burn

theorem jamie_father_burns_500_calories :
  calories_burned_daily = 500 :=
sorry

end jamie_father_burns_500_calories_l4057_405790


namespace sum_of_radii_l4057_405726

noncomputable section

-- Define the circle radius
def R : ℝ := 5

-- Define the ratios of the sectors
def ratio1 : ℝ := 1
def ratio2 : ℝ := 2
def ratio3 : ℝ := 3

-- Define the base radii of the cones
def r₁ : ℝ := (ratio1 / (ratio1 + ratio2 + ratio3)) * R
def r₂ : ℝ := (ratio2 / (ratio1 + ratio2 + ratio3)) * R
def r₃ : ℝ := (ratio3 / (ratio1 + ratio2 + ratio3)) * R

theorem sum_of_radii : r₁ + r₂ + r₃ = R := by
  sorry

end sum_of_radii_l4057_405726


namespace frog_hop_probability_l4057_405739

/-- Represents a position on the 4x4 grid -/
structure Position :=
  (x : Fin 4)
  (y : Fin 4)

/-- Defines whether a position is on the edge of the grid -/
def isEdge (p : Position) : Bool :=
  p.x = 0 || p.x = 3 || p.y = 0 || p.y = 3

/-- Represents a single hop in one of the four cardinal directions -/
inductive Direction
  | Up
  | Down
  | Left
  | Right

/-- Applies a hop in the given direction, wrapping around if necessary -/
def hop (p : Position) (d : Direction) : Position :=
  match d with
  | Direction.Up => ⟨p.x - 1, p.y⟩
  | Direction.Down => ⟨p.x + 1, p.y⟩
  | Direction.Left => ⟨p.x, p.y - 1⟩
  | Direction.Right => ⟨p.x, p.y + 1⟩

/-- The probability of ending on an edge after three hops -/
def probEndOnEdge (start : Position) : ℚ :=
  sorry

theorem frog_hop_probability :
  probEndOnEdge ⟨1, 1⟩ = 37 / 64 := by sorry

end frog_hop_probability_l4057_405739


namespace chip_drawing_probability_l4057_405727

/-- The number of tan chips in the bag -/
def num_tan : ℕ := 4

/-- The number of pink chips in the bag -/
def num_pink : ℕ := 3

/-- The number of violet chips in the bag -/
def num_violet : ℕ := 5

/-- The number of green chips in the bag -/
def num_green : ℕ := 2

/-- The total number of chips in the bag -/
def total_chips : ℕ := num_tan + num_pink + num_violet + num_green

/-- The probability of drawing the chips in the specified arrangement -/
def probability : ℚ := (num_tan.factorial * num_pink.factorial * num_violet.factorial * 6) / total_chips.factorial

theorem chip_drawing_probability : probability = 1440 / total_chips.factorial :=
sorry

end chip_drawing_probability_l4057_405727


namespace max_cube_volume_from_sheet_l4057_405736

/-- Given a rectangular sheet of dimensions 60 cm by 25 cm, 
    prove that the maximum volume of a cube that can be constructed from this sheet is 3375 cm³. -/
theorem max_cube_volume_from_sheet (sheet_length : ℝ) (sheet_width : ℝ) 
  (h_length : sheet_length = 60) (h_width : sheet_width = 25) :
  ∃ (cube_edge : ℝ), 
    cube_edge > 0 ∧
    6 * cube_edge^2 ≤ sheet_length * sheet_width ∧
    ∀ (other_edge : ℝ), 
      other_edge > 0 → 
      6 * other_edge^2 ≤ sheet_length * sheet_width → 
      other_edge^3 ≤ cube_edge^3 ∧
    cube_edge^3 = 3375 := by
  sorry

end max_cube_volume_from_sheet_l4057_405736


namespace inequality_characterization_l4057_405794

theorem inequality_characterization (x y : ℝ) :
  2 * |x + y| ≤ |x| + |y| ↔
  (x ≥ 0 ∧ -3 * x ≤ y ∧ y ≤ -(1/3) * x) ∨
  (x < 0 ∧ -(1/3) * x ≤ y ∧ y ≤ -3 * x) :=
by sorry

end inequality_characterization_l4057_405794


namespace sum_of_first_50_even_integers_l4057_405773

theorem sum_of_first_50_even_integers (sum_odd : ℕ) : 
  sum_odd = 50^2 → 
  (Finset.sum (Finset.range 50) (λ i => 2*i + 2) = sum_odd + 50) :=
by sorry

end sum_of_first_50_even_integers_l4057_405773


namespace units_digit_of_sum_l4057_405725

-- Define a function to get the units digit of a number
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Define the exponentiation operation for natural numbers
def pow (base : ℕ) (exp : ℕ) : ℕ := base ^ exp

-- Theorem statement
theorem units_digit_of_sum (a b c d : ℕ) :
  unitsDigit (pow a b + pow c d) = 9 :=
sorry

end units_digit_of_sum_l4057_405725


namespace even_sum_probability_l4057_405745

/-- Represents a wheel with a given number of even and odd sections -/
structure Wheel where
  total : ℕ
  even : ℕ
  odd : ℕ
  h1 : even + odd = total
  h2 : 0 < total

/-- The probability of getting an even number on a wheel -/
def prob_even (w : Wheel) : ℚ :=
  w.even / w.total

/-- The probability of getting an odd number on a wheel -/
def prob_odd (w : Wheel) : ℚ :=
  w.odd / w.total

/-- Wheel A with 2 even and 3 odd sections -/
def wheel_a : Wheel :=
  { total := 5
  , even := 2
  , odd := 3
  , h1 := by simp
  , h2 := by simp }

/-- Wheel B with 1 even and 1 odd section -/
def wheel_b : Wheel :=
  { total := 2
  , even := 1
  , odd := 1
  , h1 := by simp
  , h2 := by simp }

/-- The probability of getting an even sum when spinning both wheels -/
def prob_even_sum (a b : Wheel) : ℚ :=
  prob_even a * prob_even b + prob_odd a * prob_odd b

theorem even_sum_probability :
  prob_even_sum wheel_a wheel_b = 1/2 := by
  sorry


end even_sum_probability_l4057_405745


namespace equation_solution_implies_m_range_l4057_405715

theorem equation_solution_implies_m_range :
  ∀ m : ℝ,
  (∃ x : ℝ, 2^(2*x) + (m^2 - 2*m - 5)*2^x + 1 = 0) →
  m ∈ Set.Icc (-1 : ℝ) 3 := by
sorry

end equation_solution_implies_m_range_l4057_405715


namespace cookie_radius_l4057_405724

theorem cookie_radius (x y : ℝ) :
  (x^2 + y^2 + 36 = 6*x + 9*y) →
  ∃ (center_x center_y : ℝ), 
    (x - center_x)^2 + (y - center_y)^2 = (3*Real.sqrt 5 / 2)^2 :=
by sorry

end cookie_radius_l4057_405724


namespace T_value_for_K_9_l4057_405775

-- Define the equation T = 4hK + 2
def T (h K : ℝ) : ℝ := 4 * h * K + 2

-- State the theorem
theorem T_value_for_K_9 (h : ℝ) :
  (T h 7 = 58) → (T h 9 = 74) := by
  sorry

end T_value_for_K_9_l4057_405775


namespace stable_performance_lower_variance_l4057_405750

/-- Represents an athlete's shooting performance -/
structure Athlete where
  average_score : ℝ
  variance : ℝ
  sessions : ℕ

/-- Defines stability of performance based on variance -/
def more_stable (a b : Athlete) : Prop :=
  a.variance < b.variance

theorem stable_performance_lower_variance 
  (a b : Athlete) 
  (h1 : a.average_score = b.average_score) 
  (h2 : a.sessions = b.sessions) 
  (h3 : a.sessions > 0) 
  (h4 : a.variance < b.variance) : 
  more_stable a b :=
sorry

end stable_performance_lower_variance_l4057_405750


namespace same_color_pair_count_l4057_405797

/-- The number of ways to choose a pair of socks of the same color -/
def choose_same_color_pair (white : Nat) (brown : Nat) (blue : Nat) : Nat :=
  Nat.choose white 2 + Nat.choose brown 2 + Nat.choose blue 2

/-- Theorem: The number of ways to choose a pair of socks of the same color
    from 4 white, 4 brown, and 2 blue socks is 13 -/
theorem same_color_pair_count :
  choose_same_color_pair 4 4 2 = 13 := by
  sorry

end same_color_pair_count_l4057_405797


namespace boys_camp_total_l4057_405795

theorem boys_camp_total (total : ℕ) : 
  (total : ℝ) * 0.2 * 0.7 = 42 → total = 300 := by
  sorry

end boys_camp_total_l4057_405795


namespace m_xor_n_equals_target_l4057_405767

-- Define the custom set operation ⊗
def setXor (A B : Set ℝ) : Set ℝ := (A ∪ B) \ (A ∩ B)

-- Define sets M and N
def M : Set ℝ := {x | |x| < 2}
def N : Set ℝ := {x | x^2 - 4*x + 3 < 0}

-- State the theorem
theorem m_xor_n_equals_target : 
  setXor M N = {x | -2 < x ∧ x ≤ 1 ∨ 2 ≤ x ∧ x < 3} := by sorry

end m_xor_n_equals_target_l4057_405767


namespace quadratic_inequality_solution_set_l4057_405787

theorem quadratic_inequality_solution_set 
  (a b c : ℝ) 
  (h : Set.Ioo (-1 : ℝ) 2 = {x : ℝ | a * x^2 + b * x + c > 0}) :
  {x : ℝ | c * x^2 + b * x + a > 0} = Set.Ioi (1/2 : ℝ) ∪ Set.Iic (-1 : ℝ) :=
by sorry

end quadratic_inequality_solution_set_l4057_405787


namespace max_area_rectangle_perimeter_36_l4057_405798

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.width + r.height)

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.width * r.height

/-- Theorem: The maximum area of a rectangle with perimeter 36 is 81 -/
theorem max_area_rectangle_perimeter_36 :
  (∃ (r : Rectangle), perimeter r = 36 ∧ 
    ∀ (s : Rectangle), perimeter s = 36 → area s ≤ area r) ∧
  (∀ (r : Rectangle), perimeter r = 36 → area r ≤ 81) := by
  sorry

end max_area_rectangle_perimeter_36_l4057_405798


namespace ten_digit_divisible_by_11_exists_l4057_405747

def is_valid_number (n : ℕ) : Prop :=
  (n ≥ 1000000000 ∧ n < 10000000000) ∧
  (∀ d : Fin 10, ∃! p : Fin 10, (n / (10 ^ p.val) % 10) = d) ∧
  n % 11 = 0

theorem ten_digit_divisible_by_11_exists : ∃ n : ℕ, is_valid_number n :=
sorry

end ten_digit_divisible_by_11_exists_l4057_405747


namespace michaels_pie_order_cost_l4057_405742

/-- Calculate the total cost of fruit for Michael's pie order --/
theorem michaels_pie_order_cost :
  let peach_pies := 8
  let apple_pies := 6
  let blueberry_pies := 5
  let mixed_fruit_pies := 3
  let peach_per_pie := 4
  let apple_per_pie := 3
  let blueberry_per_pie := 3.5
  let mixed_fruit_per_pie := 3
  let apple_price := 1.25
  let blueberry_price := 0.90
  let peach_price := 2.50
  let mixed_fruit_per_type := mixed_fruit_per_pie / 3

  let total_peaches := peach_pies * peach_per_pie + mixed_fruit_pies * mixed_fruit_per_type
  let total_apples := apple_pies * apple_per_pie + mixed_fruit_pies * mixed_fruit_per_type
  let total_blueberries := blueberry_pies * blueberry_per_pie + mixed_fruit_pies * mixed_fruit_per_type

  let peach_cost := total_peaches * peach_price
  let apple_cost := total_apples * apple_price
  let blueberry_cost := total_blueberries * blueberry_price

  let total_cost := peach_cost + apple_cost + blueberry_cost

  total_cost = 132.20 := by
    sorry

end michaels_pie_order_cost_l4057_405742


namespace janes_profit_is_correct_l4057_405763

/-- Farm data -/
structure FarmData where
  chickenCount : ℕ
  duckCount : ℕ
  quailCount : ℕ
  chickenEggsPerWeek : ℕ
  duckEggsPerWeek : ℕ
  quailEggsPerWeek : ℕ
  chickenEggPrice : ℚ
  duckEggPrice : ℚ
  quailEggPrice : ℚ
  chickenFeedCost : ℚ
  duckFeedCost : ℚ
  quailFeedCost : ℚ

/-- Sales data for a week -/
structure WeeklySales where
  chickenEggsSoldPercent : ℚ
  duckEggsSoldPercent : ℚ
  quailEggsSoldPercent : ℚ

def calculateProfit (farm : FarmData) (sales : List WeeklySales) : ℚ :=
  sorry

def janesFarm : FarmData := {
  chickenCount := 10,
  duckCount := 8,
  quailCount := 12,
  chickenEggsPerWeek := 6,
  duckEggsPerWeek := 4,
  quailEggsPerWeek := 10,
  chickenEggPrice := 2 / 12,
  duckEggPrice := 3 / 12,
  quailEggPrice := 4 / 12,
  chickenFeedCost := 1 / 2,
  duckFeedCost := 3 / 4,
  quailFeedCost := 3 / 5
}

def janesSales : List WeeklySales := [
  { chickenEggsSoldPercent := 1, duckEggsSoldPercent := 1, quailEggsSoldPercent := 1/2 },
  { chickenEggsSoldPercent := 1, duckEggsSoldPercent := 3/4, quailEggsSoldPercent := 1 },
  { chickenEggsSoldPercent := 0, duckEggsSoldPercent := 1, quailEggsSoldPercent := 1 }
]

theorem janes_profit_is_correct :
  calculateProfit janesFarm janesSales = 876 / 10 := by
  sorry

end janes_profit_is_correct_l4057_405763


namespace circle_center_and_radius_l4057_405761

/-- A circle in the 2D plane. -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The equation of a circle. -/
def Circle.equation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

theorem circle_center_and_radius :
  ∃ (c : Circle), (∀ x y : ℝ, c.equation x y ↔ (x - 1)^2 + y^2 = 1) ∧
                  c.center = (1, 0) ∧
                  c.radius = 1 := by
  sorry


end circle_center_and_radius_l4057_405761
