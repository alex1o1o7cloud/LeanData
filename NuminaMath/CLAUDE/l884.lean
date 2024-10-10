import Mathlib

namespace new_city_total_buildings_l884_88449

/-- Represents the number of buildings of each type in Pittsburgh -/
structure PittsburghBuildings where
  stores : ℕ
  hospitals : ℕ
  schools : ℕ
  police_stations : ℕ

/-- Calculates the number of buildings for the new city based on Pittsburgh's numbers -/
def new_city_buildings (p : PittsburghBuildings) : ℕ × ℕ × ℕ × ℕ :=
  (p.stores / 2, p.hospitals * 2, p.schools - 50, p.police_stations + 5)

/-- Theorem stating that the total number of buildings in the new city is 2175 -/
theorem new_city_total_buildings (p : PittsburghBuildings) 
  (h1 : p.stores = 2000)
  (h2 : p.hospitals = 500)
  (h3 : p.schools = 200)
  (h4 : p.police_stations = 20) :
  let (new_stores, new_hospitals, new_schools, new_police) := new_city_buildings p
  new_stores + new_hospitals + new_schools + new_police = 2175 := by
  sorry

#check new_city_total_buildings

end new_city_total_buildings_l884_88449


namespace parallel_line_length_l884_88496

/-- A triangle with a base of 24 inches and a parallel line dividing it into two equal areas -/
structure DividedTriangle where
  /-- The length of the base of the triangle -/
  base : ℝ
  /-- The length of the parallel line dividing the triangle -/
  parallel_line : ℝ
  /-- The base of the triangle is 24 inches -/
  base_length : base = 24
  /-- The parallel line divides the triangle into two equal areas -/
  equal_areas : parallel_line^2 = (1/2) * base^2

/-- The length of the parallel line in the divided triangle is 12√2 -/
theorem parallel_line_length (t : DividedTriangle) : t.parallel_line = 12 * Real.sqrt 2 := by
  sorry

end parallel_line_length_l884_88496


namespace cloth_seller_gain_percentage_l884_88451

/-- Calculates the gain percentage for a cloth seller -/
theorem cloth_seller_gain_percentage 
  (total_cloth : ℝ) 
  (profit_cloth : ℝ) 
  (total_cloth_positive : total_cloth > 0)
  (profit_ratio : profit_cloth = total_cloth / 3) :
  (profit_cloth / total_cloth) * 100 = 100 / 3 := by
sorry

end cloth_seller_gain_percentage_l884_88451


namespace division_with_equal_quotient_and_remainder_l884_88433

theorem division_with_equal_quotient_and_remainder :
  {N : ℕ | ∃ k : ℕ, 2014 = N * k + k ∧ k < N} = {2013, 1006, 105, 52} := by
  sorry

end division_with_equal_quotient_and_remainder_l884_88433


namespace square_land_area_l884_88424

/-- The area of a square land plot with side length 25 units is 625 square units. -/
theorem square_land_area (side_length : ℝ) (h : side_length = 25) : side_length ^ 2 = 625 := by
  sorry

end square_land_area_l884_88424


namespace zoo_recovery_time_l884_88410

/-- The total time spent recovering escaped animals from a zoo. -/
def total_recovery_time (lions rhinos recovery_time_per_animal : ℕ) : ℕ :=
  (lions + rhinos) * recovery_time_per_animal

/-- Theorem stating that given 3 lions, 2 rhinos, and 2 hours recovery time per animal,
    the total recovery time is 10 hours. -/
theorem zoo_recovery_time :
  total_recovery_time 3 2 2 = 10 := by
  sorry

end zoo_recovery_time_l884_88410


namespace election_results_l884_88482

/-- Election results theorem -/
theorem election_results 
  (total_students : ℕ) 
  (voter_turnout : ℚ) 
  (vote_percent_A vote_percent_B vote_percent_C vote_percent_D vote_percent_E : ℚ) : 
  total_students = 5000 →
  voter_turnout = 3/5 →
  vote_percent_A = 2/5 →
  vote_percent_B = 1/4 →
  vote_percent_C = 1/5 →
  vote_percent_D = 1/10 →
  vote_percent_E = 1/20 →
  (↑total_students * voter_turnout * vote_percent_A - ↑total_students * voter_turnout * vote_percent_B : ℚ) = 450 ∧
  (↑total_students * voter_turnout * (vote_percent_C + vote_percent_D + vote_percent_E) : ℚ) = 1050 := by
  sorry

end election_results_l884_88482


namespace star_neg_x_not_2x_squared_l884_88493

-- Define the star operation
def star (x y : ℝ) : ℝ := x^2 - y^2

-- Theorem stating that x ⋆ (-x) = 2x^2 is false
theorem star_neg_x_not_2x_squared : ¬ ∀ x : ℝ, star x (-x) = 2 * x^2 := by
  sorry

end star_neg_x_not_2x_squared_l884_88493


namespace product_minus_one_divisible_by_ten_l884_88425

theorem product_minus_one_divisible_by_ten :
  ∃ k : ℤ, 11 * 21 * 31 * 41 * 51 - 1 = 10 * k := by
  sorry

end product_minus_one_divisible_by_ten_l884_88425


namespace sector_central_angle_l884_88445

/-- Given a sector with circumference 12 and area 8, its central angle in radians is either 1 or 4. -/
theorem sector_central_angle (r : ℝ) (l : ℝ) (α : ℝ) : 
  l + 2 * r = 12 → 
  1 / 2 * l * r = 8 → 
  α = l / r → 
  α = 1 ∨ α = 4 := by
sorry

end sector_central_angle_l884_88445


namespace exists_perfect_pair_with_122_l884_88492

/-- Two natural numbers form a perfect pair if their sum and product are both perfect squares. -/
def IsPerfectPair (a b : ℕ) : Prop :=
  ∃ (x y : ℕ), a + b = x^2 ∧ a * b = y^2

/-- There exists a natural number that forms a perfect pair with 122. -/
theorem exists_perfect_pair_with_122 : ∃ (n : ℕ), IsPerfectPair 122 n := by
  sorry

end exists_perfect_pair_with_122_l884_88492


namespace carnival_earnings_example_l884_88427

/-- Represents the earnings of a carnival snack booth over a period of days -/
def carnival_earnings (popcorn_sales : ℕ) (cotton_candy_multiplier : ℕ) (days : ℕ) (rent : ℕ) (ingredients_cost : ℕ) : ℕ :=
  let daily_total := popcorn_sales + popcorn_sales * cotton_candy_multiplier
  let total_revenue := daily_total * days
  let total_expenses := rent + ingredients_cost
  total_revenue - total_expenses

/-- Theorem stating that the carnival snack booth's earnings after expenses for 5 days is $895 -/
theorem carnival_earnings_example : carnival_earnings 50 3 5 30 75 = 895 := by
  sorry

end carnival_earnings_example_l884_88427


namespace rectangle_area_l884_88426

theorem rectangle_area (width : ℝ) (length : ℝ) : 
  width = 5 → length = 4 * width → width * length = 100 := by
  sorry

end rectangle_area_l884_88426


namespace inequality_system_solutions_l884_88448

theorem inequality_system_solutions : 
  {x : ℤ | x ≥ 0 ∧ 5*x - 1 < 3*(x + 1) ∧ (1 - x) / 3 ≤ 1} = {0, 1} := by
  sorry

end inequality_system_solutions_l884_88448


namespace solution_set_inequality_l884_88497

theorem solution_set_inequality (x : ℝ) : 
  (2 * x) / (x - 1) < 1 ↔ -1 < x ∧ x < 1 :=
by sorry

end solution_set_inequality_l884_88497


namespace gcf_of_2000_and_7700_l884_88470

theorem gcf_of_2000_and_7700 : Nat.gcd 2000 7700 = 100 := by
  sorry

end gcf_of_2000_and_7700_l884_88470


namespace circular_arrangement_exists_l884_88489

theorem circular_arrangement_exists : ∃ (a : Fin 12 → Fin 12), Function.Bijective a ∧
  ∀ (i j : Fin 12), i ≠ j → |a i - a j| ≠ |i.val - j.val| := by
  sorry

end circular_arrangement_exists_l884_88489


namespace f_inequality_l884_88463

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f x = f (x + p)

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x < f y

def is_symmetric_about (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ x, f (c + x) = f (c - x)

theorem f_inequality (f : ℝ → ℝ) 
  (h1 : is_periodic f 4)
  (h2 : is_increasing_on f 0 2)
  (h3 : is_symmetric_about (fun x ↦ f (x + 2)) 0) :
  f 4.5 < f 7 ∧ f 7 < f 6.5 := by
  sorry

end f_inequality_l884_88463


namespace common_elements_count_l884_88487

def S := Finset.range 2005
def T := Finset.range 2005

def multiples_of_4 (n : ℕ) : ℕ := (n + 1) * 4
def multiples_of_6 (n : ℕ) : ℕ := (n + 1) * 6

def S_set := S.image multiples_of_4
def T_set := T.image multiples_of_6

theorem common_elements_count : (S_set ∩ T_set).card = 668 := by
  sorry

end common_elements_count_l884_88487


namespace polynomial_divisibility_existence_l884_88476

theorem polynomial_divisibility_existence : ∃ (r s : ℝ),
  ∃ (q : ℝ → ℝ), ∀ (x : ℝ),
    8 * x^4 - 4 * x^3 - 42 * x^2 + 45 * x - 10 = 
    ((x - r)^2 * (x - s) * (x - 1)) * q x :=
by sorry

end polynomial_divisibility_existence_l884_88476


namespace painting_time_equation_l884_88408

theorem painting_time_equation (t : ℝ) : t > 0 → t = 5/2 := by
  intro h
  have alice_rate : ℝ := 1/4
  have bob_rate : ℝ := 1/6
  have charlie_rate : ℝ := 1/12
  have combined_rate : ℝ := alice_rate + bob_rate + charlie_rate
  have break_time : ℝ := 1/2
  have painting_equation : (combined_rate * (t - break_time) = 1) := by sorry
  sorry

end painting_time_equation_l884_88408


namespace simplification_and_constant_coefficient_l884_88412

-- Define the expression as a function of x and square
def expression (x : ℝ) (square : ℝ) : ℝ :=
  (square * x^2 + 6*x + 8) - (6*x + 5*x^2 + 2)

theorem simplification_and_constant_coefficient :
  (∀ x : ℝ, expression x 3 = -2 * x^2 + 6) ∧
  (∃! square : ℝ, ∀ x : ℝ, expression x square = (expression 0 square)) :=
by sorry

end simplification_and_constant_coefficient_l884_88412


namespace f_2014_equals_2_l884_88486

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem f_2014_equals_2
  (f : ℝ → ℝ)
  (h1 : ∀ x, f x * f (x + 2) = 1)
  (h2 : f 1 = 3)
  (h3 : f 2 = 2) :
  f 2014 = 2 :=
sorry

end f_2014_equals_2_l884_88486


namespace expression_evaluation_l884_88413

theorem expression_evaluation : -1^4 - (1/6) * (|(-2)| - (-3)^2) = 1/6 := by
  sorry

end expression_evaluation_l884_88413


namespace john_total_distance_l884_88459

/-- Calculates the total distance cycled given a constant speed and total cycling time -/
def total_distance (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time

/-- Theorem: Given John's cycling conditions, he cycles 18 miles in total -/
theorem john_total_distance :
  let speed : ℝ := 6  -- miles per hour
  let time_before_rest : ℝ := 2  -- hours
  let time_after_rest : ℝ := 1  -- hour
  let total_time : ℝ := time_before_rest + time_after_rest
  total_distance speed total_time = 18 := by
  sorry

#check john_total_distance

end john_total_distance_l884_88459


namespace line_constant_value_l884_88407

theorem line_constant_value (m n p : ℝ) (h : p = 1/3) :
  ∃ C : ℝ, (m = 6*n + C ∧ m + 2 = 6*(n + p) + C) → C = 0 :=
sorry

end line_constant_value_l884_88407


namespace square_area_12cm_l884_88454

/-- The area of a square with side length 12 cm is 144 square centimeters. -/
theorem square_area_12cm (s : ℝ) (h : s = 12) : s^2 = 144 := by
  sorry

end square_area_12cm_l884_88454


namespace evaluate_expression_l884_88494

theorem evaluate_expression : 6 - 8 * (9 - 4^2) / 2 - 3 = 31 := by
  sorry

end evaluate_expression_l884_88494


namespace sum_first_ten_natural_numbers_l884_88402

/-- The sum of the first n natural numbers -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Theorem: The sum of the first 10 natural numbers is 55 -/
theorem sum_first_ten_natural_numbers : triangular_number 10 = 55 := by
  sorry

#eval triangular_number 10  -- This should output 55

end sum_first_ten_natural_numbers_l884_88402


namespace blue_marbles_count_l884_88485

theorem blue_marbles_count (total : ℕ) (red : ℕ) (prob_red_or_white : ℚ) 
  (h_total : total = 20)
  (h_red : red = 7)
  (h_prob : prob_red_or_white = 3/4) :
  ∃ blue : ℕ, blue = 5 ∧ 
    (red : ℚ) / total + (total - red - blue : ℚ) / total = prob_red_or_white :=
by sorry

end blue_marbles_count_l884_88485


namespace infinitely_many_primes_with_quadratic_nonresidue_l884_88474

theorem infinitely_many_primes_with_quadratic_nonresidue (a : ℤ) 
  (h_odd : Odd a) (h_not_square : ∀ n : ℤ, n ^ 2 ≠ a) :
  ∃ (S : Set ℕ), (∀ p ∈ S, Prime p) ∧ 
  Set.Infinite S ∧ 
  (∀ p ∈ S, ¬ ∃ x : ℤ, x ^ 2 ≡ a [ZMOD p]) :=
sorry

end infinitely_many_primes_with_quadratic_nonresidue_l884_88474


namespace problem_solution_l884_88478

theorem problem_solution (x : ℤ) : x - (28 - (37 - (15 - 18))) = 57 → x = 69 := by
  sorry

end problem_solution_l884_88478


namespace sum_equals_four_l884_88465

/-- Custom binary operation on real numbers -/
def custom_op (x y : ℝ) : ℝ := x * (1 - y)

/-- The solution set of the inequality -/
def solution_set : Set ℝ := Set.Ioo 2 3

/-- Theorem stating the sum of a and b equals 4 -/
theorem sum_equals_four (a b : ℝ) 
  (h : ∀ x ∈ solution_set, custom_op (x - a) (x - b) > 0) 
  (h_unique : ∀ x ∉ solution_set, custom_op (x - a) (x - b) ≤ 0) : 
  a + b = 4 := by
  sorry

end sum_equals_four_l884_88465


namespace calculation_proof_l884_88431

theorem calculation_proof :
  ((-20) - (-18) + 5 + (-9) = -6) ∧
  ((-3) * ((-1)^2003) - ((-4)^2) / (-2) = 11) :=
by sorry

end calculation_proof_l884_88431


namespace sqrt_equation_implies_value_l884_88453

theorem sqrt_equation_implies_value (a b : ℝ) :
  (Real.sqrt (a - 2 * b + 4) + (a + b - 5) ^ 2 = 0) →
  (4 * Real.sqrt a - Real.sqrt 24 / Real.sqrt b = 2 * Real.sqrt 2) :=
by sorry

end sqrt_equation_implies_value_l884_88453


namespace rabbit_count_l884_88469

/-- Given a total number of heads and a relationship between rabbit and chicken feet,
    prove the number of rabbits. -/
theorem rabbit_count (total_heads : ℕ) (rabbit_feet chicken_feet : ℕ → ℕ) : 
  total_heads = 40 →
  (∀ x, rabbit_feet x = 10 * chicken_feet (total_heads - x) - 8) →
  (∃ x, x = 33 ∧ 
        rabbit_feet x = 4 * x ∧ 
        chicken_feet (total_heads - x) = 2 * (total_heads - x)) :=
by sorry

end rabbit_count_l884_88469


namespace janet_jasmine_shampoo_l884_88438

/-- The amount of rose shampoo Janet has, in bottles -/
def rose_shampoo : ℚ := 1/3

/-- The amount of shampoo Janet uses per day, in bottles -/
def daily_usage : ℚ := 1/12

/-- The number of days Janet's shampoo will last -/
def days : ℕ := 7

/-- The total amount of shampoo Janet has, in bottles -/
def total_shampoo : ℚ := daily_usage * days

/-- The amount of jasmine shampoo Janet has, in bottles -/
def jasmine_shampoo : ℚ := total_shampoo - rose_shampoo

theorem janet_jasmine_shampoo : jasmine_shampoo = 1/4 := by
  sorry

end janet_jasmine_shampoo_l884_88438


namespace area_of_triangle_ABC_l884_88442

/-- Reflects a point over the y-axis -/
def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

/-- Reflects a point over the line y=x -/
def reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

/-- Calculates the area of a triangle given three points -/
def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  let x1 := p1.1
  let y1 := p1.2
  let x2 := p2.1
  let y2 := p2.2
  let x3 := p3.1
  let y3 := p3.2
  0.5 * abs ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))

theorem area_of_triangle_ABC : 
  let A : ℝ × ℝ := (5, 3)
  let B : ℝ × ℝ := reflect_y_axis A
  let C : ℝ × ℝ := reflect_y_eq_x B
  triangle_area A B C = 40 := by sorry

end area_of_triangle_ABC_l884_88442


namespace inequality_solution_set_l884_88484

-- Define the inequality function
def f (x : ℝ) : ℝ := x^2 - 2*x - 8

-- Define the solution set
def solution_set : Set ℝ := {x | x ≤ -2 ∨ x ≥ 4}

-- Theorem statement
theorem inequality_solution_set :
  {x : ℝ | f x ≥ 0} = solution_set := by sorry

end inequality_solution_set_l884_88484


namespace largest_fraction_l884_88490

theorem largest_fraction : 
  (8 : ℚ) / 9 > 7 / 8 ∧ 
  (8 : ℚ) / 9 > 66 / 77 ∧ 
  (8 : ℚ) / 9 > 55 / 66 ∧ 
  (8 : ℚ) / 9 > 4 / 5 := by
  sorry

end largest_fraction_l884_88490


namespace magnitude_a_plus_b_unique_k_parallel_l884_88491

/-- Given vectors in R^2 -/
def a : Fin 2 → ℝ := ![3, 2]
def b : Fin 2 → ℝ := ![0, 2]
def c : Fin 2 → ℝ := ![4, 1]

/-- The magnitude of the sum of vectors a and b is 5 -/
theorem magnitude_a_plus_b : ‖a + b‖ = 5 := by sorry

/-- The unique value of k such that a + k*c is parallel to 2*a - b is 3 -/
theorem unique_k_parallel : ∃! k : ℝ, ∃ t : ℝ, a + k • c = t • (2 • a - b) ∧ k = 3 := by sorry

end magnitude_a_plus_b_unique_k_parallel_l884_88491


namespace kellys_games_l884_88471

/-- Kelly's nintendo games problem -/
theorem kellys_games (initial_games : ℕ) (given_away : ℕ) (remaining_games : ℕ) : 
  initial_games = 106 → given_away = 64 → remaining_games = initial_games - given_away → remaining_games = 42 := by
  sorry

end kellys_games_l884_88471


namespace parallelepiped_volume_l884_88488

/-- A rectangular parallelepiped divided into eight parts -/
structure Parallelepiped where
  volume_A : ℝ
  volume_C : ℝ
  volume_B_prime : ℝ
  volume_C_prime : ℝ

/-- The theorem stating that the total volume of the parallelepiped is 790 -/
theorem parallelepiped_volume 
  (p : Parallelepiped) 
  (h1 : p.volume_A = 40)
  (h2 : p.volume_C = 300)
  (h3 : p.volume_B_prime = 360)
  (h4 : p.volume_C_prime = 90) :
  p.volume_A + p.volume_C + p.volume_B_prime + p.volume_C_prime = 790 :=
by sorry

end parallelepiped_volume_l884_88488


namespace bond_coupon_income_is_135_l884_88417

/-- Represents a bond with its characteristics -/
structure Bond where
  purchase_price : ℝ
  face_value : ℝ
  current_yield : ℝ
  duration : ℕ

/-- Calculates the annual coupon income for a given bond -/
def annual_coupon_income (b : Bond) : ℝ :=
  b.current_yield * b.purchase_price

/-- Theorem stating that for the given bond, the annual coupon income is 135 rubles -/
theorem bond_coupon_income_is_135 (b : Bond) 
  (h1 : b.purchase_price = 900)
  (h2 : b.face_value = 1000)
  (h3 : b.current_yield = 0.15)
  (h4 : b.duration = 3) :
  annual_coupon_income b = 135 := by
  sorry

end bond_coupon_income_is_135_l884_88417


namespace alice_chicken_amount_l884_88443

/-- Represents the grocery items in Alice's cart -/
structure GroceryCart where
  lettuce : ℕ
  cherryTomatoes : ℕ
  sweetPotatoes : ℕ
  broccoli : ℕ
  brusselSprouts : ℕ

/-- Calculates the total cost of items in the cart excluding chicken -/
def cartCost (cart : GroceryCart) : ℚ :=
  3 + 2.5 + (0.75 * cart.sweetPotatoes) + (2 * cart.broccoli) + 2.5

/-- Theorem: Alice has 1.5 pounds of chicken in her cart -/
theorem alice_chicken_amount (cart : GroceryCart) 
  (h1 : cart.lettuce = 1)
  (h2 : cart.cherryTomatoes = 1)
  (h3 : cart.sweetPotatoes = 4)
  (h4 : cart.broccoli = 2)
  (h5 : cart.brusselSprouts = 1)
  (h6 : 35 - (cartCost cart) - 11 = 6 * chicken_amount) :
  chicken_amount = 1.5 := by
  sorry

#check alice_chicken_amount

end alice_chicken_amount_l884_88443


namespace randy_wipes_days_l884_88455

/-- Calculates the number of days Randy can use wipes given the number of packs and wipes per pack -/
def days_of_wipes (walks_per_day : ℕ) (paws : ℕ) (packs : ℕ) (wipes_per_pack : ℕ) : ℕ :=
  let wipes_per_day := walks_per_day * paws
  let total_wipes := packs * wipes_per_pack
  total_wipes / wipes_per_day

/-- Theorem stating that Randy needs wipes for 90 days -/
theorem randy_wipes_days :
  days_of_wipes 2 4 6 120 = 90 := by
  sorry

end randy_wipes_days_l884_88455


namespace grid_constant_l884_88483

/-- A function representing the assignment of positive integers to grid points -/
def GridAssignment := ℤ → ℤ → ℕ+

/-- The condition that each value is the arithmetic mean of its neighbors -/
def is_arithmetic_mean (f : GridAssignment) : Prop :=
  ∀ x y : ℤ, (f x y : ℚ) = ((f (x-1) y + f (x+1) y + f x (y-1) + f x (y+1)) : ℚ) / 4

/-- The main theorem: if a grid assignment satisfies the arithmetic mean condition,
    then it is constant across the entire grid -/
theorem grid_constant (f : GridAssignment) (h : is_arithmetic_mean f) :
  ∀ x y x' y' : ℤ, f x y = f x' y' :=
sorry

end grid_constant_l884_88483


namespace min_value_quadratic_sum_l884_88439

theorem min_value_quadratic_sum (x y z : ℝ) (h : x - 2*y + 2*z = 5) :
  ∃ (m : ℝ), m = 36 ∧ ∀ (a b c : ℝ), a - 2*b + 2*c = 5 → (a + 5)^2 + (b - 1)^2 + (c + 3)^2 ≥ m :=
by sorry

end min_value_quadratic_sum_l884_88439


namespace not_p_sufficient_not_necessary_for_not_q_l884_88446

-- Define p and q as predicates on real numbers
def p (x : ℝ) : Prop := x < -1 ∨ x > 1
def q (x : ℝ) : Prop := x < -2 ∨ x > 1

-- Theorem statement
theorem not_p_sufficient_not_necessary_for_not_q :
  (∀ x, ¬(p x) → ¬(q x)) ∧ 
  ¬(∀ x, ¬(q x) → ¬(p x)) :=
sorry

end not_p_sufficient_not_necessary_for_not_q_l884_88446


namespace fraction_proof_l884_88495

theorem fraction_proof (N : ℝ) (F : ℝ) (h1 : N = 8) (h2 : 0.5 * N = F * N + 2) : F = 0.25 := by
  sorry

end fraction_proof_l884_88495


namespace half_times_x_times_three_fourths_l884_88401

theorem half_times_x_times_three_fourths (x : ℚ) : x = 5/6 → (1/2 : ℚ) * x * (3/4 : ℚ) = 5/16 := by
  sorry

end half_times_x_times_three_fourths_l884_88401


namespace second_investment_rate_l884_88419

theorem second_investment_rate
  (total_investment : ℝ)
  (first_rate : ℝ)
  (first_amount : ℝ)
  (total_interest : ℝ)
  (h1 : total_investment = 6000)
  (h2 : first_rate = 0.09)
  (h3 : first_amount = 1800)
  (h4 : total_interest = 624)
  : (total_interest - first_amount * first_rate) / (total_investment - first_amount) = 0.11 := by
  sorry

end second_investment_rate_l884_88419


namespace perfect_square_pairs_l884_88428

theorem perfect_square_pairs (a b : ℤ) : 
  (∃ k : ℤ, a^2 + 4*b = k^2) ∧ (∃ m : ℤ, b^2 + 4*a = m^2) ↔ 
  (a = 0 ∧ b = 0) ∨ 
  (a = -4 ∧ b = -4) ∨ 
  (a = 4 ∧ b = -4) ∨ 
  (∃ k : ℕ, (a = k^2 ∧ b = 0) ∨ (a = 0 ∧ b = k^2)) ∨
  (a = -6 ∧ b = -5) ∨ 
  (a = -5 ∧ b = -6) ∨ 
  (∃ t : ℕ, (a = t ∧ b = 1 - t) ∨ (a = 1 - t ∧ b = t)) :=
sorry

end perfect_square_pairs_l884_88428


namespace tom_walking_distance_l884_88467

/-- Tom's walking rate in miles per minute -/
def walking_rate : ℚ := 2 / 36

/-- The time Tom walks in minutes -/
def walking_time : ℚ := 9

/-- The distance Tom walks in miles -/
def walking_distance : ℚ := walking_rate * walking_time

theorem tom_walking_distance :
  walking_distance = 1/2 := by sorry

end tom_walking_distance_l884_88467


namespace simplify_expression_1_simplify_expression_2_l884_88456

-- Problem 1
theorem simplify_expression_1 (x y : ℝ) : (x - 3*y) - (y - 2*x) = 3*x - 4*y := by
  sorry

-- Problem 2
theorem simplify_expression_2 (a b : ℝ) : 5*a*b^2 - 3*(2*a^2*b - 2*(a^2*b - 2*a*b^2)) = -7*a*b^2 := by
  sorry

end simplify_expression_1_simplify_expression_2_l884_88456


namespace inverse_proportionality_l884_88444

theorem inverse_proportionality (X Y K : ℝ) (h1 : XY = K - 1) (h2 : K > 1) :
  ∃ c : ℝ, ∀ x y : ℝ, (x ≠ 0 ∧ y ≠ 0) → (X = x ∧ Y = y) → x * y = c :=
sorry

end inverse_proportionality_l884_88444


namespace logarithm_simplification_l884_88461

open Real

theorem logarithm_simplification (x y : ℝ) (hx : x > 0) (hy : y > 0) (hy_neq_1 : y ≠ 1) :
  (log x^2 / log y^8) * (log y^5 / log x^4) * (log x^3 / log y^5) * (log y^8 / log x^3) * (log x^4 / log y^3) = 
  (1/3) * (log x / log y) := by
sorry

end logarithm_simplification_l884_88461


namespace prop_A_prop_B_l884_88464

-- Define the function f(x) = (x-2)^2
def f (x : ℝ) : ℝ := (x - 2)^2

-- Proposition A: f(x+2) is an even function
theorem prop_A : ∀ x : ℝ, f (x + 2) = f (-x + 2) := by sorry

-- Proposition B: f(x) is decreasing on (-∞, 2) and increasing on (2, +∞)
theorem prop_B :
  (∀ x y : ℝ, x < y → y < 2 → f y < f x) ∧
  (∀ x y : ℝ, 2 < x → x < y → f x < f y) := by sorry

end prop_A_prop_B_l884_88464


namespace complex_real_condition_l884_88458

theorem complex_real_condition (a : ℝ) :
  (Complex.I * (a - 1) = 0) → a = 1 := by
  sorry

end complex_real_condition_l884_88458


namespace largest_consecutive_sum_105_l884_88406

/-- The sum of the first n positive integers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The sum of n consecutive positive integers starting from a -/
def sum_consecutive (a n : ℕ) : ℕ := n * (2 * a + n - 1) / 2

theorem largest_consecutive_sum_105 :
  (∃ (a : ℕ), a > 0 ∧ sum_consecutive a 14 = 105) ∧
  (∀ (n : ℕ), n > 14 → ¬∃ (a : ℕ), a > 0 ∧ sum_consecutive a n = 105) :=
sorry

end largest_consecutive_sum_105_l884_88406


namespace class_average_weight_l884_88452

theorem class_average_weight (students_a : ℕ) (students_b : ℕ) (avg_weight_a : ℝ) (avg_weight_b : ℝ)
  (h1 : students_a = 36)
  (h2 : students_b = 44)
  (h3 : avg_weight_a = 40)
  (h4 : avg_weight_b = 35) :
  let total_students := students_a + students_b
  let total_weight := students_a * avg_weight_a + students_b * avg_weight_b
  total_weight / total_students = 37.25 := by
  sorry

end class_average_weight_l884_88452


namespace rect_to_cylindrical_l884_88481

/-- Conversion from rectangular to cylindrical coordinates -/
theorem rect_to_cylindrical (x y z : ℝ) :
  x = 2 ∧ y = 2 * Real.sqrt 3 ∧ z = 4 →
  ∃ (r θ : ℝ),
    r = 4 ∧
    θ = π / 3 ∧
    x = r * Real.cos θ ∧
    y = r * Real.sin θ ∧
    z = z :=
by sorry

end rect_to_cylindrical_l884_88481


namespace inequality_system_solutions_l884_88457

theorem inequality_system_solutions : 
  {x : ℕ | 5 * x - 6 ≤ 2 * (x + 3) ∧ (x : ℚ) / 4 - 1 < (x - 2 : ℚ) / 3} = {0, 1, 2, 3, 4} := by
  sorry

end inequality_system_solutions_l884_88457


namespace quadratic_point_relation_l884_88480

/-- The quadratic function f(x) = x^2 - 4x - 3 -/
def f (x : ℝ) : ℝ := x^2 - 4*x - 3

/-- Point A on the graph of f -/
def A : ℝ × ℝ := (-2, f (-2))

/-- Point B on the graph of f -/
def B : ℝ × ℝ := (1, f 1)

/-- Point C on the graph of f -/
def C : ℝ × ℝ := (4, f 4)

theorem quadratic_point_relation :
  A.2 > C.2 ∧ C.2 > B.2 := by sorry

end quadratic_point_relation_l884_88480


namespace max_value_quadratic_swap_l884_88411

/-- Given real numbers a, b, and c where |ax^2 + bx + c| has a maximum value of 1 
    on the interval x ∈ [-1,1], the maximum possible value of |cx^2 + bx + a| 
    on the interval x ∈ [-1,1] is 2. -/
theorem max_value_quadratic_swap (a b c : ℝ) 
  (h : ∀ x ∈ Set.Icc (-1) 1, |a * x^2 + b * x + c| ≤ 1) :
  (⨆ x ∈ Set.Icc (-1) 1, |c * x^2 + b * x + a|) = 2 := by
  sorry

end max_value_quadratic_swap_l884_88411


namespace remaining_pie_portion_l884_88436

-- Define the pie as 100%
def whole_pie : ℚ := 1

-- Carlos's share
def carlos_share : ℚ := 0.6

-- Maria takes half of the remainder
def maria_share_ratio : ℚ := 1/2

-- Theorem to prove
theorem remaining_pie_portion : 
  let remainder_after_carlos := whole_pie - carlos_share
  let maria_share := maria_share_ratio * remainder_after_carlos
  let final_remainder := remainder_after_carlos - maria_share
  final_remainder = 0.2 := by
sorry

end remaining_pie_portion_l884_88436


namespace base_for_five_digit_100_l884_88430

theorem base_for_five_digit_100 :
  ∃! (b : ℕ), b > 1 ∧ b^4 ≤ 100 ∧ 100 < b^5 :=
by sorry

end base_for_five_digit_100_l884_88430


namespace total_length_of_objects_l884_88466

/-- Given the lengths of various objects and their relationships, prove their total length. -/
theorem total_length_of_objects (pencil_length : ℝ) 
  (h1 : pencil_length = 12) 
  (h2 : ∃ pen_length rubber_length, 
    pen_length = rubber_length + 3 ∧ 
    pencil_length = pen_length + 2)
  (h3 : ∃ ruler_length, 
    ruler_length = 3 * rubber_length ∧ 
    ruler_length = pen_length * 1.2)
  (h4 : ∃ marker_length, marker_length = ruler_length / 2)
  (h5 : ∃ scissors_length, scissors_length = pencil_length * 0.75) :
  ∃ total_length, total_length = 69.5 ∧ 
    total_length = rubber_length + pen_length + pencil_length + 
                   marker_length + ruler_length + scissors_length :=
by sorry

end total_length_of_objects_l884_88466


namespace triangle_special_case_l884_88441

-- Define a triangle with sides a, b, c and angles A, B, C
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_special_case (t : Triangle) 
  (h1 : t.A - t.C = π / 2)  -- A - C = 90°
  (h2 : t.a + t.c = Real.sqrt 2 * t.b)  -- a + c = √2 * b
  : t.C = π / 12 := by
  sorry

end triangle_special_case_l884_88441


namespace emily_sees_leo_l884_88468

/-- The time Emily can see Leo given their speeds and distances -/
theorem emily_sees_leo (emily_speed leo_speed : ℝ) (initial_distance final_distance : ℝ) : 
  emily_speed = 15 →
  leo_speed = 10 →
  initial_distance = 0.75 →
  final_distance = 0.6 →
  (initial_distance + final_distance) / (emily_speed - leo_speed) * 60 = 16.2 := by
  sorry

end emily_sees_leo_l884_88468


namespace point_on_number_line_l884_88450

theorem point_on_number_line (a : ℝ) : 
  (∃ (A : ℝ), A = 2 * a + 1 ∧ |A| = 3) → (a = 1 ∨ a = -2) := by
  sorry

end point_on_number_line_l884_88450


namespace quadratic_root_problem_l884_88429

theorem quadratic_root_problem (a b c : ℝ) (h : a * (b - c) ≠ 0) :
  (∀ x, a * (b - c) * x^2 + b * (c - a) * x + c * (a - b) = 0 ↔ x = 1 ∨ x = (c * (a - b)) / (a * (b - c))) :=
by sorry

end quadratic_root_problem_l884_88429


namespace mark_bananas_equal_mike_matt_fruits_l884_88440

/-- Represents the number of fruits each child received -/
structure FruitDistribution where
  mike_oranges : ℕ
  matt_apples : ℕ
  mark_bananas : ℕ

/-- The fruit distribution problem -/
def annie_fruit_problem (fd : FruitDistribution) : Prop :=
  fd.mike_oranges = 3 ∧
  fd.matt_apples = 2 * fd.mike_oranges ∧
  fd.mike_oranges + fd.matt_apples + fd.mark_bananas = 18

/-- The theorem stating the relationship between Mark's bananas and the total fruits of Mike and Matt -/
theorem mark_bananas_equal_mike_matt_fruits (fd : FruitDistribution) 
  (h : annie_fruit_problem fd) : 
  fd.mark_bananas = fd.mike_oranges + fd.matt_apples := by
  sorry

#check mark_bananas_equal_mike_matt_fruits

end mark_bananas_equal_mike_matt_fruits_l884_88440


namespace perfect_square_trinomial_l884_88477

theorem perfect_square_trinomial (a b : ℝ) : a^2 + 6*a*b + 9*b^2 = (a + 3*b)^2 := by
  sorry

end perfect_square_trinomial_l884_88477


namespace cos_sixty_degrees_l884_88462

theorem cos_sixty_degrees : Real.cos (π / 3) = 1 / 2 := by sorry

end cos_sixty_degrees_l884_88462


namespace bridge_length_l884_88409

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time_s : ℝ) :
  train_length = 110 →
  train_speed_kmh = 45 →
  crossing_time_s = 30 →
  (train_speed_kmh * 1000 / 3600 * crossing_time_s) - train_length = 265 := by
  sorry

end bridge_length_l884_88409


namespace total_charge_2_hours_l884_88400

/-- Represents the pricing structure and total charge calculation for a psychologist's therapy sessions. -/
structure TherapyPricing where
  /-- The charge for the first hour of therapy -/
  first_hour_charge : ℕ
  /-- The charge for each additional hour of therapy -/
  additional_hour_charge : ℕ
  /-- The difference between the first hour charge and additional hour charge -/
  charge_difference : first_hour_charge = additional_hour_charge + 25
  /-- The total charge for 5 hours of therapy -/
  total_charge_5_hours : first_hour_charge + 4 * additional_hour_charge = 250

/-- Theorem stating that the total charge for 2 hours of therapy is $115 -/
theorem total_charge_2_hours (p : TherapyPricing) : 
  p.first_hour_charge + p.additional_hour_charge = 115 := by
  sorry


end total_charge_2_hours_l884_88400


namespace score_difference_proof_l884_88434

def score_distribution : List (ℝ × ℝ) := [
  (60, 0.15),
  (75, 0.20),
  (85, 0.25),
  (90, 0.10),
  (100, 0.30)
]

def mean_score : ℝ := (score_distribution.map (fun (score, percent) => score * percent)).sum

def median_score : ℝ := 85

theorem score_difference_proof :
  mean_score - median_score = -0.75 := by sorry

end score_difference_proof_l884_88434


namespace intersection_with_complement_l884_88432

def U : Finset Nat := {1, 2, 3, 4}
def A : Finset Nat := {1, 4}
def B : Finset Nat := {2, 4}

theorem intersection_with_complement : A ∩ (U \ B) = {1} := by sorry

end intersection_with_complement_l884_88432


namespace overall_percentage_l884_88475

theorem overall_percentage (grade1 grade2 grade3 : ℚ) 
  (h1 : grade1 = 50 / 100)
  (h2 : grade2 = 70 / 100)
  (h3 : grade3 = 90 / 100) :
  (grade1 + grade2 + grade3) / 3 = 70 / 100 := by
  sorry

end overall_percentage_l884_88475


namespace two_roots_condition_l884_88499

-- Define the equation
def f (x a : ℝ) : ℝ := 4 * x^2 - 16 * |x| + (2 * a + |x| - x)^2 - 16

-- Define the condition for exactly two distinct roots
def has_two_distinct_roots (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ a = 0 ∧ f x₂ a = 0 ∧
  ∀ x : ℝ, f x a = 0 → x = x₁ ∨ x = x₂

-- State the theorem
theorem two_roots_condition :
  ∀ a : ℝ, has_two_distinct_roots a ↔ (a > -6 ∧ a ≤ -2) ∨ (a > 2 ∧ a < Real.sqrt 8) :=
sorry

end two_roots_condition_l884_88499


namespace range_of_x_l884_88447

theorem range_of_x (x y : ℝ) (h1 : x + y = 1) (h2 : y ≤ 2) : x ≥ -1 := by
  sorry

end range_of_x_l884_88447


namespace sandbox_dimension_ratio_l884_88479

theorem sandbox_dimension_ratio 
  (V₁ V₂ : ℝ) 
  (h₁ : V₁ = 10) 
  (h₂ : V₂ = 80) 
  (k : ℝ) 
  (h₃ : V₂ = k^3 * V₁) : 
  k = 2 := by
  sorry

end sandbox_dimension_ratio_l884_88479


namespace parallel_lines_k_value_l884_88460

/-- Given two points on a line and another line equation, prove the value of k -/
theorem parallel_lines_k_value (k : ℝ) : 
  (∃ (m b : ℝ), (∀ x y : ℝ, y = m * x + b → 
    (x = 3 ∧ y = -12) ∨ (x = k ∧ y = 22)) ∧
   (∀ x y : ℝ, 4 * x + 6 * y = 36 → y = m * x + (36 / 6 - 4 * x / 6))) →
  k = -48 := by
sorry

end parallel_lines_k_value_l884_88460


namespace reginas_earnings_l884_88420

/-- Represents Regina's farm and calculates her earnings -/
def ReginasFarm : ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ :=
  fun num_cows num_pigs num_goats num_chickens num_rabbits
      cow_price pig_price goat_price chicken_price rabbit_price
      milk_income_per_cow rabbit_income_per_year maintenance_cost =>
    let total_animal_sale := num_cows * cow_price + num_pigs * pig_price +
                             num_goats * goat_price + num_chickens * chicken_price +
                             num_rabbits * rabbit_price
    let total_product_income := num_cows * milk_income_per_cow +
                                num_rabbits * rabbit_income_per_year
    total_animal_sale + total_product_income - maintenance_cost

/-- Theorem stating Regina's final earnings -/
theorem reginas_earnings :
  ReginasFarm 20 (4 * 20) ((4 * 20) / 2) (2 * 20) 30
               800 400 600 50 25
               500 10 10000 = 75050 := by
  sorry

end reginas_earnings_l884_88420


namespace counterexample_disproves_conjecture_l884_88423

def isOdd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

def isPrime (p : ℤ) : Prop := p > 1 ∧ ∀ m : ℤ, m > 1 → m < p → ¬(p % m = 0)

def isSumOfThreePrimes (n : ℤ) : Prop :=
  ∃ p q r : ℤ, isPrime p ∧ isPrime q ∧ isPrime r ∧ n = p + q + r

theorem counterexample_disproves_conjecture :
  ∃ n : ℤ, n > 5 ∧ isOdd n ∧ ¬(isSumOfThreePrimes n) →
  ¬(∀ m : ℤ, m > 5 → isOdd m → isSumOfThreePrimes m) :=
sorry

end counterexample_disproves_conjecture_l884_88423


namespace solution_set_of_inequality_l884_88421

-- Define the function representing the sum of distances
def f (x : ℝ) : ℝ := |x - 5| + |x + 1|

-- State the theorem
theorem solution_set_of_inequality :
  {x : ℝ | f x < 8} = Set.Ioo (-2 : ℝ) (6 : ℝ) := by sorry

end solution_set_of_inequality_l884_88421


namespace rotation_of_D_l884_88498

/-- Rotates a point 90 degrees clockwise about the origin -/
def rotate90Clockwise (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, -p.1)

theorem rotation_of_D : 
  let D : ℝ × ℝ := (-3, -8)
  rotate90Clockwise D = (-8, 3) := by
sorry

end rotation_of_D_l884_88498


namespace partnership_investment_ratio_l884_88403

/-- Partnership investment problem -/
theorem partnership_investment_ratio :
  ∀ (x : ℝ) (m : ℝ),
  x > 0 →  -- A's investment is positive
  (12 * x) / (12 * x + 12 * x + 4 * m * x) = 1/3 →  -- A's share proportion
  m = 3 :=  -- Ratio of C's investment to A's
by
  sorry

end partnership_investment_ratio_l884_88403


namespace power_product_eight_l884_88473

theorem power_product_eight (a b : ℕ+) (h : (2 ^ a.val) ^ b.val = 2 ^ 2) :
  2 ^ a.val * 2 ^ b.val = 8 := by
  sorry

end power_product_eight_l884_88473


namespace geometric_sequence_10th_term_l884_88418

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n, a (n + 1) = r * a n

theorem geometric_sequence_10th_term
  (a : ℕ → ℝ)
  (h_geometric : GeometricSequence a)
  (h_4th : a 4 = 16)
  (h_7th : a 7 = 128) :
  a 10 = 1024 := by
sorry

end geometric_sequence_10th_term_l884_88418


namespace simplify_sqrt_product_l884_88437

theorem simplify_sqrt_product : 
  Real.sqrt (3 * 5) * Real.sqrt (5^2 * 3^3) = 45 * Real.sqrt 5 := by
  sorry

end simplify_sqrt_product_l884_88437


namespace exists_counterexample_for_option_c_l884_88414

theorem exists_counterexample_for_option_c (h : ∃ a b : ℝ, a > b ∧ b > 0) :
  ∃ a b : ℝ, a > b ∧ b > 0 ∧ ¬(a > Real.sqrt b) :=
sorry

end exists_counterexample_for_option_c_l884_88414


namespace bag_weight_problem_l884_88416

theorem bag_weight_problem (sugar_weight salt_weight removed_weight : ℕ) 
  (h1 : sugar_weight = 16)
  (h2 : salt_weight = 30)
  (h3 : removed_weight = 4) :
  sugar_weight + salt_weight - removed_weight = 42 := by
  sorry

end bag_weight_problem_l884_88416


namespace chess_tournament_attendees_l884_88405

theorem chess_tournament_attendees (total_students : ℕ) 
  (h1 : total_students = 24) 
  (chess_program_fraction : ℚ) 
  (h2 : chess_program_fraction = 1 / 3) 
  (tournament_fraction : ℚ) 
  (h3 : tournament_fraction = 1 / 2) : ℕ :=
  by
    sorry

#check chess_tournament_attendees

end chess_tournament_attendees_l884_88405


namespace incorrect_equation_l884_88435

/-- A repeating decimal with non-repeating part N and repeating part R -/
structure RepeatingDecimal where
  N : ℕ  -- non-repeating part
  R : ℕ  -- repeating part
  t : ℕ  -- number of digits in N
  u : ℕ  -- number of digits in R
  t_pos : t > 0
  u_pos : u > 0

/-- The value of the repeating decimal -/
noncomputable def RepeatingDecimal.value (M : RepeatingDecimal) : ℝ :=
  (M.N : ℝ) / 10^M.t + (M.R : ℝ) / (10^M.t * (10^M.u - 1))

/-- The theorem stating that the equation in option D is incorrect -/
theorem incorrect_equation (M : RepeatingDecimal) :
  ¬(10^M.t * (10^M.u - 1) * M.value = (M.R : ℝ) * ((M.N : ℝ) - 1)) := by
  sorry

end incorrect_equation_l884_88435


namespace x_squared_plus_2x_is_quadratic_l884_88422

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x^2 + 2x = 0 is a quadratic equation -/
theorem x_squared_plus_2x_is_quadratic :
  is_quadratic_equation (λ x => x^2 + 2*x) :=
by
  sorry


end x_squared_plus_2x_is_quadratic_l884_88422


namespace gcd_462_330_l884_88415

theorem gcd_462_330 : Nat.gcd 462 330 = 66 := by
  sorry

end gcd_462_330_l884_88415


namespace five_double_prime_value_l884_88404

-- Define the prime operation
noncomputable def prime (q : ℝ) : ℝ := 3 * q - 3

-- State the theorem
theorem five_double_prime_value : prime (prime 5) = 33 := by
  sorry

end five_double_prime_value_l884_88404


namespace chuzhou_gdp_scientific_notation_l884_88472

/-- The GDP of Chuzhou City in 2022 in billions of yuan -/
def chuzhou_gdp : ℝ := 3600

/-- Conversion factor from billion to scientific notation -/
def billion_to_scientific : ℝ := 10^9

theorem chuzhou_gdp_scientific_notation :
  chuzhou_gdp * billion_to_scientific = 3.6 * 10^12 := by
  sorry

end chuzhou_gdp_scientific_notation_l884_88472
