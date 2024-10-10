import Mathlib

namespace sum_of_odd_periodic_function_l3325_332546

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_periodic_3 (f : ℝ → ℝ) : Prop := ∀ x, f (x + 3) = f x

theorem sum_of_odd_periodic_function 
  (f : ℝ → ℝ) 
  (h_odd : is_odd_function f) 
  (h_periodic : is_periodic_3 f) 
  (h_value : f (-1) = 1) : 
  f 1 + f 2 + f 3 + f 4 + f 5 + f 6 + f 7 + f 8 + f 9 + f 10 = -1 :=
sorry

end sum_of_odd_periodic_function_l3325_332546


namespace seashells_found_l3325_332587

/-- The number of seashells found by Joan and Jessica -/
theorem seashells_found (joan_shells jessica_shells : ℕ) 
  (h1 : joan_shells = 6) 
  (h2 : jessica_shells = 8) : 
  joan_shells + jessica_shells = 14 := by
  sorry

end seashells_found_l3325_332587


namespace timothy_movie_count_l3325_332573

theorem timothy_movie_count (timothy_prev : ℕ) 
  (h1 : timothy_prev + (timothy_prev + 7) + 2 * (timothy_prev + 7) + timothy_prev / 2 = 129) : 
  timothy_prev = 24 := by
  sorry

end timothy_movie_count_l3325_332573


namespace complex_cube_equation_positive_integer_components_l3325_332580

theorem complex_cube_equation : ∃ (d : ℤ), (1 + 3*I : ℂ)^3 = -26 + d*I := by sorry

theorem positive_integer_components : (1 : ℤ) > 0 ∧ (3 : ℤ) > 0 := by sorry

end complex_cube_equation_positive_integer_components_l3325_332580


namespace grid_sum_l3325_332549

theorem grid_sum (p q r s : ℕ+) 
  (h_pq : p * q = 6)
  (h_rs : r * s = 8)
  (h_pr : p * r = 4)
  (h_qs : q * s = 12)
  (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s) :
  p + q + r + s = 13 := by
  sorry

end grid_sum_l3325_332549


namespace local_max_range_l3325_332540

-- Define the function f and its derivative
def f : ℝ → ℝ := sorry

-- Define the derivative of f
def f_deriv (a : ℝ) (x : ℝ) : ℝ := a * (x + 1) * (x - a)

-- State the theorem
theorem local_max_range (a : ℝ) :
  (∀ x, HasDerivAt f (f_deriv a x) x) →  -- f' is the derivative of f
  (∃ δ > 0, ∀ x, x ≠ a → |x - a| < δ → f x ≤ f a) →  -- local maximum at x = a
  -1 < a ∧ a < 0 :=
sorry

end local_max_range_l3325_332540


namespace m_condition_necessary_not_sufficient_l3325_332516

-- Define the condition for m
def m_condition (m : ℝ) : Prop := 2 < m ∧ m < 6

-- Define the condition for the equation to represent an ellipse
def is_ellipse (m : ℝ) : Prop := m > 2 ∧ m < 6 ∧ m ≠ 4

-- Theorem stating that m_condition is necessary but not sufficient for is_ellipse
theorem m_condition_necessary_not_sufficient :
  (∀ m : ℝ, is_ellipse m → m_condition m) ∧
  ¬(∀ m : ℝ, m_condition m → is_ellipse m) := by sorry

end m_condition_necessary_not_sufficient_l3325_332516


namespace pool_length_is_ten_l3325_332590

/-- Represents a rectangular swimming pool with a surrounding deck -/
structure PoolWithDeck where
  poolLength : ℝ
  poolWidth : ℝ
  deckWidth : ℝ

/-- Calculates the total area of the pool and deck -/
def totalArea (p : PoolWithDeck) : ℝ :=
  (p.poolLength + 2 * p.deckWidth) * (p.poolWidth + 2 * p.deckWidth)

/-- Theorem: The length of the pool is 10 feet given the specified conditions -/
theorem pool_length_is_ten :
  ∃ (p : PoolWithDeck),
    p.poolWidth = 12 ∧
    p.deckWidth = 4 ∧
    totalArea p = 360 ∧
    p.poolLength = 10 := by
  sorry

end pool_length_is_ten_l3325_332590


namespace inscribed_circle_radius_l3325_332531

theorem inscribed_circle_radius (a b c : ℝ) (h₁ : a = 5) (h₂ : b = 6) (h₃ : c = 7) :
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  area / s = 2 * Real.sqrt 6 / 3 := by sorry

end inscribed_circle_radius_l3325_332531


namespace divergent_series_with_convergent_min_series_l3325_332553

theorem divergent_series_with_convergent_min_series :
  ∃ (x : ℕ → ℝ), 
    (∀ n, x n > 0) ∧ 
    (∀ n, x (n + 1) < x n) ∧ 
    (¬ Summable x) ∧
    (Summable (fun n => min (x (n + 1)) (1 / ((n + 1 : ℝ) * Real.log (n + 1))))) := by
  sorry

end divergent_series_with_convergent_min_series_l3325_332553


namespace repeating_decimal_sum_l3325_332593

theorem repeating_decimal_sum : 
  (∃ (x y z : ℚ), 
    (1000 * x - x = 123) ∧ 
    (10000 * y - y = 4567) ∧ 
    (100 * z - z = 89) ∧ 
    (x + y + z = 14786 / 9999)) := by sorry

end repeating_decimal_sum_l3325_332593


namespace storks_on_fence_l3325_332550

/-- The number of storks initially on the fence -/
def initial_storks : ℕ := 4

/-- The number of birds initially on the fence -/
def initial_birds : ℕ := 3

/-- The number of additional storks that joined -/
def additional_storks : ℕ := 6

/-- The total number of birds and storks after additional storks joined -/
def total_after : ℕ := 13

theorem storks_on_fence :
  initial_birds + initial_storks + additional_storks = total_after :=
by sorry

end storks_on_fence_l3325_332550


namespace digit_sum_puzzle_l3325_332586

theorem digit_sum_puzzle (c o u n t s : ℕ) : 
  c ≠ 0 → o ≠ 0 → u ≠ 0 → n ≠ 0 → t ≠ 0 → s ≠ 0 →
  c + o = u →
  u + n = t →
  t + c = s →
  o + n + s = 18 →
  t = 9 := by
sorry

end digit_sum_puzzle_l3325_332586


namespace alice_probability_after_two_turns_l3325_332524

-- Define the game parameters
def alice_toss_prob : ℚ := 1/2
def alice_keep_prob : ℚ := 1/2
def bob_toss_prob : ℚ := 2/5
def bob_keep_prob : ℚ := 3/5

-- Define the probability that Alice has the ball after two turns
def alice_has_ball_after_two_turns : ℚ := 
  alice_toss_prob * bob_toss_prob + alice_keep_prob * alice_keep_prob

-- Theorem statement
theorem alice_probability_after_two_turns : 
  alice_has_ball_after_two_turns = 9/20 := by sorry

end alice_probability_after_two_turns_l3325_332524


namespace coefficient_of_x_six_in_expansion_l3325_332552

theorem coefficient_of_x_six_in_expansion (x : ℝ) : 
  ∃ (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ), 
    (2*x^2 + 1)^5 = a₀ + a₁*x^2 + a₂*x^4 + a₃*x^6 + a₄*x^8 + a₅*x^10 ∧ 
    a₃ = 80 := by
  sorry

end coefficient_of_x_six_in_expansion_l3325_332552


namespace round_trip_speed_l3325_332530

/-- Proves that given specific conditions for a round trip, the outward speed is 3 km/hr -/
theorem round_trip_speed (return_speed : ℝ) (total_time : ℝ) (one_way_distance : ℝ)
  (h1 : return_speed = 2)
  (h2 : total_time = 5)
  (h3 : one_way_distance = 6) :
  (one_way_distance / (total_time - one_way_distance / return_speed) = 3) :=
by sorry

end round_trip_speed_l3325_332530


namespace gcd_9157_2695_l3325_332551

theorem gcd_9157_2695 : Nat.gcd 9157 2695 = 1 := by
  sorry

end gcd_9157_2695_l3325_332551


namespace last_locker_opened_l3325_332548

/-- Represents the locker opening pattern described in the problem -/
def lockerOpeningPattern (n : ℕ) : Prop :=
  ∃ (lastLocker : ℕ),
    -- There are 2048 lockers
    n = 2048 ∧
    -- The last locker opened is 2041
    lastLocker = 2041 ∧
    -- The pattern follows the described rules
    (∀ k : ℕ, k ≤ n → ∃ (trip : ℕ),
      -- Each trip opens lockers based on the trip number
      (k % trip = 0 → k ≠ lastLocker) ∧
      -- The last locker is only opened in the final trip
      (k = lastLocker → ∀ j < trip, k % j ≠ 0))

/-- Theorem stating that the last locker opened is 2041 -/
theorem last_locker_opened (n : ℕ) (h : lockerOpeningPattern n) :
  ∃ (lastLocker : ℕ), lastLocker = 2041 ∧ 
  ∀ k : ℕ, k ≤ n → k ≠ lastLocker → ∃ (trip : ℕ), k % trip = 0 :=
  sorry

end last_locker_opened_l3325_332548


namespace sum_of_reciprocals_l3325_332544

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 10) (h2 : x * y = 20) :
  1 / x + 1 / y = 1 / 2 := by
  sorry

end sum_of_reciprocals_l3325_332544


namespace terminating_decimal_fractions_l3325_332578

theorem terminating_decimal_fractions (n : ℕ) : n > 1 → (∃ (a b c d : ℕ), 1 / n = a / (2^b * 5^c) ∧ 1 / (n + 1) = d / (2^b * 5^c)) ↔ n = 4 := by
  sorry

end terminating_decimal_fractions_l3325_332578


namespace decimal_expansion_of_three_sevenths_l3325_332571

/-- The length of the smallest repeating block in the decimal expansion of 3/7 -/
def repeatingBlockLength : ℕ := 6

/-- The fraction we're considering -/
def fraction : ℚ := 3/7

theorem decimal_expansion_of_three_sevenths :
  ∃ (d : ℕ → ℕ) (n : ℕ),
    (∀ k, d k < 10) ∧
    (∀ k, d (k + n) = d k) ∧
    (∀ m, m < n → ∃ k, d (k + m) ≠ d k) ∧
    fraction = ∑' k, (d k : ℚ) / 10^(k + 1) ∧
    n = repeatingBlockLength :=
sorry

end decimal_expansion_of_three_sevenths_l3325_332571


namespace smallest_cube_with_divisor_l3325_332515

theorem smallest_cube_with_divisor (p q r : ℕ) : 
  Prime p → Prime q → Prime r → p ≠ q → p ≠ r → q ≠ r →
  (∀ m : ℕ, m < (p * q * r^2)^3 → ¬(∃ k : ℕ, m = k^3 ∧ p^2 * q^3 * r^5 ∣ m)) →
  (p * q * r^2)^3 = (p * q * r^2)^3 ∧ p^2 * q^3 * r^5 ∣ (p * q * r^2)^3 := by
  sorry

#check smallest_cube_with_divisor

end smallest_cube_with_divisor_l3325_332515


namespace max_digits_product_5_4_l3325_332505

theorem max_digits_product_5_4 : 
  ∃ (a b : ℕ), 
    10000 ≤ a ∧ a < 100000 ∧
    1000 ≤ b ∧ b < 10000 ∧
    (∀ (x y : ℕ), 
      10000 ≤ x ∧ x < 100000 ∧ 
      1000 ≤ y ∧ y < 10000 → 
      x * y < 1000000000) ∧
    999999999 < a * b :=
by sorry

end max_digits_product_5_4_l3325_332505


namespace cloth_selling_price_l3325_332527

/-- Calculates the total selling price of cloth given the length, profit per meter, and cost price per meter. -/
def total_selling_price (length : ℕ) (profit_per_meter : ℕ) (cost_per_meter : ℕ) : ℕ :=
  length * (profit_per_meter + cost_per_meter)

/-- Theorem stating that the total selling price of 85 meters of cloth with a profit of Rs. 5 per meter and a cost price of Rs. 100 per meter is Rs. 8925. -/
theorem cloth_selling_price :
  total_selling_price 85 5 100 = 8925 := by
  sorry

end cloth_selling_price_l3325_332527


namespace regions_bound_l3325_332535

/-- Represents a plane in 3D space -/
structure Plane3D where
  -- Add necessary fields to define a plane

/-- The number of regions formed by three planes in 3D space -/
def num_regions (p1 p2 p3 : Plane3D) : ℕ :=
  sorry

theorem regions_bound (p1 p2 p3 : Plane3D) :
  4 ≤ num_regions p1 p2 p3 ∧ num_regions p1 p2 p3 ≤ 8 := by
  sorry

end regions_bound_l3325_332535


namespace total_spent_on_flowers_l3325_332509

def roses_quantity : ℕ := 5
def roses_price : ℕ := 6
def daisies_quantity : ℕ := 3
def daisies_price : ℕ := 4
def tulips_quantity : ℕ := 2
def tulips_price : ℕ := 5

theorem total_spent_on_flowers :
  roses_quantity * roses_price +
  daisies_quantity * daisies_price +
  tulips_quantity * tulips_price = 52 := by
sorry

end total_spent_on_flowers_l3325_332509


namespace shaded_area_square_with_circles_l3325_332506

/-- The area of the shaded region in a square with side length 6 and inscribed circles
    of radius 2√3 at each corner is equal to 36 - 12√3 - 4π. -/
theorem shaded_area_square_with_circles (square_side : ℝ) (circle_radius : ℝ)
  (h_side : square_side = 6)
  (h_radius : circle_radius = 2 * Real.sqrt 3) :
  let total_area := square_side ^ 2
  let triangle_area := 8 * (1 / 2 * (square_side / 2) * circle_radius)
  let sector_area := 4 * (1 / 12 * π * circle_radius ^ 2)
  total_area - triangle_area - sector_area = 36 - 12 * Real.sqrt 3 - 4 * π :=
by sorry

end shaded_area_square_with_circles_l3325_332506


namespace tank_capacity_l3325_332581

theorem tank_capacity (initial_fill : Real) (added_amount : Real) (final_fill : Real) :
  initial_fill = 3/4 →
  added_amount = 4 →
  final_fill = 9/10 →
  ∃ (capacity : Real), capacity = 80/3 ∧
    initial_fill * capacity + added_amount = final_fill * capacity :=
by sorry

end tank_capacity_l3325_332581


namespace chess_match_average_time_l3325_332534

/-- Proves that in a chess match with given conditions, one player's average move time is 28 seconds -/
theorem chess_match_average_time (total_moves : ℕ) (opponent_avg_time : ℕ) (match_duration : ℕ) :
  total_moves = 30 →
  opponent_avg_time = 40 →
  match_duration = 17 * 60 →
  ∃ (player_avg_time : ℕ), player_avg_time = 28 ∧ 
    (total_moves / 2) * (player_avg_time + opponent_avg_time) = match_duration := by
  sorry

end chess_match_average_time_l3325_332534


namespace complex_expression_equality_l3325_332519

theorem complex_expression_equality : 
  let a : ℂ := 3 - 2*I
  let b : ℂ := -2 + 3*I
  3*a + 4*b = 1 + 6*I :=
by sorry

end complex_expression_equality_l3325_332519


namespace largest_awesome_prime_l3325_332584

def is_awesome_prime (p : ℕ) : Prop :=
  Nat.Prime p ∧ ∀ q : ℕ, 0 < q → q < p → Nat.Prime (p + 2 * q)

theorem largest_awesome_prime : 
  (∃ p : ℕ, is_awesome_prime p) ∧ 
  (∀ p : ℕ, is_awesome_prime p → p ≤ 3) :=
sorry

end largest_awesome_prime_l3325_332584


namespace second_class_males_count_l3325_332502

/-- Represents the number of students in each class by gender -/
structure ClassComposition where
  males : ℕ
  females : ℕ

/-- Represents the composition of the three classes -/
structure SquareDancingClasses where
  class1 : ClassComposition
  class2 : ClassComposition
  class3 : ClassComposition

def total_males (classes : SquareDancingClasses) : ℕ :=
  classes.class1.males + classes.class2.males + classes.class3.males

def total_females (classes : SquareDancingClasses) : ℕ :=
  classes.class1.females + classes.class2.females + classes.class3.females

theorem second_class_males_count 
  (classes : SquareDancingClasses)
  (h1 : classes.class1 = ⟨17, 13⟩)
  (h2 : classes.class2.females = 18)
  (h3 : classes.class3 = ⟨15, 17⟩)
  (h4 : total_males classes - total_females classes = 2) :
  classes.class2.males = 18 :=
sorry

end second_class_males_count_l3325_332502


namespace rotating_squares_intersection_area_l3325_332569

/-- The area of intersection of two rotating unit squares after 5 minutes -/
theorem rotating_squares_intersection_area : 
  let revolution_rate : ℝ := 2 * Real.pi / 60 -- radians per minute
  let rotation_time : ℝ := 5 -- minutes
  let rotation_angle : ℝ := revolution_rate * rotation_time
  let intersection_area : ℝ := (1 - Real.cos rotation_angle) * (1 - Real.sin rotation_angle)
  intersection_area = (2 - Real.sqrt 3) / 4 := by
sorry


end rotating_squares_intersection_area_l3325_332569


namespace opposite_of_2023_l3325_332577

theorem opposite_of_2023 : 
  ∀ x : ℤ, x + 2023 = 0 ↔ x = -2023 := by sorry

end opposite_of_2023_l3325_332577


namespace nonreal_roots_product_l3325_332520

theorem nonreal_roots_product (x : ℂ) : 
  (x^4 - 6*x^3 + 15*x^2 - 20*x = 396) →
  (∃ z w : ℂ, z ≠ w ∧ z.im ≠ 0 ∧ w.im ≠ 0 ∧ 
   (x^4 - 6*x^3 + 15*x^2 - 20*x - 396 = 0 → x = z ∨ x = w) ∧
   z * w = 4 + Real.sqrt 412) :=
by sorry

end nonreal_roots_product_l3325_332520


namespace distance_proof_l3325_332512

theorem distance_proof (v1 v2 : ℝ) : 
  (5 * v1 + 5 * v2 = 30) →
  (3 * (v1 + 2) + 3 * (v2 + 2) = 30) →
  30 = 30 := by
  sorry

end distance_proof_l3325_332512


namespace max_consecutive_interesting_l3325_332561

def is_interesting (n : ℕ) : Prop :=
  ∃ p q : ℕ, Prime p ∧ Prime q ∧ n = p * q

theorem max_consecutive_interesting :
  (∃ a : ℕ, ∀ k : ℕ, k < 3 → is_interesting (a + k)) ∧
  (∀ a : ℕ, ∃ k : ℕ, k < 4 → ¬is_interesting (a + k)) :=
sorry

end max_consecutive_interesting_l3325_332561


namespace olympic_mascots_arrangement_l3325_332517

/-- The number of possible arrangements of 5 items with specific constraints -/
def num_arrangements : ℕ := 16

/-- The number of ways to choose 1 item from 2 -/
def choose_one_from_two : ℕ := 2

/-- The number of ways to arrange 2 items -/
def arrange_two : ℕ := 2

theorem olympic_mascots_arrangement :
  num_arrangements = 2 * choose_one_from_two * choose_one_from_two * arrange_two :=
sorry

end olympic_mascots_arrangement_l3325_332517


namespace ellipse_sum_l3325_332560

/-- Theorem: For an ellipse with center (-3, 1), horizontal semi-major axis length 4,
    and vertical semi-minor axis length 2, the sum of h, k, a, and c is equal to 4. -/
theorem ellipse_sum (h k a c : ℝ) : 
  h = -3 ∧ k = 1 ∧ a = 4 ∧ c = 2 → h + k + a + c = 4 := by
  sorry

end ellipse_sum_l3325_332560


namespace right_triangle_area_l3325_332564

theorem right_triangle_area (hypotenuse : ℝ) (angle : ℝ) :
  hypotenuse = 12 →
  angle = 30 * π / 180 →
  let shortest_side := hypotenuse / 2
  let longest_side := hypotenuse / 2 * Real.sqrt 3
  let area := shortest_side * longest_side / 2
  area = 18 * Real.sqrt 3 := by sorry

end right_triangle_area_l3325_332564


namespace square_perimeter_l3325_332556

theorem square_perimeter (side_length : ℝ) (h : side_length = 40) :
  4 * side_length = 160 :=
by sorry

end square_perimeter_l3325_332556


namespace expression_evaluation_l3325_332555

theorem expression_evaluation (x : ℤ) 
  (h1 : 1 - x > (-1 - x) / 2) 
  (h2 : x + 1 > 0) 
  (h3 : x ≠ 1) 
  (h4 : x ≠ 0) : 
  (1 + (3*x - 1) / (x + 1)) / (x / (x^2 - 1)) = 4 := by
  sorry

end expression_evaluation_l3325_332555


namespace quadratic_real_root_l3325_332567

/-- A quadratic equation x^2 + bx + 25 = 0 has at least one real root if and only if b ∈ (-∞, -10] ∪ [10, ∞) -/
theorem quadratic_real_root (b : ℝ) : 
  (∃ x : ℝ, x^2 + b*x + 25 = 0) ↔ b ≤ -10 ∨ b ≥ 10 := by
  sorry

end quadratic_real_root_l3325_332567


namespace conor_eggplants_per_day_l3325_332510

/-- The number of eggplants Conor can chop in a day -/
def eggplants_per_day : ℕ := sorry

/-- The number of carrots Conor can chop in a day -/
def carrots_per_day : ℕ := 9

/-- The number of potatoes Conor can chop in a day -/
def potatoes_per_day : ℕ := 8

/-- The number of days Conor works per week -/
def work_days_per_week : ℕ := 4

/-- The total number of vegetables Conor chops in a week -/
def total_vegetables_per_week : ℕ := 116

theorem conor_eggplants_per_day :
  eggplants_per_day = 12 :=
by sorry

end conor_eggplants_per_day_l3325_332510


namespace range_of_a_l3325_332542

-- Define propositions p and q
def p (x : ℝ) : Prop := (4*x - 3)^2 ≤ 1
def q (x a : ℝ) : Prop := x^2 - (2*a + 1)*x + a*(a + 1) ≤ 0

-- Define the set A
def A : Set ℝ := {x | p x}

-- Define the set B
def B (a : ℝ) : Set ℝ := {x | q x a}

-- Define the condition that ¬p is necessary but not sufficient for ¬q
def condition (a : ℝ) : Prop :=
  (∀ x, ¬(p x) → ¬(q x a)) ∧ 
  (∃ x, ¬(p x) ∧ (q x a))

-- Theorem statement
theorem range_of_a :
  ∀ a : ℝ, condition a ↔ (0 ≤ a ∧ a ≤ 1/2) :=
sorry

end range_of_a_l3325_332542


namespace orthogonal_circles_product_l3325_332545

theorem orthogonal_circles_product (x y u v : ℝ) 
  (h1 : x^2 + y^2 = 1)
  (h2 : u^2 + v^2 = 1)
  (h3 : x*u + y*v = 0) :
  x*y + u*v = 0 := by
  sorry

end orthogonal_circles_product_l3325_332545


namespace value_of_a_minus_b_l3325_332575

theorem value_of_a_minus_b (a b c : ℝ) 
  (h1 : a - (b - 2*c) = 19) 
  (h2 : a - b - 2*c = 7) : 
  a - b = 13 := by
sorry

end value_of_a_minus_b_l3325_332575


namespace triangle_properties_l3325_332513

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  ha : a > 0
  hb : b > 0
  hc : c > 0
  hA : 0 < A ∧ A < π
  hB : 0 < B ∧ B < π
  hC : 0 < C ∧ C < π
  sum_angles : A + B + C = π

/-- The vector m in the problem -/
def m (t : Triangle) : ℝ × ℝ := (t.a + t.c, t.b - t.a)

/-- The vector n in the problem -/
def n (t : Triangle) : ℝ × ℝ := (t.a - t.c, t.b)

/-- The dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem triangle_properties (t : Triangle) 
  (h_perp : dot_product (m t) (n t) = 0)
  (h_sin : 2 * Real.sin (t.A / 2) ^ 2 + 2 * Real.sin (t.B / 2) ^ 2 = 1) :
  t.C = π / 3 ∧ t.A = π / 3 ∧ t.B = π / 3 := by
  sorry

end triangle_properties_l3325_332513


namespace matrix_equation_solution_l3325_332582

theorem matrix_equation_solution :
  let N : Matrix (Fin 2) (Fin 2) ℝ := !![2, 4; 1, 4]
  N^4 - 5 • N^3 + 9 • N^2 - 5 • N = !![6, 12; 3, 6] := by sorry

end matrix_equation_solution_l3325_332582


namespace unique_triplet_solution_l3325_332558

theorem unique_triplet_solution :
  ∀ (x y ℓ : ℕ), x^3 + y^3 - 53 = 7^ℓ ↔ x = 3 ∧ y = 3 ∧ ℓ = 0 := by
  sorry

end unique_triplet_solution_l3325_332558


namespace sum_of_solutions_l3325_332574

theorem sum_of_solutions (x₁ x₂ : ℝ) : 
  (2 * x₁^2 - 7 * x₁ - 9 = 0) → 
  (2 * x₂^2 - 7 * x₂ - 9 = 0) → 
  (x₁ ≠ x₂) →
  (x₁ + x₂ = 7/2) := by
sorry

end sum_of_solutions_l3325_332574


namespace joe_caramel_probability_l3325_332503

/-- Represents the set of candies in Joe's pocket -/
structure CandySet :=
  (lemon : ℕ)
  (caramel : ℕ)

/-- Calculates the probability of selecting a caramel-flavored candy -/
def probability_caramel (cs : CandySet) : ℚ :=
  cs.caramel / (cs.lemon + cs.caramel)

/-- Theorem stating that the probability of selecting a caramel-flavored candy
    from Joe's set is 3/7 -/
theorem joe_caramel_probability :
  let joe_candies : CandySet := { lemon := 4, caramel := 3 }
  probability_caramel joe_candies = 3 / 7 := by
  sorry


end joe_caramel_probability_l3325_332503


namespace households_with_appliances_l3325_332508

theorem households_with_appliances (total : ℕ) (tv : ℕ) (fridge : ℕ) (both : ℕ) :
  total = 100 →
  tv = 65 →
  fridge = 84 →
  both = 53 →
  tv + fridge - both = 96 := by
  sorry

end households_with_appliances_l3325_332508


namespace fourth_quadrant_properties_l3325_332562

open Real

-- Define the fourth quadrant
def fourth_quadrant (α : ℝ) : Prop := 3 * π / 2 < α ∧ α < 2 * π

theorem fourth_quadrant_properties (α : ℝ) (h : fourth_quadrant α) :
  (∃ α, fourth_quadrant α ∧ cos (2 * α) > 0) ∧
  (∀ α, fourth_quadrant α → sin (2 * α) < 0) ∧
  (¬ ∃ α, fourth_quadrant α ∧ tan (α / 2) < 0) ∧
  (∃ α, fourth_quadrant α ∧ cos (α / 2) < 0) :=
by sorry

end fourth_quadrant_properties_l3325_332562


namespace total_envelopes_l3325_332572

/-- The number of stamps needed for an envelope weighing more than 5 pounds -/
def heavy_envelope_stamps : ℕ := 5

/-- The number of stamps needed for an envelope weighing less than 5 pounds -/
def light_envelope_stamps : ℕ := 2

/-- The total number of stamps Micah bought -/
def total_stamps : ℕ := 52

/-- The number of envelopes weighing less than 5 pounds -/
def light_envelopes : ℕ := 6

/-- Theorem stating the total number of envelopes Micah bought -/
theorem total_envelopes : 
  ∃ (heavy_envelopes : ℕ), 
    light_envelopes * light_envelope_stamps + 
    heavy_envelopes * heavy_envelope_stamps = total_stamps ∧
    light_envelopes + heavy_envelopes = 14 := by
  sorry

end total_envelopes_l3325_332572


namespace vector_dot_product_and_trigonometry_l3325_332529

/-- Given vectors a and b, and a function f, prove the following statements. -/
theorem vector_dot_product_and_trigonometry 
  (a : ℝ × ℝ) 
  (b : ℝ → ℝ × ℝ) 
  (f : ℝ → ℝ) 
  (h_a : a = (Real.sqrt 3, 1))
  (h_b : ∀ x, b x = (Real.cos x, Real.sin x))
  (h_f : ∀ x, f x = a.1 * (b x).1 + a.2 * (b x).2)
  (h_x : ∀ x, 0 < x ∧ x < Real.pi)
  (α : ℝ)
  (h_α : f α = 2 * Real.sqrt 2 / 3) :
  (∃ x, a.1 * (b x).1 + a.2 * (b x).2 = 0 → x = 2 * Real.pi / 3) ∧ 
  Real.sin (2 * α + Real.pi / 6) = -5 / 9 := by
  sorry

end vector_dot_product_and_trigonometry_l3325_332529


namespace inequality_proof_l3325_332597

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 / (b + c)) + (b^2 / (c + a)) + (c^2 / (a + b)) ≥ (a + b + c) / 2 := by
  sorry

end inequality_proof_l3325_332597


namespace optimal_game_outcome_l3325_332568

/-- Represents a player in the game -/
inductive Player
| First
| Second

/-- Represents a strategy for a player -/
def Strategy := List ℤ → ℤ

/-- The game state, including the current sum and remaining numbers -/
structure GameState :=
  (sum : ℤ)
  (remaining : List ℤ)

/-- The result of playing the game with given strategies -/
def playGame (firstStrategy : Strategy) (secondStrategy : Strategy) : ℤ :=
  sorry

/-- An optimal strategy for the first player -/
def optimalFirstStrategy : Strategy :=
  sorry

/-- An optimal strategy for the second player -/
def optimalSecondStrategy : Strategy :=
  sorry

/-- The theorem stating the optimal outcome of the game -/
theorem optimal_game_outcome :
  playGame optimalFirstStrategy optimalSecondStrategy = 30 :=
sorry

end optimal_game_outcome_l3325_332568


namespace fermat_5_divisible_by_641_fermat_numbers_coprime_l3325_332525

-- Define Fermat numbers
def F (n : ℕ) : ℕ := 2^(2^n) + 1

-- Theorem 1: F_5 is divisible by 641
theorem fermat_5_divisible_by_641 : 
  641 ∣ F 5 := by sorry

-- Theorem 2: F_k and F_n are relatively prime for k ≠ n
theorem fermat_numbers_coprime {k n : ℕ} (h : k ≠ n) : 
  Nat.gcd (F k) (F n) = 1 := by sorry

end fermat_5_divisible_by_641_fermat_numbers_coprime_l3325_332525


namespace calf_cost_l3325_332514

/-- Given a cow and a calf where the total cost is $990 and the cow costs 8 times as much as the calf, 
    the cost of the calf is $110. -/
theorem calf_cost (total_cost : ℕ) (cow_calf_ratio : ℕ) (calf_cost : ℕ) : 
  total_cost = 990 → 
  cow_calf_ratio = 8 → 
  calf_cost + cow_calf_ratio * calf_cost = total_cost → 
  calf_cost = 110 := by
  sorry

end calf_cost_l3325_332514


namespace funfair_unsold_tickets_l3325_332595

/-- Calculates the number of unsold tickets at a school funfair --/
theorem funfair_unsold_tickets (total_rolls : ℕ) (tickets_per_roll : ℕ)
  (fourth_grade_percent : ℚ) (fifth_grade_percent : ℚ) (sixth_grade_percent : ℚ)
  (seventh_grade_percent : ℚ) (eighth_grade_percent : ℚ) (ninth_grade_tickets : ℕ) :
  total_rolls = 50 →
  tickets_per_roll = 250 →
  fourth_grade_percent = 30 / 100 →
  fifth_grade_percent = 40 / 100 →
  sixth_grade_percent = 25 / 100 →
  seventh_grade_percent = 35 / 100 →
  eighth_grade_percent = 20 / 100 →
  ninth_grade_tickets = 150 →
  ∃ (unsold : ℕ), unsold = 1898 := by
  sorry

#check funfair_unsold_tickets

end funfair_unsold_tickets_l3325_332595


namespace window_side_length_l3325_332585

/-- Represents the dimensions of a window pane -/
structure Pane where
  width : ℝ
  height : ℝ
  ratio : height = 3 * width

/-- Represents the dimensions of a square window -/
structure SquareWindow where
  pane : Pane
  borderWidth : ℝ
  sideLength : ℝ
  paneCount : ℕ
  isSquare : sideLength = 3 * pane.width + 4 * borderWidth
  hasPanes : paneCount = 9

/-- Theorem: The side length of the square window is 20 inches -/
theorem window_side_length (w : SquareWindow) (h : w.borderWidth = 2) : w.sideLength = 20 :=
by sorry

end window_side_length_l3325_332585


namespace sum_of_polynomials_l3325_332511

-- Define the polynomials
def f (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def g (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def h (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2
def j (x : ℝ) : ℝ := x^2 - x - 3

-- State the theorem
theorem sum_of_polynomials (x : ℝ) : 
  f x + g x + h x + j x = -3 * x^2 + 11 * x - 15 := by
  sorry

end sum_of_polynomials_l3325_332511


namespace family_eye_count_l3325_332536

-- Define the family members and their eye counts
def mother_eyes : ℕ := 1
def father_eyes : ℕ := 3
def num_children : ℕ := 3
def eyes_per_child : ℕ := 4

-- Theorem statement
theorem family_eye_count :
  mother_eyes + father_eyes + num_children * eyes_per_child = 16 :=
by sorry

end family_eye_count_l3325_332536


namespace emily_has_ten_employees_l3325_332570

/-- Calculates the number of employees Emily has based on salary information. -/
def calculate_employees (emily_original_salary : ℕ) (emily_new_salary : ℕ) 
                        (employee_original_salary : ℕ) (employee_new_salary : ℕ) : ℕ :=
  (emily_original_salary - emily_new_salary) / (employee_new_salary - employee_original_salary)

/-- Proves that Emily has 10 employees given the salary information. -/
theorem emily_has_ten_employees :
  calculate_employees 1000000 850000 20000 35000 = 10 := by
  sorry

end emily_has_ten_employees_l3325_332570


namespace no_convex_polygon_with_1974_diagonals_l3325_332576

theorem no_convex_polygon_with_1974_diagonals :
  ¬ ∃ (N : ℕ), N > 0 ∧ N * (N - 3) / 2 = 1974 := by
  sorry

end no_convex_polygon_with_1974_diagonals_l3325_332576


namespace smallest_meeting_time_l3325_332523

/-- The number of horses -/
def num_horses : ℕ := 8

/-- The time taken by horse k to complete one lap -/
def lap_time (k : ℕ) : ℕ := k^2

/-- Predicate to check if a time t is when at least 4 horses are at the starting point -/
def at_least_four_horses_meet (t : ℕ) : Prop :=
  ∃ (h1 h2 h3 h4 : ℕ), 
    h1 ≤ num_horses ∧ h2 ≤ num_horses ∧ h3 ≤ num_horses ∧ h4 ≤ num_horses ∧
    h1 ≠ h2 ∧ h1 ≠ h3 ∧ h1 ≠ h4 ∧ h2 ≠ h3 ∧ h2 ≠ h4 ∧ h3 ≠ h4 ∧
    t % (lap_time h1) = 0 ∧ t % (lap_time h2) = 0 ∧ t % (lap_time h3) = 0 ∧ t % (lap_time h4) = 0

/-- The smallest positive time when at least 4 horses meet at the starting point -/
def S : ℕ := 144

theorem smallest_meeting_time : 
  (S > 0) ∧ 
  at_least_four_horses_meet S ∧ 
  ∀ t, 0 < t ∧ t < S → ¬(at_least_four_horses_meet t) :=
by sorry

end smallest_meeting_time_l3325_332523


namespace largest_integer_less_than_150_with_remainder_2_mod_9_l3325_332566

theorem largest_integer_less_than_150_with_remainder_2_mod_9 : ∃ n : ℕ, n < 150 ∧ n % 9 = 2 ∧ ∀ m : ℕ, m < 150 ∧ m % 9 = 2 → m ≤ n :=
by sorry

end largest_integer_less_than_150_with_remainder_2_mod_9_l3325_332566


namespace intersection_point_correct_l3325_332504

/-- The line equation y = x + 3 -/
def line_equation (x y : ℝ) : Prop := y = x + 3

/-- A point lies on the y-axis if its x-coordinate is 0 -/
def on_y_axis (x y : ℝ) : Prop := x = 0

/-- The intersection point of the line y = x + 3 and the y-axis -/
def intersection_point : ℝ × ℝ := (0, 3)

theorem intersection_point_correct :
  let (x, y) := intersection_point
  line_equation x y ∧ on_y_axis x y :=
by sorry

end intersection_point_correct_l3325_332504


namespace part_one_part_two_part_three_l3325_332583

-- Part 1
theorem part_one : Real.sqrt 16 + (1 - Real.sqrt 3) ^ 0 - 2⁻¹ = 4.5 := by sorry

-- Part 2
def system_solution (x : ℝ) : Prop :=
  -2 * x + 6 ≥ 4 ∧ (4 * x + 1) / 3 > x - 1

theorem part_two : ∀ x : ℝ, system_solution x ↔ -4 < x ∧ x ≤ 1 := by sorry

-- Part 3
theorem part_three : {x : ℕ | system_solution x} = {0, 1} := by sorry

end part_one_part_two_part_three_l3325_332583


namespace sandbag_weight_l3325_332518

theorem sandbag_weight (bag_capacity : ℝ) (fill_percentage : ℝ) (weight_increase : ℝ) : 
  bag_capacity = 250 →
  fill_percentage = 0.8 →
  weight_increase = 0.4 →
  (bag_capacity * fill_percentage * (1 + weight_increase)) = 280 := by
  sorry

end sandbag_weight_l3325_332518


namespace direct_proportion_b_value_l3325_332547

/-- A function f is a direct proportion function if there exists a constant k such that f x = k * x for all x. -/
def IsDirectProportion (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x

/-- The function we're considering -/
def f (b : ℝ) (x : ℝ) : ℝ := x + b - 2

theorem direct_proportion_b_value :
  (IsDirectProportion (f b)) → b = 2 := by
  sorry

end direct_proportion_b_value_l3325_332547


namespace black_ball_probability_l3325_332539

/-- Given a bag of 100 balls with 45 red balls and a probability of 0.23 for drawing a white ball,
    the probability of drawing a black ball is 0.32. -/
theorem black_ball_probability
  (total_balls : ℕ)
  (red_balls : ℕ)
  (white_prob : ℝ)
  (h_total : total_balls = 100)
  (h_red : red_balls = 45)
  (h_white_prob : white_prob = 0.23)
  : (total_balls - red_balls - (white_prob * total_balls)) / total_balls = 0.32 := by
  sorry


end black_ball_probability_l3325_332539


namespace append_nine_to_two_digit_number_l3325_332500

/-- Given a two-digit number with tens digit t and units digit u,
    appending 9 to the right results in 100t + 10u + 9 -/
theorem append_nine_to_two_digit_number (t u : ℕ) 
  (h1 : t ≥ 1 ∧ t ≤ 9) (h2 : u ≥ 0 ∧ u ≤ 9) :
  (10 * t + u) * 10 + 9 = 100 * t + 10 * u + 9 := by
  sorry


end append_nine_to_two_digit_number_l3325_332500


namespace rides_second_day_l3325_332537

def rides_first_day : ℕ := 4
def total_rides : ℕ := 7

theorem rides_second_day : total_rides - rides_first_day = 3 := by
  sorry

end rides_second_day_l3325_332537


namespace perpendicular_vectors_exist_minimum_dot_product_l3325_332596

/-- Given vectors in 2D space -/
def OA : Fin 2 → ℝ := ![5, 1]
def OB : Fin 2 → ℝ := ![1, 7]
def OC : Fin 2 → ℝ := ![4, 2]

/-- Vector OM as a function of t -/
def OM (t : ℝ) : Fin 2 → ℝ := fun i => t * OC i

/-- Vector MA as a function of t -/
def MA (t : ℝ) : Fin 2 → ℝ := fun i => OA i - OM t i

/-- Vector MB as a function of t -/
def MB (t : ℝ) : Fin 2 → ℝ := fun i => OB i - OM t i

/-- Dot product of MA and MB -/
def MA_dot_MB (t : ℝ) : ℝ := (MA t 0) * (MB t 0) + (MA t 1) * (MB t 1)

theorem perpendicular_vectors_exist :
  ∃ t : ℝ, MA_dot_MB t = 0 ∧ t = (5 + Real.sqrt 10) / 5 ∨ t = (5 - Real.sqrt 10) / 5 := by
  sorry

theorem minimum_dot_product :
  ∃ t : ℝ, ∀ s : ℝ, MA_dot_MB t ≤ MA_dot_MB s ∧ OM t = OC := by
  sorry

end perpendicular_vectors_exist_minimum_dot_product_l3325_332596


namespace cubic_inequality_l3325_332557

theorem cubic_inequality (a b : ℝ) (h : a < b) :
  a^3 - 3*a ≤ b^3 - 3*b + 4 ∧
  (a^3 - 3*a = b^3 - 3*b + 4 ↔ a = -1 ∧ b = 1) :=
by sorry

end cubic_inequality_l3325_332557


namespace B_2_2_l3325_332598

def B : ℕ → ℕ → ℕ
| 0, n => n + 2
| m+1, 0 => B m 2
| m+1, n+1 => B m (B (m+1) n)

theorem B_2_2 : B 2 2 = 8 := by sorry

end B_2_2_l3325_332598


namespace smallest_sum_with_conditions_l3325_332521

theorem smallest_sum_with_conditions (a b : ℕ+) 
  (h1 : Nat.gcd (a + b) 330 = 1)
  (h2 : (a : ℕ)^(a : ℕ) % (b : ℕ)^(b : ℕ) = 0)
  (h3 : ¬(∃k : ℕ, b = k * a)) :
  (∀ c d : ℕ+, 
    Nat.gcd (c + d) 330 = 1 → 
    (c : ℕ)^(c : ℕ) % (d : ℕ)^(d : ℕ) = 0 → 
    ¬(∃k : ℕ, d = k * c) → 
    a + b ≤ c + d) ∧ 
  a + b = 147 := by
sorry

end smallest_sum_with_conditions_l3325_332521


namespace smallest_multiple_three_is_solution_three_is_smallest_l3325_332554

theorem smallest_multiple (x : ℕ) : x > 0 ∧ 675 ∣ (450 * x) → x ≥ 3 :=
sorry

theorem three_is_solution : 675 ∣ (450 * 3) :=
sorry

theorem three_is_smallest : ∀ y : ℕ, y > 0 ∧ y < 3 → ¬(675 ∣ (450 * y)) :=
sorry

end smallest_multiple_three_is_solution_three_is_smallest_l3325_332554


namespace equation_root_implies_m_value_l3325_332591

theorem equation_root_implies_m_value (x m : ℝ) :
  x > 0 →
  (x - 1) / (x - 5) = m * x / (10 - 2 * x) →
  m = -8/5 :=
by sorry

end equation_root_implies_m_value_l3325_332591


namespace fraction_addition_l3325_332599

theorem fraction_addition : (2 : ℚ) / 5 + (1 : ℚ) / 3 = (11 : ℚ) / 15 := by
  sorry

end fraction_addition_l3325_332599


namespace currency_notes_count_l3325_332589

theorem currency_notes_count 
  (total_amount : ℕ) 
  (denomination_1 : ℕ) 
  (denomination_2 : ℕ) 
  (count_denomination_2 : ℕ) : 
  total_amount = 5000 ∧ 
  denomination_1 = 95 ∧ 
  denomination_2 = 45 ∧ 
  count_denomination_2 = 71 → 
  ∃ (count_denomination_1 : ℕ), 
    count_denomination_1 * denomination_1 + count_denomination_2 * denomination_2 = total_amount ∧ 
    count_denomination_1 + count_denomination_2 = 90 :=
by sorry

end currency_notes_count_l3325_332589


namespace cube_volume_from_surface_area_l3325_332565

/-- Theorem: A cube with surface area approximately 600 square cc has a volume of 1000 cubic cc. -/
theorem cube_volume_from_surface_area :
  ∃ (s : ℝ), s > 0 ∧ 6 * s^2 = 599.9999999999998 → s^3 = 1000 :=
by
  sorry

end cube_volume_from_surface_area_l3325_332565


namespace range_of_f_l3325_332563

-- Define the function f
def f (x : ℝ) := x^2 - 6*x - 9

-- State the theorem
theorem range_of_f :
  ∃ (a b : ℝ), a < b ∧
  (Set.Icc a b) = {y | ∃ x ∈ Set.Ioo 1 4, f x = y} ∧
  a = -18 ∧ b = -14 := by
  sorry

end range_of_f_l3325_332563


namespace class_duty_assignment_l3325_332522

theorem class_duty_assignment (num_boys num_girls : ℕ) 
  (h1 : num_boys = 16) 
  (h2 : num_girls = 14) : 
  num_boys * num_girls = 224 := by
  sorry

end class_duty_assignment_l3325_332522


namespace isosceles_triangle_not_unique_l3325_332532

-- Define an isosceles triangle
structure IsoscelesTriangle where
  side : ℝ
  base : ℝ
  radius : ℝ
  side_positive : 0 < side
  base_positive : 0 < base
  radius_positive : 0 < radius

-- Theorem statement
theorem isosceles_triangle_not_unique (r : ℝ) (hr : 0 < r) :
  ∃ (t1 t2 : IsoscelesTriangle), t1.radius = r ∧ t2.radius = r ∧ t1 ≠ t2 := by
  sorry

end isosceles_triangle_not_unique_l3325_332532


namespace dice_sum_symmetry_l3325_332526

/-- The number of dice being rolled -/
def num_dice : ℕ := 9

/-- The minimum value on each die -/
def min_value : ℕ := 1

/-- The maximum value on each die -/
def max_value : ℕ := 6

/-- The sum we're comparing to -/
def comparison_sum : ℕ := 15

/-- The function to calculate the symmetric sum -/
def symmetric_sum (s : ℕ) : ℕ :=
  2 * ((num_dice * min_value + num_dice * max_value) / 2) - s

theorem dice_sum_symmetry :
  symmetric_sum comparison_sum = 48 :=
sorry

end dice_sum_symmetry_l3325_332526


namespace expand_expression_l3325_332594

theorem expand_expression (x y : ℝ) : (2*x + 3) * (5*y + 7) = 10*x*y + 14*x + 15*y + 21 := by
  sorry

end expand_expression_l3325_332594


namespace regions_count_l3325_332559

-- Define the points
def A : ℝ × ℝ := (2, 2)
def B : ℝ × ℝ := (-2, 2)
def C : ℝ × ℝ := (-2, -2)
def D : ℝ × ℝ := (2, -2)
def E : ℝ × ℝ := (1, 0)
def F : ℝ × ℝ := (0, 1)
def G : ℝ × ℝ := (-1, 0)
def H : ℝ × ℝ := (0, -1)

-- Define the set of all points
def points : Set (ℝ × ℝ) := {A, B, C, D, E, F, G, H}

-- Define the square ABCD
def squareABCD : Set (ℝ × ℝ) := {(x, y) | -2 ≤ x ∧ x ≤ 2 ∧ -2 ≤ y ∧ y ≤ 2}

-- Define a function to count regions formed by line segments
def countRegions (pts : Set (ℝ × ℝ)) (square : Set (ℝ × ℝ)) : ℕ := sorry

-- Theorem statement
theorem regions_count : countRegions points squareABCD = 60 := by sorry

end regions_count_l3325_332559


namespace difference_C₁_C₂_l3325_332588

/-- Triangle ABC with given angle measures and altitude from C --/
structure TriangleABC where
  A : ℝ
  B : ℝ
  C : ℝ
  C₁ : ℝ
  C₂ : ℝ
  angleA_eq : A = 30
  angleB_eq : B = 70
  sum_angles : A + B + C = 180
  C_split : C = C₁ + C₂
  right_angle_AC₁ : A + C₁ + 90 = 180
  right_angle_BC₂ : B + C₂ + 90 = 180

/-- Theorem: In the given triangle, C₁ - C₂ = 40° --/
theorem difference_C₁_C₂ (t : TriangleABC) : t.C₁ - t.C₂ = 40 := by
  sorry

end difference_C₁_C₂_l3325_332588


namespace ordered_triples_solution_l3325_332592

theorem ordered_triples_solution :
  ∀ a b c : ℝ,
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (⌊a⌋ * b * c = 3 ∧ a * ⌊b⌋ * c = 4 ∧ a * b * ⌊c⌋ = 5) →
  ((a = Real.sqrt 30 / 3 ∧ b = Real.sqrt 30 / 4 ∧ c = 2 * Real.sqrt 30 / 5) ∨
   (a = Real.sqrt 30 / 3 ∧ b = Real.sqrt 30 / 2 ∧ c = Real.sqrt 30 / 5)) :=
by sorry

end ordered_triples_solution_l3325_332592


namespace kabadi_players_l3325_332501

theorem kabadi_players (kho_kho_only : ℕ) (both : ℕ) (total : ℕ) :
  kho_kho_only = 35 →
  both = 5 →
  total = 45 →
  ∃ kabadi : ℕ, kabadi = 15 ∧ total = kabadi + kho_kho_only - both :=
by sorry

end kabadi_players_l3325_332501


namespace condition_analysis_l3325_332579

theorem condition_analysis (a : ℝ) : 
  (∀ a, a > 0 → a^2 + a ≥ 0) ∧ 
  (∃ a, a^2 + a ≥ 0 ∧ a ≤ 0) :=
by sorry

end condition_analysis_l3325_332579


namespace stool_height_is_75cm_l3325_332528

/-- Represents the problem setup for Alice's light bulb replacement task -/
structure LightBulbProblem where
  ceiling_height : ℝ
  bulb_below_ceiling : ℝ
  alice_height : ℝ
  alice_reach : ℝ
  decorative_item_below_ceiling : ℝ

/-- Calculates the required stool height for Alice to reach the light bulb -/
def calculate_stool_height (p : LightBulbProblem) : ℝ :=
  p.ceiling_height - p.bulb_below_ceiling - (p.alice_height + p.alice_reach)

/-- Theorem stating that the stool height Alice needs is 75 cm -/
theorem stool_height_is_75cm (p : LightBulbProblem) 
    (h1 : p.ceiling_height = 300)
    (h2 : p.bulb_below_ceiling = 15)
    (h3 : p.alice_height = 160)
    (h4 : p.alice_reach = 50)
    (h5 : p.decorative_item_below_ceiling = 20) :
    calculate_stool_height p = 75 := by
  sorry

#eval calculate_stool_height {
  ceiling_height := 300,
  bulb_below_ceiling := 15,
  alice_height := 160,
  alice_reach := 50,
  decorative_item_below_ceiling := 20
}

end stool_height_is_75cm_l3325_332528


namespace min_value_inequality_l3325_332541

theorem min_value_inequality (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h1 : a + 2*b = 1) (h2 : c + 2*d = 1) : 
  1/a + 1/(b*c*d) > 25 := by
sorry

end min_value_inequality_l3325_332541


namespace total_payment_l3325_332507

/-- The cost of potatoes in yuan per kilogram -/
def potato_cost : ℝ := 1

/-- The cost of celery in yuan per kilogram -/
def celery_cost : ℝ := 0.7

/-- The total cost of buying potatoes and celery -/
def total_cost (a b : ℝ) : ℝ := a * potato_cost + b * celery_cost

theorem total_payment (a b : ℝ) : total_cost a b = a + 0.7 * b := by
  sorry

end total_payment_l3325_332507


namespace divisibility_implies_equality_l3325_332543

theorem divisibility_implies_equality (a b : ℕ) (ha : a > 0) (hb : b > 0) 
  (h : a * b ∣ (a^2 + b^2)) : a = b := by
  sorry

end divisibility_implies_equality_l3325_332543


namespace fraction_evaluation_l3325_332538

theorem fraction_evaluation : (1 - 1/4) / (1 - 1/3) = 9/8 := by sorry

end fraction_evaluation_l3325_332538


namespace may_to_june_increase_l3325_332533

-- Define the percentage changes
def march_to_april_increase : ℝ := 0.10
def april_to_may_decrease : ℝ := 0.20
def overall_increase : ℝ := 0.3200000000000003

-- Define the function to calculate the final value after percentage changes
def final_value (initial : ℝ) (increase1 : ℝ) (decrease : ℝ) (increase2 : ℝ) : ℝ :=
  initial * (1 + increase1) * (1 - decrease) * (1 + increase2)

-- Theorem to prove
theorem may_to_june_increase (initial : ℝ) (initial_pos : initial > 0) :
  ∃ (may_to_june : ℝ), 
    final_value initial march_to_april_increase april_to_may_decrease may_to_june = 
    initial * (1 + overall_increase) ∧ 
    may_to_june = 0.50 := by
  sorry

end may_to_june_increase_l3325_332533
