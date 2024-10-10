import Mathlib

namespace monotone_function_k_range_l3255_325549

/-- Given a function f(x) = e^x + kx - ln x that is monotonically increasing on (1, +∞),
    prove that k ∈ [1-e, +∞) -/
theorem monotone_function_k_range (k : ℝ) :
  (∀ x > 1, Monotone (fun x => Real.exp x + k * x - Real.log x)) →
  k ∈ Set.Ici (1 - Real.exp 1) :=
sorry

end monotone_function_k_range_l3255_325549


namespace annes_cats_weight_l3255_325521

/-- Given Anne's cats' weights, prove the total weight she carries -/
theorem annes_cats_weight (female_weight : ℝ) (male_weight_ratio : ℝ) : 
  female_weight = 2 → 
  male_weight_ratio = 2 → 
  female_weight + female_weight * male_weight_ratio = 6 := by
  sorry

end annes_cats_weight_l3255_325521


namespace stamp_sale_difference_l3255_325577

def red_stamps : ℕ := 30
def white_stamps : ℕ := 80
def red_stamp_price : ℚ := 50 / 100
def white_stamp_price : ℚ := 20 / 100

theorem stamp_sale_difference :
  white_stamps * white_stamp_price - red_stamps * red_stamp_price = 1 := by sorry

end stamp_sale_difference_l3255_325577


namespace infinite_solutions_condition_l3255_325510

theorem infinite_solutions_condition (b : ℝ) :
  (∀ x : ℝ, 5 * (3 * x - b) = 3 * (5 * x + 15)) ↔ b = -9 := by sorry

end infinite_solutions_condition_l3255_325510


namespace expression_equality_l3255_325538

theorem expression_equality : (2 + Real.sqrt 3) ^ 0 + 3 * Real.tan (π / 6) - |Real.sqrt 3 - 2| + (1 / 2)⁻¹ = 1 + 2 * Real.sqrt 3 := by
  sorry

end expression_equality_l3255_325538


namespace right_triangle_stable_l3255_325596

/-- A shape is considered stable if it maintains its form without deformation under normal conditions. -/
def Stable (shape : Type) : Prop := sorry

/-- A parallelogram is a quadrilateral with opposite sides parallel. -/
def Parallelogram : Type := sorry

/-- A square is a quadrilateral with four equal sides and four right angles. -/
def Square : Type := sorry

/-- A rectangle is a quadrilateral with four right angles. -/
def Rectangle : Type := sorry

/-- A right triangle is a triangle with one right angle. -/
def RightTriangle : Type := sorry

/-- Theorem stating that among the given shapes, only the right triangle is inherently stable. -/
theorem right_triangle_stable :
  ¬Stable Parallelogram ∧
  ¬Stable Square ∧
  ¬Stable Rectangle ∧
  Stable RightTriangle :=
sorry

end right_triangle_stable_l3255_325596


namespace quadratic_inequality_solution_set_l3255_325592

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 - (1/6)*x - 1/6 < 0} = Set.Ioo (-1/3 : ℝ) (1/2 : ℝ) := by
  sorry

end quadratic_inequality_solution_set_l3255_325592


namespace opposite_of_negative_five_l3255_325536

-- Define the concept of opposite
def opposite (a : ℤ) : ℤ := -a

-- Theorem statement
theorem opposite_of_negative_five : opposite (-5) = 5 := by
  sorry

end opposite_of_negative_five_l3255_325536


namespace seats_needed_for_zoo_trip_l3255_325527

theorem seats_needed_for_zoo_trip (total_children : ℕ) (children_per_seat : ℕ) (seats_needed : ℕ) : 
  total_children = 58 → children_per_seat = 2 → seats_needed = total_children / children_per_seat → seats_needed = 29 := by
  sorry

end seats_needed_for_zoo_trip_l3255_325527


namespace dans_age_l3255_325587

theorem dans_age (dans_present_age : ℕ) : dans_present_age = 6 :=
  by
  have h : dans_present_age + 18 = 8 * (dans_present_age - 3) :=
    by sorry
  
  sorry

end dans_age_l3255_325587


namespace no_real_solution_for_log_equation_l3255_325530

theorem no_real_solution_for_log_equation :
  ∀ x : ℝ, ¬(Real.log (x + 5) + Real.log (x - 3) = Real.log (x^2 - 8*x + 15)) :=
by sorry

end no_real_solution_for_log_equation_l3255_325530


namespace third_restaurant_meals_l3255_325589

/-- The number of meals served by Gordon's third restaurant per day -/
def third_restaurant_meals_per_day (
  total_restaurants : ℕ)
  (first_restaurant_meals_per_day : ℕ)
  (second_restaurant_meals_per_day : ℕ)
  (total_meals_per_week : ℕ) : ℕ :=
  (total_meals_per_week - 7 * (first_restaurant_meals_per_day + second_restaurant_meals_per_day)) / 7

theorem third_restaurant_meals (
  total_restaurants : ℕ)
  (first_restaurant_meals_per_day : ℕ)
  (second_restaurant_meals_per_day : ℕ)
  (total_meals_per_week : ℕ)
  (h1 : total_restaurants = 3)
  (h2 : first_restaurant_meals_per_day = 20)
  (h3 : second_restaurant_meals_per_day = 40)
  (h4 : total_meals_per_week = 770) :
  third_restaurant_meals_per_day total_restaurants first_restaurant_meals_per_day second_restaurant_meals_per_day total_meals_per_week = 50 :=
by
  sorry

end third_restaurant_meals_l3255_325589


namespace seating_theorem_l3255_325509

def seating_arrangements (total_people : ℕ) (rows : ℕ) (people_per_row : ℕ) 
  (specific_front : ℕ) (specific_back : ℕ) : ℕ :=
  let front_arrangements := Nat.descFactorial people_per_row specific_front
  let back_arrangements := Nat.descFactorial people_per_row specific_back
  let remaining_people := total_people - specific_front - specific_back
  let remaining_arrangements := Nat.factorial remaining_people
  front_arrangements * back_arrangements * remaining_arrangements

theorem seating_theorem : 
  seating_arrangements 8 2 4 2 1 = 5760 := by
  sorry

end seating_theorem_l3255_325509


namespace book_selection_theorem_l3255_325534

theorem book_selection_theorem :
  let mystery_count : ℕ := 4
  let fantasy_count : ℕ := 3
  let biography_count : ℕ := 3
  let different_genre_pairs : ℕ := 
    mystery_count * fantasy_count + 
    mystery_count * biography_count + 
    fantasy_count * biography_count
  different_genre_pairs = 33 := by
  sorry

end book_selection_theorem_l3255_325534


namespace younger_son_future_age_l3255_325541

def age_difference : ℕ := 10
def elder_son_current_age : ℕ := 40
def years_in_future : ℕ := 30

theorem younger_son_future_age :
  let younger_son_current_age := elder_son_current_age - age_difference
  younger_son_current_age + years_in_future = 60 := by sorry

end younger_son_future_age_l3255_325541


namespace contrapositive_proof_l3255_325585

theorem contrapositive_proof : 
  (∀ a b : ℝ, a^2 + b^2 = 0 → a = 0 ∧ b = 0) ↔ 
  (∀ a b : ℝ, a ≠ 0 ∨ b ≠ 0 → a^2 + b^2 ≠ 0) :=
by sorry

end contrapositive_proof_l3255_325585


namespace expression_value_l3255_325560

theorem expression_value : 
  (12 - 11 + 10 - 9 + 8 - 7 + 6 - 5 + 4 - 3 + 2 - 1) / 
  (2 - 3 + 4 - 5 + 6 - 7 + 8 - 9 + 10 - 11 + 12) = 6 / 7 := by
  sorry

end expression_value_l3255_325560


namespace M_characterization_l3255_325542

def M (m : ℝ) : Set ℝ := {x | x^2 - m*x + 6 = 0}

def valid_set (S : Set ℝ) : Prop :=
  S = {2, 3} ∨ S = {1, 6} ∨ S = ∅

def valid_m (m : ℝ) : Prop :=
  m = 7 ∨ m = 5 ∨ (m > -2*Real.sqrt 6 ∧ m < 2*Real.sqrt 6)

theorem M_characterization (m : ℝ) :
  (M m ∩ {1, 2, 3, 6} = M m) →
  (valid_set (M m) ∧ valid_m m) :=
sorry

end M_characterization_l3255_325542


namespace polynomial_factorization_l3255_325578

theorem polynomial_factorization (x : ℝ) :
  x^2 - 6*x + 9 - 64*x^4 = (-8*x^2 + x - 3) * (8*x^2 + x - 3) :=
by sorry

end polynomial_factorization_l3255_325578


namespace franks_candy_bags_l3255_325516

theorem franks_candy_bags (pieces_per_bag : ℕ) (total_pieces : ℕ) (h1 : pieces_per_bag = 11) (h2 : total_pieces = 22) :
  total_pieces / pieces_per_bag = 2 :=
by sorry

end franks_candy_bags_l3255_325516


namespace local_minimum_of_f_l3255_325584

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 12*x

-- State the theorem
theorem local_minimum_of_f :
  ∃ (x₀ : ℝ), (∀ (x : ℝ), f x ≥ f x₀) ∧ x₀ = 2 :=
sorry

end local_minimum_of_f_l3255_325584


namespace nearest_integer_to_a_fifth_l3255_325514

theorem nearest_integer_to_a_fifth (a b c : ℝ) 
  (h_order : a ≥ b ∧ b ≥ c)
  (h_eq1 : a^2 * b * c + a * b^2 * c + a * b * c^2 + 8 = a + b + c)
  (h_eq2 : a^2 * b + a^2 * c + b^2 * c + b^2 * a + c^2 * a + c^2 * b + 3 * a * b * c = -4)
  (h_eq3 : a^2 * b^2 * c + a * b^2 * c^2 + a^2 * b * c^2 = 2 + a * b + b * c + c * a)
  (h_sum_pos : a + b + c > 0) :
  ∃ (n : ℤ), |n - a^5| < 1/2 ∧ n = 1279 := by
sorry

end nearest_integer_to_a_fifth_l3255_325514


namespace trapezoid_area_l3255_325512

/-- Given two equilateral triangles and four congruent trapezoids between them,
    this theorem proves that the area of one trapezoid is 8 square units. -/
theorem trapezoid_area
  (outer_triangle_area : ℝ)
  (inner_triangle_area : ℝ)
  (num_trapezoids : ℕ)
  (h_outer_area : outer_triangle_area = 36)
  (h_inner_area : inner_triangle_area = 4)
  (h_num_trapezoids : num_trapezoids = 4) :
  (outer_triangle_area - inner_triangle_area) / num_trapezoids = 8 := by
  sorry

end trapezoid_area_l3255_325512


namespace jellybean_count_is_84_l3255_325552

/-- Calculates the final number of jellybeans in a jar after a series of actions. -/
def final_jellybean_count (initial : ℕ) (samantha_took : ℕ) (shelby_ate : ℕ) : ℕ :=
  let remaining_after_samantha := initial - samantha_took
  let remaining_after_shelby := remaining_after_samantha - shelby_ate
  let scarlett_returned := shelby_ate
  let shannon_added := (samantha_took + shelby_ate) / 2
  remaining_after_shelby + scarlett_returned + shannon_added

/-- Theorem stating that given the initial conditions, the final number of jellybeans is 84. -/
theorem jellybean_count_is_84 :
  final_jellybean_count 90 24 12 = 84 := by
  sorry

#eval final_jellybean_count 90 24 12

end jellybean_count_is_84_l3255_325552


namespace chess_game_players_l3255_325553

def number_of_players : ℕ := 15
def total_games : ℕ := 105

theorem chess_game_players :
  ∃ k : ℕ,
    k > 0 ∧
    k < number_of_players ∧
    (number_of_players.choose k) = total_games ∧
    k = 2 := by
  sorry

end chess_game_players_l3255_325553


namespace simplify_fraction_l3255_325572

theorem simplify_fraction : (5^5 + 5^3) / (5^4 - 5^2 + 1) = 16250 / 601 := by
  sorry

end simplify_fraction_l3255_325572


namespace books_from_first_shop_l3255_325519

theorem books_from_first_shop 
  (total_first : ℕ) 
  (books_second : ℕ) 
  (total_second : ℕ) 
  (avg_price : ℕ) :
  total_first = 1500 →
  books_second = 60 →
  total_second = 340 →
  avg_price = 16 →
  ∃ (books_first : ℕ), 
    (total_first + total_second) / (books_first + books_second) = avg_price ∧
    books_first = 55 :=
by sorry

end books_from_first_shop_l3255_325519


namespace car_speed_second_hour_l3255_325556

/-- Given a car traveling for two hours with a speed of 98 km/h in the first hour
    and an average speed of 84 km/h over the two hours,
    prove that the speed of the car in the second hour is 70 km/h. -/
theorem car_speed_second_hour :
  let speed_first_hour : ℝ := 98
  let average_speed : ℝ := 84
  let total_time : ℝ := 2
  let speed_second_hour : ℝ := (average_speed * total_time) - speed_first_hour
  speed_second_hour = 70 := by
sorry

end car_speed_second_hour_l3255_325556


namespace rational_roots_of_quadratic_l3255_325558

theorem rational_roots_of_quadratic 
  (p q n : ℚ) : 
  ∃ (x : ℚ), (p + q + n) * x^2 - 2*(p + q) * x + (p + q - n) = 0 :=
by sorry

end rational_roots_of_quadratic_l3255_325558


namespace golden_retriever_age_l3255_325511

-- Define the weight gain per year
def weight_gain_per_year : ℕ := 11

-- Define the current weight
def current_weight : ℕ := 88

-- Define the age of the golden retriever
def age : ℕ := current_weight / weight_gain_per_year

-- Theorem to prove
theorem golden_retriever_age :
  age = 8 :=
by sorry

end golden_retriever_age_l3255_325511


namespace sum_of_A_and_C_is_five_l3255_325507

-- Define the multiplication problem
def multiplication_problem (A B C D : Nat) : Prop :=
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
  A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧
  (100 * A + 10 * B + A) * D = 1000 * C + 100 * B + 10 * A + D

-- State the theorem
theorem sum_of_A_and_C_is_five :
  ∀ A B C D : Nat, multiplication_problem A B C D → A + C = 5 :=
by sorry

end sum_of_A_and_C_is_five_l3255_325507


namespace min_value_sum_reciprocals_l3255_325517

theorem min_value_sum_reciprocals (a b c d e f : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (pos_d : 0 < d) (pos_e : 0 < e) (pos_f : 0 < f)
  (sum_eq_10 : a + b + c + d + e + f = 10) :
  (4/a + 9/b + 16/c + 25/d + 36/e + 49/f) ≥ 72.9 := by
  sorry

end min_value_sum_reciprocals_l3255_325517


namespace convex_polygon_in_rectangle_l3255_325515

/-- A convex polygon in the plane -/
structure ConvexPolygon where
  vertices : Set (ℝ × ℝ)
  convex : Convex ℝ (convexHull ℝ vertices)
  finite : Finite vertices

/-- The area of a polygon -/
def area (p : ConvexPolygon) : ℝ := sorry

/-- A rectangle in the plane -/
structure Rectangle where
  lower_left : ℝ × ℝ
  upper_right : ℝ × ℝ
  valid : lower_left.1 < upper_right.1 ∧ lower_left.2 < upper_right.2

/-- The area of a rectangle -/
def rectangleArea (r : Rectangle) : ℝ :=
  (r.upper_right.1 - r.lower_left.1) * (r.upper_right.2 - r.lower_left.2)

/-- A polygon is contained in a rectangle -/
def contained (p : ConvexPolygon) (r : Rectangle) : Prop := sorry

theorem convex_polygon_in_rectangle :
  ∀ (p : ConvexPolygon), area p = 1 →
  ∃ (r : Rectangle), contained p r ∧ rectangleArea r ≤ 2 := by sorry

end convex_polygon_in_rectangle_l3255_325515


namespace option_B_more_cost_effective_l3255_325548

/-- The cost function for Option A -/
def cost_A (x : ℝ) : ℝ := 60 + 18 * x

/-- The cost function for Option B -/
def cost_B (x : ℝ) : ℝ := 150 + 15 * x

/-- Theorem: Option B is more cost-effective for 40 kilograms of blueberries -/
theorem option_B_more_cost_effective :
  cost_B 40 < cost_A 40 := by
  sorry

end option_B_more_cost_effective_l3255_325548


namespace max_tiles_on_floor_l3255_325545

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Calculates the maximum number of tiles that can fit in a given direction -/
def maxTilesInDirection (floor : Rectangle) (tile : Rectangle) : ℕ :=
  (floor.width / tile.width) * (floor.height / tile.height)

/-- Theorem: The maximum number of 20x30 tiles on a 100x150 floor is 25 -/
theorem max_tiles_on_floor :
  let floor := Rectangle.mk 100 150
  let tile := Rectangle.mk 20 30
  let maxTiles := max (maxTilesInDirection floor tile) (maxTilesInDirection floor (Rectangle.mk tile.height tile.width))
  maxTiles = 25 := by
  sorry

#check max_tiles_on_floor

end max_tiles_on_floor_l3255_325545


namespace none_of_statements_true_l3255_325567

theorem none_of_statements_true (s x y : ℝ) 
  (h_s : s > 1) 
  (h_xy : x^2 * y ≠ 0) 
  (h_ineq : x * s^2 > y * s^2) : 
  ¬(-x > -y) ∧ ¬(-x > y) ∧ ¬(1 > -y/x) ∧ ¬(1 < y/x) := by
sorry

end none_of_statements_true_l3255_325567


namespace taco_truck_beef_per_taco_l3255_325597

theorem taco_truck_beef_per_taco 
  (total_beef : ℝ) 
  (selling_price : ℝ) 
  (cost_per_taco : ℝ) 
  (total_profit : ℝ) 
  (h1 : total_beef = 100)
  (h2 : selling_price = 2)
  (h3 : cost_per_taco = 1.5)
  (h4 : total_profit = 200) :
  ∃ (beef_per_taco : ℝ), 
    beef_per_taco = 1/4 ∧ 
    (total_beef / beef_per_taco) * (selling_price - cost_per_taco) = total_profit := by
  sorry

end taco_truck_beef_per_taco_l3255_325597


namespace complex_power_sum_l3255_325502

theorem complex_power_sum (i : ℂ) : i^2 = -1 → i^123 + i^223 + i^323 = -3*i := by
  sorry

end complex_power_sum_l3255_325502


namespace distance_between_cars_l3255_325551

theorem distance_between_cars (initial_distance : ℝ) (car1_distance : ℝ) (car2_distance : ℝ) :
  initial_distance = 105 →
  car1_distance = 50 →
  car2_distance = 35 →
  initial_distance - (car1_distance + car2_distance) = 20 := by
sorry

end distance_between_cars_l3255_325551


namespace remainder_8457_mod_9_l3255_325518

/-- The sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- A number is congruent to the sum of its digits modulo 9 -/
axiom sum_of_digits_mod_9 (n : ℕ) : n ≡ sum_of_digits n [MOD 9]

/-- The remainder when 8457 is divided by 9 is 6 -/
theorem remainder_8457_mod_9 : 8457 % 9 = 6 := by sorry

end remainder_8457_mod_9_l3255_325518


namespace greatest_divisor_with_remainders_l3255_325559

theorem greatest_divisor_with_remainders :
  ∃ (d : ℕ), d > 0 ∧
    (150 % d = 50) ∧
    (230 % d = 5) ∧
    (175 % d = 25) ∧
    (∀ (k : ℕ), k > 0 →
      (150 % k = 50) →
      (230 % k = 5) →
      (175 % k = 25) →
      k ≤ d) ∧
    d = 25 := by
  sorry

end greatest_divisor_with_remainders_l3255_325559


namespace fraction_inequality_l3255_325599

theorem fraction_inequality (a b : ℝ) : ¬(∀ a b, a / b = (a + 1) / (b + 1)) :=
sorry

end fraction_inequality_l3255_325599


namespace blue_marbles_count_l3255_325531

/-- The number of blue marbles Jason has -/
def jason_blue_marbles : ℕ := 44

/-- The number of blue marbles Tom has -/
def tom_blue_marbles : ℕ := 24

/-- The total number of blue marbles Jason and Tom have together -/
def total_blue_marbles : ℕ := jason_blue_marbles + tom_blue_marbles

theorem blue_marbles_count : total_blue_marbles = 68 := by
  sorry

end blue_marbles_count_l3255_325531


namespace power_comparison_l3255_325555

theorem power_comparison : 3^17 < 8^9 ∧ 8^9 < 4^15 := by
  sorry

end power_comparison_l3255_325555


namespace sams_mystery_books_l3255_325523

theorem sams_mystery_books (total_books : ℝ) (used_adventure_books : ℝ) (new_crime_books : ℝ)
  (h1 : total_books = 45)
  (h2 : used_adventure_books = 13)
  (h3 : new_crime_books = 15) :
  total_books - (used_adventure_books + new_crime_books) = 17 :=
by sorry

end sams_mystery_books_l3255_325523


namespace subset_relation_l3255_325574

theorem subset_relation (M N : Set ℕ) : 
  M = {1, 2, 3, 4} → N = {2, 3, 4} → N ⊆ M := by
  sorry

end subset_relation_l3255_325574


namespace hank_reads_seven_days_a_week_l3255_325535

/-- Represents Hank's reading habits and total reading time in a week -/
structure ReadingHabits where
  weekdayReadingTime : ℕ  -- Daily reading time on weekdays in minutes
  weekendReadingTime : ℕ  -- Daily reading time on weekends in minutes
  totalWeeklyTime : ℕ     -- Total reading time in a week in minutes

/-- Calculates the number of days Hank reads in a week based on his reading habits -/
def daysReadingPerWeek (habits : ReadingHabits) : ℕ :=
  if (5 * habits.weekdayReadingTime + 2 * habits.weekendReadingTime) = habits.totalWeeklyTime
  then 7
  else 0

/-- Theorem stating that Hank reads 7 days a week given his reading habits -/
theorem hank_reads_seven_days_a_week :
  let habits : ReadingHabits := {
    weekdayReadingTime := 90,
    weekendReadingTime := 180,
    totalWeeklyTime := 810
  }
  daysReadingPerWeek habits = 7 := by sorry

end hank_reads_seven_days_a_week_l3255_325535


namespace square_area_from_rectangle_l3255_325513

theorem square_area_from_rectangle (s r l b : ℝ) : 
  r = s →                  -- radius of circle equals side of square
  l = (2 / 5) * r →        -- length of rectangle is two-fifths of radius
  b = 10 →                 -- breadth of rectangle is 10 units
  l * b = 120 →            -- area of rectangle is 120 sq. units
  s^2 = 900 :=             -- area of square is 900 sq. units
by sorry

end square_area_from_rectangle_l3255_325513


namespace arith_geom_seq_iff_not_squarefree_l3255_325554

/-- A sequence in ℤ/mℤ is both arithmetic and geometric progression -/
def is_arith_geom_seq (m : ℕ) (seq : ℕ → ℕ) : Prop :=
  ∃ (a d r : ℕ), ∀ n : ℕ,
    (seq n) % m = (a + n * d) % m ∧
    (seq n) % m = (a * r^n) % m

/-- A sequence is nonconstant -/
def is_nonconstant (m : ℕ) (seq : ℕ → ℕ) : Prop :=
  ∃ i j : ℕ, (seq i) % m ≠ (seq j) % m

/-- m is not squarefree -/
def not_squarefree (m : ℕ) : Prop :=
  ∃ p : ℕ, Prime p ∧ p^2 ∣ m

/-- Main theorem -/
theorem arith_geom_seq_iff_not_squarefree (m : ℕ) :
  (∃ seq : ℕ → ℕ, is_arith_geom_seq m seq ∧ is_nonconstant m seq) ↔ not_squarefree m :=
sorry

end arith_geom_seq_iff_not_squarefree_l3255_325554


namespace perpendicular_lines_slope_l3255_325525

theorem perpendicular_lines_slope (a : ℝ) : 
  (∃ (x y : ℝ), y = a * x - 2) ∧ 
  (∃ (x y : ℝ), y = (a + 2) * x + 1) ∧ 
  (a * (a + 2) = -1) → 
  a = -1 := by
sorry

end perpendicular_lines_slope_l3255_325525


namespace ml_to_litre_fraction_l3255_325586

theorem ml_to_litre_fraction (ml_per_litre : ℝ) (volume_ml : ℝ) :
  ml_per_litre = 1000 →
  volume_ml = 30 →
  volume_ml / ml_per_litre = 0.03 := by
sorry

end ml_to_litre_fraction_l3255_325586


namespace purely_imaginary_implies_a_eq_neg_two_l3255_325582

/-- A complex number z is purely imaginary if its real part is zero and its imaginary part is non-zero -/
def isPurelyImaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

/-- The complex number z defined in terms of real number a -/
def z (a : ℝ) : ℂ := Complex.mk (a^2 + a - 2) (a^2 - 3*a + 2)

/-- Theorem: If z(a) is purely imaginary, then a = -2 -/
theorem purely_imaginary_implies_a_eq_neg_two :
  ∀ a : ℝ, isPurelyImaginary (z a) → a = -2 := by sorry

end purely_imaginary_implies_a_eq_neg_two_l3255_325582


namespace xiaomin_house_coordinates_l3255_325591

/-- Represents a 2D point with x and y coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- The school's position -/
def school : Point := { x := 0, y := 0 }

/-- Xiaomin's house position relative to the school -/
def house_relative : Point := { x := 200, y := -150 }

/-- Theorem stating that Xiaomin's house coordinates are (200, -150) -/
theorem xiaomin_house_coordinates :
  ∃ (p : Point), p.x = school.x + house_relative.x ∧ p.y = school.y + house_relative.y ∧ 
  p.x = 200 ∧ p.y = -150 := by
  sorry

end xiaomin_house_coordinates_l3255_325591


namespace volume_problem_l3255_325544

/-- Given a volume that is the product of three numbers, where two of the numbers are 18 and 6,
    and 48 cubes of edge 3 can be inserted into this volume, prove that the first number in the product is 12. -/
theorem volume_problem (volume : ℝ) (first_number : ℝ) : 
  volume = first_number * 18 * 6 →
  volume = 48 * (3 : ℝ)^3 →
  first_number = 12 := by
  sorry

end volume_problem_l3255_325544


namespace houses_with_neither_feature_l3255_325526

theorem houses_with_neither_feature (total : ℕ) (garage : ℕ) (pool : ℕ) (both : ℕ) 
  (h1 : total = 85) 
  (h2 : garage = 50) 
  (h3 : pool = 40) 
  (h4 : both = 35) : 
  total - (garage + pool - both) = 30 := by
  sorry

end houses_with_neither_feature_l3255_325526


namespace solve_burger_problem_l3255_325508

/-- Represents the problem of calculating the number of double burgers bought. -/
def BurgerProblem (total_cost : ℚ) (total_burgers : ℕ) (single_cost : ℚ) (double_cost : ℚ) : Prop :=
  ∃ (single_burgers double_burgers : ℕ),
    single_burgers + double_burgers = total_burgers ∧
    single_cost * single_burgers + double_cost * double_burgers = total_cost ∧
    double_burgers = 29

/-- Theorem stating the solution to the burger problem. -/
theorem solve_burger_problem :
  BurgerProblem 64.5 50 1 1.5 :=
sorry

end solve_burger_problem_l3255_325508


namespace train_length_calculation_l3255_325537

/-- Calculates the length of a train given its speed, time to cross a bridge, and the bridge length -/
theorem train_length_calculation (train_speed : ℝ) (cross_time : ℝ) (bridge_length : ℝ) :
  train_speed = 65 * (1000 / 3600) →
  cross_time = 13.568145317605362 →
  bridge_length = 145 →
  ∃ (train_length : ℝ), abs (train_length - 100) < 0.1 := by
  sorry

#check train_length_calculation

end train_length_calculation_l3255_325537


namespace necessary_but_not_sufficient_condition_l3255_325540

theorem necessary_but_not_sufficient_condition (a : ℝ) :
  (a > 2 → a ∈ Set.Ici 2) ∧ (∃ x, x ∈ Set.Ici 2 ∧ ¬(x > 2)) := by
  sorry

end necessary_but_not_sufficient_condition_l3255_325540


namespace line_of_symmetry_between_circles_l3255_325524

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 9
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 4*y - 1 = 0

-- Define the line of symmetry
def line_l (x y : ℝ) : Prop := x - y - 2 = 0

-- Define symmetry between points with respect to a line
def symmetric_points (x1 y1 x2 y2 a b c : ℝ) : Prop :=
  a * (x1 + x2) + b * (y1 + y2) + 2 * c = 0

-- Theorem statement
theorem line_of_symmetry_between_circles :
  ∀ (x1 y1 x2 y2 : ℝ),
    circle1 x1 y1 → circle2 x2 y2 →
    symmetric_points x1 y1 x2 y2 1 (-1) (-2) →
    line_l ((x1 + x2) / 2) ((y1 + y2) / 2) :=
sorry

end line_of_symmetry_between_circles_l3255_325524


namespace five_distinct_dice_probability_l3255_325594

def standard_dice_sides : ℕ := 6

def distinct_rolls (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | k + 1 => (standard_dice_sides - k) * distinct_rolls k

theorem five_distinct_dice_probability : 
  (distinct_rolls 5 : ℚ) / (standard_dice_sides ^ 5) = 5 / 54 := by
  sorry

end five_distinct_dice_probability_l3255_325594


namespace lines_perpendicular_l3255_325570

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the two lines
def line1 (t : Triangle) (x y : Real) : Prop :=
  x * Real.sin t.A + t.a * y + t.c = 0

def line2 (t : Triangle) (x y : Real) : Prop :=
  t.b * x - y * Real.sin t.B + Real.sin t.C = 0

-- Theorem statement
theorem lines_perpendicular (t : Triangle) : 
  (∀ x y, line1 t x y → line2 t x y → False) ∨ 
  (∃ x y, line1 t x y ∧ line2 t x y) :=
sorry

end lines_perpendicular_l3255_325570


namespace f_properties_l3255_325547

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 1 then 1/4 + Real.log x / Real.log 4
  else 2^(-x) - 1/4

theorem f_properties :
  (∀ x, f x ≥ 1/4) ∧
  (∀ x, f x = 3/4 ↔ x = 0 ∨ x = 2) := by
sorry

end f_properties_l3255_325547


namespace equation_solution_l3255_325503

theorem equation_solution (x : ℝ) :
  x ≠ -2 → x ≠ 2 →
  (1 / (x + 2) + (x + 6) / (x^2 - 4) = 1) ↔ x = 4 := by
  sorry

end equation_solution_l3255_325503


namespace number_equation_solution_l3255_325563

theorem number_equation_solution : 
  ∃ x : ℝ, (45 + 3 * x = 72) ∧ x = 9 := by sorry

end number_equation_solution_l3255_325563


namespace three_digit_numbers_with_conditions_l3255_325571

theorem three_digit_numbers_with_conditions :
  ∃ (a b : ℕ),
    100 ≤ a ∧ a < b ∧ b < 1000 ∧
    ∃ (k : ℕ), a + b = 498 * k ∧
    ∃ (m : ℕ), b = 5 * m * a ∧
    a = 166 ∧ b = 830 := by
  sorry

end three_digit_numbers_with_conditions_l3255_325571


namespace sphere_radius_in_truncated_cone_l3255_325580

/-- Represents a truncated cone with given radii of horizontal bases -/
structure TruncatedCone where
  bottomRadius : ℝ
  topRadius : ℝ

/-- Represents a sphere with a given radius -/
structure Sphere where
  radius : ℝ

/-- Predicate to check if a sphere is tangent to a truncated cone -/
def isTangent (cone : TruncatedCone) (sphere : Sphere) : Prop :=
  -- The actual implementation of this predicate is complex and would depend on geometric calculations
  sorry

/-- The main theorem stating the radius of the sphere tangent to the truncated cone -/
theorem sphere_radius_in_truncated_cone (cone : TruncatedCone) 
  (h1 : cone.bottomRadius = 24)
  (h2 : cone.topRadius = 8) :
  ∃ (sphere : Sphere), isTangent cone sphere ∧ sphere.radius = 8 * Real.sqrt 3 := by
  sorry

end sphere_radius_in_truncated_cone_l3255_325580


namespace tornado_distance_l3255_325501

/-- Given a tornado that transported objects as follows:
  * A car was transported 200 feet
  * A lawn chair was blown twice as far as the car
  * A birdhouse flew three times farther than the lawn chair
  This theorem proves that the birdhouse flew 1200 feet. -/
theorem tornado_distance (car_distance : ℕ) (lawn_chair_multiplier : ℕ) (birdhouse_multiplier : ℕ)
  (h1 : car_distance = 200)
  (h2 : lawn_chair_multiplier = 2)
  (h3 : birdhouse_multiplier = 3) :
  birdhouse_multiplier * (lawn_chair_multiplier * car_distance) = 1200 := by
  sorry

#check tornado_distance

end tornado_distance_l3255_325501


namespace total_bread_served_l3255_325593

-- Define the quantities of bread served
def wheat_bread : ℚ := 1.25
def white_bread : ℚ := 3/4
def rye_bread : ℚ := 0.6
def multigrain_bread : ℚ := 7/10

-- Theorem to prove
theorem total_bread_served :
  wheat_bread + white_bread + rye_bread + multigrain_bread = 3 + 3/10 := by
  sorry

end total_bread_served_l3255_325593


namespace f_properties_l3255_325522

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a * x / Real.exp x) + (1/2) * x^2 - x

def monotonic_intervals (a : ℝ) : Prop :=
  (a ≤ 0 → (∀ x y, x < y → x < 1 → f a x > f a y) ∧ 
            (∀ x y, x < y → 1 < x → f a x < f a y)) ∧
  (a = Real.exp 1 → (∀ x y, x < y → f a x < f a y)) ∧
  (0 < a ∧ a < Real.exp 1 → 
    (∀ x y, x < y → y < Real.log a → f a x < f a y) ∧
    (∀ x y, x < y → Real.log a < x ∧ y < 1 → f a x > f a y) ∧
    (∀ x y, x < y → 1 < x → f a x < f a y)) ∧
  (Real.exp 1 < a → 
    (∀ x y, x < y → y < 1 → f a x < f a y) ∧
    (∀ x y, x < y → 1 < x ∧ y < Real.log a → f a x > f a y) ∧
    (∀ x y, x < y → Real.log a < x → f a x < f a y))

def number_of_zeros (a : ℝ) : Prop :=
  (Real.exp 1 / 2 < a → ∃! x, f a x = 0) ∧
  ((a = 1 ∨ a = Real.exp 1 / 2) → ∃ x y, x ≠ y ∧ f a x = 0 ∧ f a y = 0 ∧ 
    (∀ z, f a z = 0 → z = x ∨ z = y)) ∧
  (1 < a ∧ a < Real.exp 1 / 2 → ∃ x y z, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    f a x = 0 ∧ f a y = 0 ∧ f a z = 0 ∧ 
    (∀ w, f a w = 0 → w = x ∨ w = y ∨ w = z))

theorem f_properties (a : ℝ) (h : 1 ≤ a) : 
  monotonic_intervals a ∧ number_of_zeros a := by sorry

end f_properties_l3255_325522


namespace circle_plus_92_composed_thrice_l3255_325550

def circle_plus (N : ℝ) : ℝ := 0.75 * N + 2

theorem circle_plus_92_composed_thrice :
  circle_plus (circle_plus (circle_plus 92)) = 43.4375 := by
  sorry

end circle_plus_92_composed_thrice_l3255_325550


namespace largest_s_value_l3255_325562

theorem largest_s_value : ∃ (s : ℝ), 
  (∀ (t : ℝ), (15 * t^2 - 40 * t + 18) / (4 * t - 3) + 6 * t = 7 * t - 1 → t ≤ s) ∧
  (15 * s^2 - 40 * s + 18) / (4 * s - 3) + 6 * s = 7 * s - 1 ∧
  s = 3 :=
by sorry

end largest_s_value_l3255_325562


namespace equal_distribution_l3255_325595

/-- Represents the weight of a mouse's cheese slice -/
structure CheeseSlice where
  weight : ℝ

/-- Represents the total cheese and its distribution -/
structure Cheese where
  total_weight : ℝ
  white : CheeseSlice
  gray : CheeseSlice
  fat : CheeseSlice
  thin : CheeseSlice

/-- The conditions of the cheese distribution problem -/
def cheese_distribution (c : Cheese) : Prop :=
  c.thin.weight = c.fat.weight - 20 ∧
  c.white.weight = c.gray.weight - 8 ∧
  c.white.weight = c.total_weight / 4 ∧
  c.total_weight = c.white.weight + c.gray.weight + c.fat.weight + c.thin.weight

/-- The theorem stating the equal distribution of surplus cheese -/
theorem equal_distribution (c : Cheese) (h : cheese_distribution c) :
  ∃ (new_c : Cheese),
    cheese_distribution new_c ∧
    new_c.white.weight = new_c.gray.weight ∧
    new_c.fat.weight = new_c.thin.weight ∧
    new_c.fat.weight = c.fat.weight - 6 ∧
    new_c.thin.weight = c.thin.weight + 14 :=
  sorry

end equal_distribution_l3255_325595


namespace tangent_slope_three_points_l3255_325543

theorem tangent_slope_three_points (x : ℝ) :
  (3 * x^2 = 3) → (x = 1 ∨ x = -1) := by sorry

#check tangent_slope_three_points

end tangent_slope_three_points_l3255_325543


namespace interlaced_roots_l3255_325598

/-- A quadratic function -/
def QuadraticFunction := ℝ → ℝ

/-- Predicate to check if two real numbers are distinct roots of a quadratic function -/
def are_distinct_roots (f : QuadraticFunction) (x y : ℝ) : Prop :=
  x ≠ y ∧ f x = 0 ∧ f y = 0

/-- Predicate to check if four real numbers are interlaced -/
def are_interlaced (x₁ x₂ x₃ x₄ : ℝ) : Prop :=
  (x₁ < x₃ ∧ x₃ < x₂ ∧ x₂ < x₄) ∨ (x₃ < x₁ ∧ x₁ < x₄ ∧ x₄ < x₂)

theorem interlaced_roots 
  (f g : QuadraticFunction) (x₁ x₂ x₃ x₄ : ℝ) 
  (h₁ : are_distinct_roots f x₁ x₂)
  (h₂ : are_distinct_roots g x₃ x₄)
  (h₃ : g x₁ * g x₂ < 0) :
  are_interlaced x₁ x₂ x₃ x₄ :=
sorry

end interlaced_roots_l3255_325598


namespace bobby_candy_problem_l3255_325539

theorem bobby_candy_problem (total_candy : ℕ) (chocolate_eaten : ℕ) (gummy_eaten : ℕ)
  (h1 : total_candy = 36)
  (h2 : chocolate_eaten = 12)
  (h3 : gummy_eaten = 9)
  (h4 : chocolate_eaten = 2 * (chocolate_eaten + (total_candy - chocolate_eaten - gummy_eaten)) / 3)
  (h5 : gummy_eaten = 3 * (gummy_eaten + (total_candy - chocolate_eaten - gummy_eaten)) / 4) :
  total_candy - chocolate_eaten - gummy_eaten = 9 := by
  sorry

end bobby_candy_problem_l3255_325539


namespace not_perfect_square_l3255_325505

theorem not_perfect_square (m : ℕ) : ¬ ∃ (n : ℕ), ((4 * 10^(2*m+1) + 5) / 9 : ℚ) = n^2 := by
  sorry

end not_perfect_square_l3255_325505


namespace price_change_l3255_325504

theorem price_change (P : ℝ) (h : P > 0) :
  let price_2012 := P * 1.25
  let price_2013 := price_2012 * 0.88
  (price_2013 - P) / P * 100 = 10 := by
  sorry

end price_change_l3255_325504


namespace geometric_sequence_common_ratio_l3255_325579

/-- Given a geometric sequence with first term a₁ and common ratio q,
    if the sum of the first two terms S₂ = 3a₁, then q = 2 -/
theorem geometric_sequence_common_ratio (a₁ : ℝ) (q : ℝ) :
  a₁ ≠ 0 → a₁ + a₁ * q = 3 * a₁ → q = 2 := by
  sorry

end geometric_sequence_common_ratio_l3255_325579


namespace conference_arrangements_l3255_325576

/-- The number of lecturers at the conference -/
def total_lecturers : ℕ := 8

/-- The number of lecturers with specific ordering requirements -/
def ordered_lecturers : ℕ := 3

/-- Calculate the number of permutations for the remaining lecturers -/
def remaining_permutations : ℕ := (total_lecturers - ordered_lecturers).factorial

/-- Calculate the number of ways to arrange the ordered lecturers -/
def ordered_arrangements : ℕ := (total_lecturers - 2) * (total_lecturers - 1) * total_lecturers

/-- The total number of possible arrangements -/
def total_arrangements : ℕ := ordered_arrangements * remaining_permutations

theorem conference_arrangements :
  total_arrangements = 40320 := by sorry

end conference_arrangements_l3255_325576


namespace benny_cards_l3255_325569

theorem benny_cards (added_cards : ℕ) (remaining_cards : ℕ) : 
  added_cards = 4 →
  remaining_cards = 34 →
  ∃ (initial_cards : ℕ),
    initial_cards + added_cards = 2 * remaining_cards ∧
    initial_cards = 64 := by
  sorry

end benny_cards_l3255_325569


namespace train_combined_speed_l3255_325573

/-- The combined speed of two trains moving in opposite directions -/
theorem train_combined_speed 
  (train1_length : ℝ) 
  (train1_time : ℝ) 
  (train2_speed : ℝ) 
  (h1 : train1_length = 180) 
  (h2 : train1_time = 12) 
  (h3 : train2_speed = 30) : 
  train1_length / train1_time + train2_speed = 45 := by
  sorry

end train_combined_speed_l3255_325573


namespace cylinder_volume_ratio_l3255_325590

/-- Given two cylinders with the following properties:
  * S₁ and S₂ are their base areas
  * υ₁ and υ₂ are their volumes
  * They have equal lateral areas
  * S₁/S₂ = 16/9
Then υ₁/υ₂ = 4/3 -/
theorem cylinder_volume_ratio (S₁ S₂ υ₁ υ₂ : ℝ) (h_positive : S₁ > 0 ∧ S₂ > 0 ∧ υ₁ > 0 ∧ υ₂ > 0)
    (h_base_ratio : S₁ / S₂ = 16 / 9) (h_equal_lateral : ∃ (r₁ r₂ h₁ h₂ : ℝ), 
    S₁ = π * r₁^2 ∧ S₂ = π * r₂^2 ∧ υ₁ = S₁ * h₁ ∧ υ₂ = S₂ * h₂ ∧ 2 * π * r₁ * h₁ = 2 * π * r₂ * h₂) :
  υ₁ / υ₂ = 4 / 3 := by
  sorry

end cylinder_volume_ratio_l3255_325590


namespace profit_percentage_example_l3255_325581

/-- Calculate the profit percentage given selling price and cost price -/
def profit_percentage (selling_price : ℚ) (cost_price : ℚ) : ℚ :=
  ((selling_price - cost_price) / cost_price) * 100

/-- Theorem: The profit percentage is 25% when the selling price is 400 and the cost price is 320 -/
theorem profit_percentage_example : profit_percentage 400 320 = 25 := by
  sorry

end profit_percentage_example_l3255_325581


namespace power_product_equality_l3255_325529

theorem power_product_equality (a b : ℝ) : (a * b^2)^2 = a^2 * b^4 := by
  sorry

end power_product_equality_l3255_325529


namespace device_usage_probability_l3255_325557

theorem device_usage_probability (pA pB pC : ℝ) 
  (hA : pA = 0.4) 
  (hB : pB = 0.5) 
  (hC : pC = 0.7) 
  (hpA : 0 ≤ pA ∧ pA ≤ 1) 
  (hpB : 0 ≤ pB ∧ pB ≤ 1) 
  (hpC : 0 ≤ pC ∧ pC ≤ 1) :
  1 - (1 - pA) * (1 - pB) * (1 - pC) = 0.91 := by
  sorry

end device_usage_probability_l3255_325557


namespace complex_power_magnitude_l3255_325568

theorem complex_power_magnitude : 
  Complex.abs ((2 + 2 * Complex.I * Real.sqrt 3) ^ 6) = 4096 := by
  sorry

end complex_power_magnitude_l3255_325568


namespace gcd_lcm_product_24_60_l3255_325583

theorem gcd_lcm_product_24_60 : Nat.gcd 24 60 * Nat.lcm 24 60 = 1440 := by
  sorry

end gcd_lcm_product_24_60_l3255_325583


namespace expression_evaluation_l3255_325564

theorem expression_evaluation : 3^(1^(0^8)) + ((3^1)^0)^8 = 4 := by sorry

end expression_evaluation_l3255_325564


namespace max_points_world_cup_group_l3255_325528

/-- The maximum sum of points for all teams in a World Cup group stage -/
theorem max_points_world_cup_group (n : ℕ) (win_points tie_points : ℕ) : 
  n = 4 → win_points = 3 → tie_points = 1 → 
  (n.choose 2) * win_points = 18 :=
by sorry

end max_points_world_cup_group_l3255_325528


namespace equation_solution_range_l3255_325575

theorem equation_solution_range (m : ℝ) : 
  (∃ x : ℝ, x ≥ 0 ∧ x ≠ 1 ∧ (2 * x + m) / (x - 1) = 1) → 
  (m ≤ -1 ∧ m ≠ -2) :=
by sorry

end equation_solution_range_l3255_325575


namespace triangle_isosceles_l3255_325565

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  h_positive : a > 0 ∧ b > 0 ∧ c > 0
  h_angle : 0 < A ∧ A < π

-- Define the theorem
theorem triangle_isosceles (t : Triangle) 
  (h : Real.log (t.a^2) = Real.log (t.b^2) + Real.log (t.c^2) - Real.log (2 * t.b * t.c * Real.cos t.A)) :
  t.a = t.b ∨ t.a = t.c := by
  sorry


end triangle_isosceles_l3255_325565


namespace barkley_buried_bones_l3255_325520

/-- Proves that Barkley has buried 42 bones after 5 months -/
theorem barkley_buried_bones 
  (bones_per_month : ℕ) 
  (months_passed : ℕ) 
  (available_bones : ℕ) 
  (h1 : bones_per_month = 10)
  (h2 : months_passed = 5)
  (h3 : available_bones = 8) :
  bones_per_month * months_passed - available_bones = 42 := by
  sorry

end barkley_buried_bones_l3255_325520


namespace corgi_price_calculation_l3255_325500

theorem corgi_price_calculation (x : ℝ) : 
  (2 * (x + 0.3 * x) = 2600) → x = 1000 := by
  sorry

end corgi_price_calculation_l3255_325500


namespace diagonals_bisect_in_special_quadrilaterals_l3255_325533

-- Define a type for quadrilaterals
inductive Quadrilateral
  | Parallelogram
  | Rectangle
  | Rhombus
  | Square

-- Define a function to check if diagonals bisect each other
def diagonalsBisectEachOther (q : Quadrilateral) : Prop :=
  match q with
  | Quadrilateral.Parallelogram => true
  | Quadrilateral.Rectangle => true
  | Quadrilateral.Rhombus => true
  | Quadrilateral.Square => true

-- Theorem statement
theorem diagonals_bisect_in_special_quadrilaterals (q : Quadrilateral) :
  diagonalsBisectEachOther q := by
  sorry

end diagonals_bisect_in_special_quadrilaterals_l3255_325533


namespace shrub_height_after_two_years_l3255_325532

def shrub_height (initial_height : ℝ) (years : ℕ) : ℝ :=
  initial_height * (3 ^ years)

theorem shrub_height_after_two_years 
  (h : shrub_height (shrub_height 9 2) 3 = 243) : 
  shrub_height 9 2 = 9 :=
by
  sorry

#check shrub_height_after_two_years

end shrub_height_after_two_years_l3255_325532


namespace triangle_abc_properties_l3255_325561

theorem triangle_abc_properties (a b c A B C : ℝ) (h1 : c * Real.cos A = 5) 
  (h2 : a * Real.sin C = 4) (h3 : (1/2) * a * b * Real.sin C = 16) : 
  c = Real.sqrt 41 ∧ a + b + c = 13 + Real.sqrt 41 := by
  sorry

end triangle_abc_properties_l3255_325561


namespace quadratic_roots_after_modification_l3255_325506

theorem quadratic_roots_after_modification (a b t l : ℝ) :
  -1 < t → t < 0 →
  (∀ x, x^2 + a*x + b = 0 ↔ x = t ∨ x = l) →
  ∃ r₁ r₂, r₁ ≠ r₂ ∧ ∀ x, x^2 + (a+t)*x + (b+t) = 0 ↔ x = r₁ ∨ x = r₂ :=
by sorry

end quadratic_roots_after_modification_l3255_325506


namespace system_solution_l3255_325588

theorem system_solution :
  ∀ (x y : ℝ),
    (x * y^2 = 15 * x^2 + 17 * x * y + 15 * y^2) ∧
    (x^2 * y = 20 * x^2 + 3 * y^2) →
    ((x = 0 ∧ y = 0) ∨ (x = -19 ∧ y = -2)) := by
  sorry

end system_solution_l3255_325588


namespace car_speed_problem_l3255_325566

theorem car_speed_problem (distance : ℝ) (original_time : ℝ) (new_time_factor : ℝ) :
  distance = 450 ∧ original_time = 6 ∧ new_time_factor = 3/2 →
  (distance / (original_time * new_time_factor)) = 50 := by
  sorry

end car_speed_problem_l3255_325566


namespace coordinate_sum_of_point_d_l3255_325546

/-- Given a point C at (0, 0) and a point D on the line y = 5,
    if the slope of segment CD is 3/4,
    then the sum of the x- and y-coordinates of point D is 35/3. -/
theorem coordinate_sum_of_point_d (D : ℝ × ℝ) : 
  D.2 = 5 →                  -- D is on the line y = 5
  (D.2 - 0) / (D.1 - 0) = 3/4 →  -- slope of CD is 3/4
  D.1 + D.2 = 35/3 := by
sorry

end coordinate_sum_of_point_d_l3255_325546
