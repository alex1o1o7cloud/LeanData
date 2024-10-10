import Mathlib

namespace juice_drink_cost_l3787_378733

theorem juice_drink_cost (initial_amount : ℕ) (pizza_cost : ℕ) (pizza_quantity : ℕ) 
  (juice_quantity : ℕ) (return_amount : ℕ) : 
  initial_amount = 50 → 
  pizza_cost = 12 → 
  pizza_quantity = 2 → 
  juice_quantity = 2 → 
  return_amount = 22 → 
  (initial_amount - return_amount - pizza_cost * pizza_quantity) / juice_quantity = 2 :=
by sorry

end juice_drink_cost_l3787_378733


namespace radian_to_degree_conversion_l3787_378762

theorem radian_to_degree_conversion (π : ℝ) (h : π * (180 / π) = 180) :
  (23 / 12) * π * (180 / π) = 345 :=
sorry

end radian_to_degree_conversion_l3787_378762


namespace continuous_function_with_three_preimages_l3787_378705

theorem continuous_function_with_three_preimages :
  ∃ f : ℝ → ℝ, Continuous f ∧
    ∀ y : ℝ, ∃! (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
      f x₁ = y ∧ f x₂ = y ∧ f x₃ = y ∧
      ∀ x : ℝ, f x = y → x = x₁ ∨ x = x₂ ∨ x = x₃ :=
by
  sorry

end continuous_function_with_three_preimages_l3787_378705


namespace greatest_integer_fraction_is_integer_l3787_378727

theorem greatest_integer_fraction_is_integer : 
  ∀ y : ℤ, y > 12 → ¬(∃ k : ℤ, (y^2 - 3*y + 4) / (y - 4) = k) ∧ 
  ∃ k : ℤ, (12^2 - 3*12 + 4) / (12 - 4) = k := by
sorry

end greatest_integer_fraction_is_integer_l3787_378727


namespace floor_divisibility_l3787_378742

theorem floor_divisibility (n : ℕ) : 
  (2^(n+1) : ℤ) ∣ ⌊((1 : ℝ) + Real.sqrt 3)^(2*n+1)⌋ := by
  sorry

end floor_divisibility_l3787_378742


namespace sum_of_a_and_b_is_one_l3787_378764

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A (a b : ℝ) : Set ℝ := {x | (x^2 + a*x + b)*(x - 1) = 0}

-- Define the theorem
theorem sum_of_a_and_b_is_one 
  (B C : Set ℝ) 
  (a b : ℝ) 
  (h1 : A a b ∩ B = {1, 2})
  (h2 : A a b ∩ (C ∪ B) = {3}) :
  a + b = 1 := by
  sorry

end sum_of_a_and_b_is_one_l3787_378764


namespace average_temperature_proof_l3787_378723

theorem average_temperature_proof (temp_first_3_days : ℝ) (temp_thur_fri : ℝ) (temp_remaining : ℝ) :
  temp_first_3_days = 40 →
  temp_thur_fri = 80 →
  (3 * temp_first_3_days + 2 * temp_thur_fri + temp_remaining) / 7 = 60 := by
  sorry

#check average_temperature_proof

end average_temperature_proof_l3787_378723


namespace smaller_solution_of_quadratic_l3787_378791

theorem smaller_solution_of_quadratic (x : ℝ) :
  x^2 + 17*x - 60 = 0 ∧ ∀ y, y^2 + 17*y - 60 = 0 → x ≤ y →
  x = -20 :=
sorry

end smaller_solution_of_quadratic_l3787_378791


namespace orchids_in_vase_orchids_count_is_two_l3787_378794

theorem orchids_in_vase (initial_roses : ℕ) (initial_orchids : ℕ) 
  (current_roses : ℕ) (rose_orchid_difference : ℕ) : ℕ :=
  let current_orchids := current_roses - rose_orchid_difference
  current_orchids

#check orchids_in_vase 5 3 12 10

theorem orchids_count_is_two :
  orchids_in_vase 5 3 12 10 = 2 := by
  sorry

end orchids_in_vase_orchids_count_is_two_l3787_378794


namespace ab_equals_six_l3787_378751

theorem ab_equals_six (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end ab_equals_six_l3787_378751


namespace distance_to_yz_plane_l3787_378734

/-- The distance from a point to the yz-plane -/
def distToYZPlane (x y z : ℝ) : ℝ := |x|

/-- The distance from a point to the x-axis -/
def distToXAxis (x y z : ℝ) : ℝ := |y|

/-- Point P satisfies the given conditions -/
def satisfiesConditions (x y z : ℝ) : Prop :=
  y = -6 ∧ x^2 + z^2 = 36 ∧ distToXAxis x y z = (1/2) * distToYZPlane x y z

theorem distance_to_yz_plane (x y z : ℝ) 
  (h : satisfiesConditions x y z) : distToYZPlane x y z = 12 := by
  sorry

end distance_to_yz_plane_l3787_378734


namespace square_difference_eq_85_solutions_l3787_378744

theorem square_difference_eq_85_solutions : 
  ∃! n : ℕ, n = (Finset.filter (fun p : ℕ × ℕ => 
    p.1 > 0 ∧ p.2 > 0 ∧ p.1^2 - p.2^2 = 85) (Finset.product (Finset.range 1000) (Finset.range 1000))).card :=
by
  sorry

end square_difference_eq_85_solutions_l3787_378744


namespace x_over_y_value_l3787_378701

theorem x_over_y_value (x y : ℝ) (h : 16 * x = 0.24 * 90 * y) : x / y = 1.35 := by
  sorry

end x_over_y_value_l3787_378701


namespace square_difference_l3787_378719

theorem square_difference (a b : ℝ) (h1 : a + b = 10) (h2 : a - b = 4) :
  a^2 - b^2 = 40 := by sorry

end square_difference_l3787_378719


namespace table_tennis_cost_calculation_l3787_378792

/-- Represents the cost calculation for table tennis equipment purchase options. -/
def TableTennisCost (x : ℕ) : Prop :=
  (x > 20) →
  let racketPrice : ℕ := 80
  let ballPrice : ℕ := 20
  let racketCount : ℕ := 20
  let option1Cost : ℕ := racketPrice * racketCount + ballPrice * (x - racketCount)
  let option2Cost : ℕ := ((racketPrice * racketCount + ballPrice * x) * 9) / 10
  (option1Cost = 20 * x + 1200) ∧ (option2Cost = 18 * x + 1440)

/-- Theorem stating the cost calculation for both options is correct for any valid x. -/
theorem table_tennis_cost_calculation (x : ℕ) : TableTennisCost x := by
  sorry

end table_tennis_cost_calculation_l3787_378792


namespace range_of_a_l3787_378777

-- Define the propositions
def proposition_p (a : ℝ) : Prop :=
  ∀ m : ℝ, m ∈ Set.Icc (-1) 1 → a^2 - 5*a - 3 ≥ Real.sqrt (m^2 + 8)

def proposition_q (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + a*x + 2 < 0

-- State the theorem
theorem range_of_a :
  ∀ a : ℝ, proposition_p a ∧ ¬proposition_q a ↔ a ∈ Set.Icc (-2 * Real.sqrt 2) (-1) :=
sorry

end range_of_a_l3787_378777


namespace impossibleToRemoveAllPieces_l3787_378745

/-- Represents the color of a cell or piece -/
inductive Color
| Black
| White

/-- Represents a move on the board -/
structure Move where
  piece1 : Nat × Nat
  piece2 : Nat × Nat
  newPos1 : Nat × Nat
  newPos2 : Nat × Nat

/-- Represents the state of the board -/
structure BoardState where
  pieces : List (Nat × Nat)

/-- Returns the color of a cell given its coordinates -/
def cellColor (pos : Nat × Nat) : Color :=
  if (pos.1 + pos.2) % 2 == 0 then Color.Black else Color.White

/-- Checks if two positions are adjacent -/
def isAdjacent (pos1 pos2 : Nat × Nat) : Bool :=
  (pos1.1 == pos2.1 && (pos1.2 + 1 == pos2.2 || pos1.2 == pos2.2 + 1)) ||
  (pos1.2 == pos2.2 && (pos1.1 + 1 == pos2.1 || pos1.1 == pos2.1 + 1))

/-- Checks if a move is valid -/
def isValidMove (m : Move) : Bool :=
  isAdjacent m.piece1 m.newPos1 && isAdjacent m.piece2 m.newPos2

/-- Applies a move to the board state -/
def applyMove (state : BoardState) (m : Move) : BoardState :=
  sorry

/-- Theorem: It is impossible to remove all pieces from the board -/
theorem impossibleToRemoveAllPieces :
  ∀ (moves : List Move),
    let initialState : BoardState := { pieces := List.range 506 }
    let finalState := moves.foldl applyMove initialState
    finalState.pieces.length > 0 := by
  sorry

end impossibleToRemoveAllPieces_l3787_378745


namespace expand_and_simplify_l3787_378709

theorem expand_and_simplify (y : ℝ) : -3 * (y - 4) * (y + 9) = -3 * y^2 - 15 * y + 108 := by
  sorry

end expand_and_simplify_l3787_378709


namespace cos_600_degrees_l3787_378758

theorem cos_600_degrees : Real.cos (600 * π / 180) = -(1/2) := by
  sorry

end cos_600_degrees_l3787_378758


namespace smallest_whole_number_larger_than_triangle_perimeter_l3787_378728

theorem smallest_whole_number_larger_than_triangle_perimeter : 
  ∀ s : ℝ, 
  s > 0 → 
  7 + s > 17 → 
  17 + s > 7 → 
  7 + 17 > s → 
  48 > 7 + 17 + s ∧ 
  ∀ n : ℕ, n < 48 → ∃ t : ℝ, t > 0 ∧ 7 + t > 17 ∧ 17 + t > 7 ∧ 7 + 17 > t ∧ n ≤ 7 + 17 + t :=
by sorry

end smallest_whole_number_larger_than_triangle_perimeter_l3787_378728


namespace bike_distance_proof_l3787_378713

/-- Calculates the distance traveled given speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Proves that a bike traveling at 90 km/h for 5 hours covers 450 km -/
theorem bike_distance_proof :
  let speed : ℝ := 90
  let time : ℝ := 5
  distance speed time = 450 := by sorry

end bike_distance_proof_l3787_378713


namespace largest_angle_in_three_three_four_triangle_l3787_378714

/-- A triangle with interior angles in the ratio 3:3:4 has its largest angle measuring 72° -/
theorem largest_angle_in_three_three_four_triangle : 
  ∀ (a b c : ℝ), 
  a > 0 → b > 0 → c > 0 →
  a + b + c = 180 →
  a / 3 = b / 3 ∧ b / 3 = c / 4 →
  max a (max b c) = 72 := by
sorry

end largest_angle_in_three_three_four_triangle_l3787_378714


namespace parabola_homothety_transform_l3787_378715

/-- A homothety transformation centered at (0,0) with ratio k > 0 -/
structure Homothety where
  k : ℝ
  h_pos : k > 0

/-- The equation of a parabola in the form 2py = x^2 -/
def Parabola (p : ℝ) (x y : ℝ) : Prop := 2 * p * y = x^2

theorem parabola_homothety_transform (p : ℝ) (h_p : p ≠ 0) :
  ∃ (h : Homothety), ∀ (x y : ℝ),
    Parabola p x y ↔ y = x^2 := by
  sorry

end parabola_homothety_transform_l3787_378715


namespace largest_digit_sum_l3787_378773

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

theorem largest_digit_sum (a b c z : ℕ) : 
  is_digit a → is_digit b → is_digit c →
  (100 * a + 10 * b + c : ℚ) / 1000 = 1 / z →
  0 < z → z ≤ 12 →
  a + b + c ≤ 8 ∧ ∃ a' b' c' z', 
    is_digit a' ∧ is_digit b' ∧ is_digit c' ∧
    (100 * a' + 10 * b' + c' : ℚ) / 1000 = 1 / z' ∧
    0 < z' ∧ z' ≤ 12 ∧
    a' + b' + c' = 8 :=
by sorry

end largest_digit_sum_l3787_378773


namespace range_of_expression_l3787_378765

theorem range_of_expression (a b c : ℝ) 
  (h1 : -3 < b) (h2 : b < a) (h3 : a < -1) 
  (h4 : -2 < c) (h5 : c < -1) : 
  0 < (a - b) * c^2 ∧ (a - b) * c^2 < 8 := by
  sorry

end range_of_expression_l3787_378765


namespace total_orange_purchase_l3787_378739

def initial_purchase : ℕ := 10
def additional_purchase : ℕ := 5
def num_weeks : ℕ := 3
def doubling_weeks : ℕ := 2

theorem total_orange_purchase :
  let first_week := initial_purchase + additional_purchase
  let subsequent_weeks := 2 * first_week * doubling_weeks
  first_week + subsequent_weeks = 75 := by sorry

end total_orange_purchase_l3787_378739


namespace K_on_circle_S₂_l3787_378787

-- Define the circles and points
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def S : Circle := { center := (0, 0), radius := 2 }
def S₁ : Circle := { center := (1, 0), radius := 1 }
def S₂ : Circle := { center := (3, 0), radius := 1 }

def B : ℝ × ℝ := (2, 0)
def A : ℝ × ℝ := (2, 0)

-- Define the intersection point K
def K : ℝ × ℝ := sorry

-- Define the properties of the circles
def S₁_tangent_to_S : Prop :=
  (S₁.center.1 - S.center.1)^2 + (S₁.center.2 - S.center.2)^2 = (S.radius - S₁.radius)^2

def S₂_tangent_to_S₁ : Prop :=
  (S₂.center.1 - S₁.center.1)^2 + (S₂.center.2 - S₁.center.2)^2 = (S₁.radius + S₂.radius)^2

def S₂_not_tangent_to_S : Prop :=
  (S₂.center.1 - S.center.1)^2 + (S₂.center.2 - S.center.2)^2 ≠ (S.radius - S₂.radius)^2

def K_on_line_AB : Prop :=
  (K.2 - A.2) * (B.1 - A.1) = (K.1 - A.1) * (B.2 - A.2)

def K_on_circle_S : Prop :=
  (K.1 - S.center.1)^2 + (K.2 - S.center.2)^2 = S.radius^2

-- Theorem to prove
theorem K_on_circle_S₂ (h1 : S₁_tangent_to_S) (h2 : S₂_tangent_to_S₁) 
    (h3 : S₂_not_tangent_to_S) (h4 : K_on_line_AB) (h5 : K_on_circle_S) :
  (K.1 - S₂.center.1)^2 + (K.2 - S₂.center.2)^2 = S₂.radius^2 := by
  sorry

end K_on_circle_S₂_l3787_378787


namespace cow_daily_water_consumption_l3787_378770

/-- The number of cows on Mr. Reyansh's farm -/
def num_cows : ℕ := 40

/-- The ratio of sheep to cows on Mr. Reyansh's farm -/
def sheep_to_cow_ratio : ℕ := 10

/-- The ratio of water consumption of a sheep to a cow -/
def sheep_to_cow_water_ratio : ℚ := 1/4

/-- Total water usage for all animals in a week (in liters) -/
def total_weekly_water : ℕ := 78400

/-- The number of days in a week -/
def days_in_week : ℕ := 7

theorem cow_daily_water_consumption :
  ∃ (cow_daily_water : ℚ),
    cow_daily_water * (num_cows : ℚ) * days_in_week +
    cow_daily_water * sheep_to_cow_water_ratio * (num_cows * sheep_to_cow_ratio : ℚ) * days_in_week =
    total_weekly_water ∧
    cow_daily_water = 80 := by
  sorry

end cow_daily_water_consumption_l3787_378770


namespace emerson_first_part_distance_l3787_378775

/-- Emerson's rowing trip distances -/
structure RowingTrip where
  total : ℕ
  second : ℕ
  third : ℕ

/-- The distance covered in the first part of the rowing trip -/
def firstPartDistance (trip : RowingTrip) : ℕ :=
  trip.total - (trip.second + trip.third)

/-- Theorem: The first part distance of Emerson's specific trip is 6 miles -/
theorem emerson_first_part_distance :
  firstPartDistance ⟨39, 15, 18⟩ = 6 := by
  sorry

end emerson_first_part_distance_l3787_378775


namespace product_sum_squares_l3787_378732

theorem product_sum_squares (x y : ℝ) :
  x * y = 120 ∧ x^2 + y^2 = 289 → x + y = 22 ∨ x + y = -22 := by
  sorry

end product_sum_squares_l3787_378732


namespace problem_1_problem_2_l3787_378757

-- Problem 1
theorem problem_1 : (-1/3)⁻¹ - Real.sqrt 12 - (2 - Real.sqrt 3)^0 = -4 - 2 * Real.sqrt 3 := by
  sorry

-- Problem 2
theorem problem_2 : (1 + 1/2) + (2^2 - 1)/2 = 3 := by
  sorry

end problem_1_problem_2_l3787_378757


namespace quadratic_roots_ratio_l3787_378767

theorem quadratic_roots_ratio (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (∃ x₁ x₂ : ℝ, (x₁ + x₂ = -a ∧ x₁ * x₂ = b) ∧
               (2*x₁ + 2*x₂ = -b ∧ 4*x₁*x₂ = c)) →
  a / c = 1 / 8 :=
by sorry

end quadratic_roots_ratio_l3787_378767


namespace dog_grouping_theorem_l3787_378747

/-- The number of ways to divide 12 dogs into groups of 4, 6, and 2,
    with Fluffy in the 4-dog group and Nipper in the 6-dog group -/
def dog_grouping_ways : ℕ :=
  let total_dogs : ℕ := 12
  let group1_size : ℕ := 4
  let group2_size : ℕ := 6
  let group3_size : ℕ := 2
  let remaining_dogs : ℕ := total_dogs - 2  -- Fluffy and Nipper are already placed
  Nat.choose remaining_dogs (group1_size - 1) * Nat.choose (remaining_dogs - (group1_size - 1)) (group2_size - 1)

theorem dog_grouping_theorem : dog_grouping_ways = 2520 := by
  sorry

end dog_grouping_theorem_l3787_378747


namespace gcd_of_specific_numbers_l3787_378798

theorem gcd_of_specific_numbers : 
  let m : ℕ := 555555555
  let n : ℕ := 1111111111
  Nat.gcd m n = 1 := by
sorry

end gcd_of_specific_numbers_l3787_378798


namespace is_quadratic_equation_f_l3787_378790

-- Define a quadratic equation in one variable
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

-- Define the specific equation (x-1)(x+2)=1
def f (x : ℝ) : ℝ := (x - 1) * (x + 2) - 1

-- Theorem statement
theorem is_quadratic_equation_f : is_quadratic_equation f := by
  sorry

end is_quadratic_equation_f_l3787_378790


namespace smallest_palindrome_base2_and_base5_l3787_378718

/-- Checks if a natural number is a palindrome in the given base. -/
def is_palindrome (n : ℕ) (base : ℕ) : Prop := sorry

/-- Converts a natural number to its representation in the given base. -/
def to_base (n : ℕ) (base : ℕ) : List ℕ := sorry

/-- Checks if a natural number has exactly k digits in the given base. -/
def has_k_digits (n : ℕ) (k : ℕ) (base : ℕ) : Prop := sorry

theorem smallest_palindrome_base2_and_base5 :
  ∀ n : ℕ,
  (has_k_digits n 5 2 ∧ is_palindrome n 2 ∧ is_palindrome (to_base n 5).length 5) →
  n ≥ 27 :=
by sorry

end smallest_palindrome_base2_and_base5_l3787_378718


namespace hyperbola_eccentricity_l3787_378786

-- Define the hyperbola and its properties
def Hyperbola (a : ℝ) : Prop :=
  a > 0 ∧ ∃ (x y : ℝ), x^2 / a^2 - y^2 / 5 = 1

-- Define the asymptote
def Asymptote (a : ℝ) : Prop :=
  ∃ (x y : ℝ), y = (Real.sqrt 5 / 2) * x

-- Define eccentricity
def Eccentricity (e : ℝ) (a : ℝ) : Prop :=
  e = Real.sqrt (a^2 + 5) / a

-- Theorem statement
theorem hyperbola_eccentricity (a : ℝ) :
  Hyperbola a → Asymptote a → Eccentricity (3/2) a := by sorry

end hyperbola_eccentricity_l3787_378786


namespace binomial_fraction_is_integer_l3787_378774

theorem binomial_fraction_is_integer (k n : ℕ) (h1 : 1 ≤ k) (h2 : k < n) : 
  ∃ m : ℤ, (n - 2*k - 1 : ℚ) / (k + 1 : ℚ) * (n.choose k) = m := by
  sorry

end binomial_fraction_is_integer_l3787_378774


namespace quadratic_shift_theorem_l3787_378748

/-- The quadratic function y = x^2 - 4x - 3 -/
def f (x : ℝ) : ℝ := x^2 - 4*x - 3

/-- The shifted quadratic function y = x^2 - 4x - 3 + a -/
def f_shifted (x a : ℝ) : ℝ := f x + a

theorem quadratic_shift_theorem :
  /- The value of a that makes the parabola pass through (0,1) is 4 -/
  (∃ a : ℝ, f_shifted 0 a = 1 ∧ a = 4) ∧
  /- The values of a that make the parabola intersect the coordinate axes at exactly 2 points are 3 and 7 -/
  (∃ a₁ a₂ : ℝ, 
    ((f_shifted 0 a₁ = 0 ∨ (∃ x : ℝ, x ≠ 0 ∧ f_shifted x a₁ = 0)) ∧
     (∃! x : ℝ, f_shifted x a₁ = 0)) ∧
    ((f_shifted 0 a₂ = 0 ∨ (∃ x : ℝ, x ≠ 0 ∧ f_shifted x a₂ = 0)) ∧
     (∃! x : ℝ, f_shifted x a₂ = 0)) ∧
    a₁ = 3 ∧ a₂ = 7) :=
by sorry

end quadratic_shift_theorem_l3787_378748


namespace factor_expression_l3787_378752

theorem factor_expression (x : ℝ) : 5*x*(x+2) + 9*(x+2) = (x+2)*(5*x+9) := by
  sorry

end factor_expression_l3787_378752


namespace mixed_groups_count_l3787_378729

theorem mixed_groups_count (total_children : ℕ) (total_groups : ℕ) (group_size : ℕ) 
  (boy_boy_photos : ℕ) (girl_girl_photos : ℕ) :
  total_children = 300 →
  total_groups = 100 →
  group_size = 3 →
  total_children = total_groups * group_size →
  boy_boy_photos = 100 →
  girl_girl_photos = 56 →
  (∃ (mixed_groups : ℕ), 
    mixed_groups = 72 ∧
    mixed_groups * 2 + boy_boy_photos + girl_girl_photos = total_groups * group_size) :=
by sorry


end mixed_groups_count_l3787_378729


namespace parallel_vectors_imply_y_equals_one_l3787_378796

/-- Two vectors in ℝ² -/
def a : ℝ × ℝ := (-1, 2)
def b (y : ℝ) : ℝ × ℝ := (1, -2*y)

/-- Definition of parallel vectors in ℝ² -/
def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

/-- Theorem: If a and b(y) are parallel, then y = 1 -/
theorem parallel_vectors_imply_y_equals_one :
  parallel a (b y) → y = 1 := by
  sorry

end parallel_vectors_imply_y_equals_one_l3787_378796


namespace point_in_fourth_quadrant_m_range_l3787_378736

/-- A point in the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the fourth quadrant -/
def in_fourth_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- The main theorem -/
theorem point_in_fourth_quadrant_m_range (m : ℝ) :
  in_fourth_quadrant ⟨m + 3, m - 5⟩ ↔ -3 < m ∧ m < 5 := by
  sorry


end point_in_fourth_quadrant_m_range_l3787_378736


namespace male_students_count_l3787_378726

/-- Calculates the total number of male students in first and second year -/
def total_male_students (total_first_year : ℕ) (female_first_year : ℕ) (male_second_year : ℕ) : ℕ :=
  (total_first_year - female_first_year) + male_second_year

/-- Proves that the total number of male students in first and second year is 620 -/
theorem male_students_count : 
  total_male_students 695 329 254 = 620 := by
  sorry

end male_students_count_l3787_378726


namespace penguin_sea_horse_difference_l3787_378781

/-- Given a ratio of sea horses to penguins and the number of sea horses,
    calculate the difference between the number of penguins and sea horses. -/
theorem penguin_sea_horse_difference 
  (ratio_sea_horses : ℕ) 
  (ratio_penguins : ℕ) 
  (num_sea_horses : ℕ) 
  (h1 : ratio_sea_horses = 5) 
  (h2 : ratio_penguins = 11) 
  (h3 : num_sea_horses = 70) :
  (ratio_penguins * (num_sea_horses / ratio_sea_horses)) - num_sea_horses = 84 :=
by
  sorry

#check penguin_sea_horse_difference

end penguin_sea_horse_difference_l3787_378781


namespace downstream_speed_calculation_l3787_378759

/-- Represents the speed of a rower in different conditions -/
structure RowerSpeed where
  upstream : ℝ
  stillWater : ℝ
  downstream : ℝ

/-- Calculates the downstream speed of a rower given upstream and still water speeds -/
def calculateDownstreamSpeed (upstream stillWater : ℝ) : ℝ :=
  2 * stillWater - upstream

/-- Theorem stating that given the upstream and still water speeds, 
    the calculated downstream speed is correct -/
theorem downstream_speed_calculation 
  (speed : RowerSpeed) 
  (h1 : speed.upstream = 25)
  (h2 : speed.stillWater = 32) :
  speed.downstream = calculateDownstreamSpeed speed.upstream speed.stillWater ∧ 
  speed.downstream = 39 := by
  sorry

end downstream_speed_calculation_l3787_378759


namespace unique_solution_condition_l3787_378722

theorem unique_solution_condition (m : ℝ) : 
  (∃! x : ℝ, (3 * x + 4) * (x - 8) = -95 + m * x) ↔ 
  (m = -20 - 2 * Real.sqrt 189 ∨ m = -20 + 2 * Real.sqrt 189) := by
sorry

end unique_solution_condition_l3787_378722


namespace apple_count_equality_l3787_378708

/-- The number of apples Marin has -/
def marins_apples : ℕ := 3

/-- The number of apples David has -/
def davids_apples : ℕ := 3

/-- The difference between Marin's and David's apple counts -/
def apple_difference : ℤ := marins_apples - davids_apples

theorem apple_count_equality : apple_difference = 0 := by
  sorry

end apple_count_equality_l3787_378708


namespace digit_difference_in_base_d_l3787_378700

/-- Given two digits A and B in base d > 7 such that AB + AA = 174 in base d,
    prove that A - B = 3 in base d, assuming A > B. -/
theorem digit_difference_in_base_d (d A B : ℕ) : 
  d > 7 →
  A < d →
  B < d →
  A > B →
  (A * d + B) + (A * d + A) = 1 * d * d + 7 * d + 4 →
  A - B = 3 :=
sorry

end digit_difference_in_base_d_l3787_378700


namespace tank_capacity_l3787_378707

/-- Represents a cylindrical water tank -/
structure WaterTank where
  capacity : ℝ
  initial_volume : ℝ
  added_volume : ℝ
  final_volume : ℝ

/-- Theorem stating the conditions and the result to be proved -/
theorem tank_capacity (tank : WaterTank) 
  (h1 : tank.initial_volume = tank.capacity / 6)
  (h2 : tank.added_volume = 4)
  (h3 : tank.final_volume = tank.initial_volume + tank.added_volume)
  (h4 : tank.final_volume = tank.capacity / 5) :
  tank.capacity = 120 := by
  sorry

end tank_capacity_l3787_378707


namespace modular_inverse_13_mod_2000_l3787_378710

theorem modular_inverse_13_mod_2000 : ∃ x : ℤ, 0 ≤ x ∧ x < 2000 ∧ (13 * x) % 2000 = 1 :=
by
  use 1077
  sorry

end modular_inverse_13_mod_2000_l3787_378710


namespace quadratic_factorization_l3787_378704

theorem quadratic_factorization (C D : ℤ) :
  (∀ x, 15 * x^2 - 56 * x + 48 = (C * x - 8) * (D * x - 6)) →
  C * D + C = 18 := by
sorry

end quadratic_factorization_l3787_378704


namespace average_math_chem_score_l3787_378797

theorem average_math_chem_score (math physics chem : ℕ) : 
  math + physics = 40 →
  chem = physics + 20 →
  (math + chem) / 2 = 30 := by
sorry

end average_math_chem_score_l3787_378797


namespace reunion_handshakes_l3787_378720

/-- The number of handshakes when n boys each shake hands once with every other boy -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: For 9 boys at a reunion, where each boy shakes hands exactly once with each of the others, 
    the total number of handshakes is 36. -/
theorem reunion_handshakes : handshakes 9 = 36 := by
  sorry

end reunion_handshakes_l3787_378720


namespace union_complement_when_a_is_one_subset_iff_a_in_range_l3787_378799

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | 0 < 2*x + a ∧ 2*x + a ≤ 3}
def B : Set ℝ := {x | -1/2 < x ∧ x < 2}

-- Theorem 1
theorem union_complement_when_a_is_one :
  (Set.univ \ B) ∪ (A 1) = {x | x ≤ 1 ∨ x ≥ 2} := by sorry

-- Theorem 2
theorem subset_iff_a_in_range :
  ∀ a : ℝ, A a ⊆ B ↔ -1 < a ∧ a ≤ 1 := by sorry

end union_complement_when_a_is_one_subset_iff_a_in_range_l3787_378799


namespace quadratic_independent_of_x_squared_l3787_378760

/-- For a quadratic polynomial -3x^2 + mx^2 - x + 3, if its value is independent of the quadratic term of x, then m = 3 -/
theorem quadratic_independent_of_x_squared (m : ℝ) : 
  (∀ x : ℝ, ∃ k : ℝ, -3*x^2 + m*x^2 - x + 3 = -x + k) → m = 3 := by
  sorry

end quadratic_independent_of_x_squared_l3787_378760


namespace abs_neg_one_tenth_l3787_378711

theorem abs_neg_one_tenth : |(-1/10 : ℚ)| = 1/10 := by
  sorry

end abs_neg_one_tenth_l3787_378711


namespace third_number_in_list_l3787_378738

theorem third_number_in_list (a b c d e : ℕ) : 
  a = 60 → 
  e = 300 → 
  a * b * c = 810000 → 
  b * c * d = 2430000 → 
  c * d * e = 8100000 → 
  c = 150 := by
sorry

end third_number_in_list_l3787_378738


namespace less_than_reciprocal_check_l3787_378735

def is_less_than_reciprocal (x : ℚ) : Prop :=
  x ≠ 0 ∧ x < 1 / x

theorem less_than_reciprocal_check :
  is_less_than_reciprocal (-3) ∧
  is_less_than_reciprocal (3/4) ∧
  ¬is_less_than_reciprocal (-1/2) ∧
  ¬is_less_than_reciprocal 3 ∧
  ¬is_less_than_reciprocal 0 :=
by sorry

end less_than_reciprocal_check_l3787_378735


namespace bird_sale_problem_l3787_378785

theorem bird_sale_problem (x y : ℝ) :
  x > 0 ∧ y > 0 ∧             -- Both purchase prices are positive
  0.8 * x = 1.2 * y ∧         -- Both birds sold for the same price
  (0.8 * x - x) + (1.2 * y - y) = -10 -- Total loss is 10 units
  →
  x = 30 ∧ y = 20 ∧ 0.8 * x = 24 := by
sorry

end bird_sale_problem_l3787_378785


namespace prism_volume_l3787_378795

/-- The volume of a right rectangular prism with given face areas -/
theorem prism_volume (side_area front_area bottom_area : ℝ) 
  (h_side : side_area = 20)
  (h_front : front_area = 12)
  (h_bottom : bottom_area = 8) :
  ∃ (a b c : ℝ), 
    a * b = side_area ∧ 
    b * c = front_area ∧ 
    a * c = bottom_area ∧ 
    a * b * c = 8 * Real.sqrt 30 := by
  sorry

end prism_volume_l3787_378795


namespace downstream_distance_l3787_378756

-- Define the given parameters
def boat_speed : ℝ := 22
def stream_speed : ℝ := 5
def travel_time : ℝ := 3

-- Define the theorem
theorem downstream_distance :
  boat_speed + stream_speed * travel_time = 81 := by
  sorry

end downstream_distance_l3787_378756


namespace parabola_properties_l3787_378712

/-- Parabola C: y² = 2px with focus F(2,0) and point A(6,3) -/
def Parabola (p : ℝ) : Set (ℝ × ℝ) :=
  {point : ℝ × ℝ | point.2^2 = 2 * p * point.1}

def F : ℝ × ℝ := (2, 0)
def A : ℝ × ℝ := (6, 3)

/-- The value of p for the given parabola -/
def p_value : ℝ := 4

/-- The minimum value of |MA| + |MF| where M is on the parabola -/
def min_distance : ℝ := 8

theorem parabola_properties :
  ∃ (p : ℝ), p = p_value ∧
  (∀ (M : ℝ × ℝ), M ∈ Parabola p →
    ∀ (d : ℝ), d = Real.sqrt ((M.1 - A.1)^2 + (M.2 - A.2)^2) + Real.sqrt ((M.1 - F.1)^2 + (M.2 - F.2)^2) →
    d ≥ min_distance) :=
sorry

end parabola_properties_l3787_378712


namespace linear_function_sum_l3787_378776

/-- A linear function f with specific properties -/
def f (x : ℝ) : ℝ := sorry

/-- The sum of f(2), f(4), ..., f(2n) -/
def sum_f (n : ℕ) : ℝ := sorry

theorem linear_function_sum :
  (f 0 = 1) →
  (∃ r : ℝ, f 1 * r = f 4 ∧ f 4 * r = f 13) →
  (∀ x y : ℝ, f (x + y) = f x + f y - f 0) →
  ∀ n : ℕ, sum_f n = n * (2 * n + 3) :=
sorry

end linear_function_sum_l3787_378776


namespace ashton_initial_boxes_l3787_378793

/-- The number of pencils in each box -/
def pencils_per_box : ℕ := 14

/-- The number of pencils Ashton gave to his brother -/
def pencils_given : ℕ := 6

/-- The number of pencils Ashton had left after giving some away -/
def pencils_left : ℕ := 22

/-- The number of boxes Ashton had initially -/
def initial_boxes : ℕ := 2

theorem ashton_initial_boxes :
  initial_boxes * pencils_per_box = pencils_left + pencils_given :=
sorry

end ashton_initial_boxes_l3787_378793


namespace existence_of_unsolvable_degree_l3787_378755

-- Define a polynomial equation of degree n
def PolynomialEquation (n : ℕ) := ℕ → ℝ → Prop

-- Define a solution expressed in terms of radicals
def RadicalSolution (n : ℕ) := ℕ → ℝ → Prop

-- Axiom: Quadratic equations have solutions in terms of radicals
axiom quadratic_solvable : ∀ (eq : PolynomialEquation 2), ∃ (sol : RadicalSolution 2), sol 2 = eq 2

-- Axiom: Cubic equations have solutions in terms of radicals
axiom cubic_solvable : ∀ (eq : PolynomialEquation 3), ∃ (sol : RadicalSolution 3), sol 3 = eq 3

-- Axiom: Quartic equations have solutions in terms of radicals
axiom quartic_solvable : ∀ (eq : PolynomialEquation 4), ∃ (sol : RadicalSolution 4), sol 4 = eq 4

-- Theorem: There exists a degree n such that not all polynomial equations of degree ≥ n are solvable by radicals
theorem existence_of_unsolvable_degree :
  ∃ (n : ℕ), ∀ (m : ℕ), m ≥ n → ¬(∀ (eq : PolynomialEquation m), ∃ (sol : RadicalSolution m), sol m = eq m) :=
sorry

end existence_of_unsolvable_degree_l3787_378755


namespace intersection_complement_equality_l3787_378780

open Set

def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 4}
def B : Set ℝ := {x | 3 ≤ x ∧ x ≤ 5}

theorem intersection_complement_equality :
  A ∩ (Set.univ \ B) = {x | -1 ≤ x ∧ x < 3} := by sorry

end intersection_complement_equality_l3787_378780


namespace books_bought_proof_l3787_378783

/-- Calculates the number of books bought given a ratio and total items -/
def books_bought (book_ratio pen_ratio notebook_ratio total_items : ℕ) : ℕ :=
  let total_ratio := book_ratio + pen_ratio + notebook_ratio
  let sets := total_items / total_ratio
  book_ratio * sets

/-- Theorem: Given the specified ratio and total items, prove the number of books bought -/
theorem books_bought_proof :
  books_bought 7 3 2 600 = 350 := by
  sorry

end books_bought_proof_l3787_378783


namespace cosine_expression_equals_negative_one_l3787_378743

theorem cosine_expression_equals_negative_one :
  (Real.cos (64 * π / 180) * Real.cos (4 * π / 180) - Real.cos (86 * π / 180) * Real.cos (26 * π / 180)) /
  (Real.cos (71 * π / 180) * Real.cos (41 * π / 180) - Real.cos (49 * π / 180) * Real.cos (19 * π / 180)) = -1 := by
  sorry

end cosine_expression_equals_negative_one_l3787_378743


namespace local_minimum_of_f_l3787_378746

/-- The function f(x) defined as (x^2 + ax - 1)e^(x-1) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x^2 + a*x - 1) * Real.exp (x - 1)

/-- The derivative of f(x) with respect to x -/
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := (x^2 + x - 2) * Real.exp (x - 1)

theorem local_minimum_of_f (a : ℝ) :
  (f_deriv a (-2) = 0) →  -- x = -2 is a point of extremum
  (∃ (x : ℝ), x > -2 ∧ x < 1 ∧ ∀ y, y > -2 ∧ y < 1 → f a x ≤ f a y) ∧ -- local minimum exists
  (f a 1 = -1) -- the local minimum value is -1
  := by sorry

end local_minimum_of_f_l3787_378746


namespace ratio_of_P_and_Q_l3787_378750

-- Define the equation as a function
def equation (P Q : ℤ) (x : ℝ) : Prop :=
  (P / (x + 6) + Q / (x^2 - 5*x) = (x^2 - x + 15) / (x^3 + x^2 - 30*x)) ∧ 
  (x ≠ -6) ∧ (x ≠ 0) ∧ (x ≠ 5)

-- State the theorem
theorem ratio_of_P_and_Q (P Q : ℤ) :
  (∀ x : ℝ, equation P Q x) → Q / P = 5 / 6 := by
  sorry

end ratio_of_P_and_Q_l3787_378750


namespace subset_implies_a_geq_one_l3787_378703

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 + 2*x - 3 < 0}
def B (a : ℝ) : Set ℝ := {x : ℝ | x - a < 0}

-- State the theorem
theorem subset_implies_a_geq_one (a : ℝ) : A ⊆ B a → a ≥ 1 := by
  sorry

end subset_implies_a_geq_one_l3787_378703


namespace f_properties_l3787_378782

noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 6 + Real.cos x ^ 6

theorem f_properties :
  (∀ x, f x ∈ Set.Icc (1/4 : ℝ) 1) ∧
  (∀ ε > 0, ∃ p ∈ Set.Ioo 0 ε, ∀ x, f (x + p) = f x) ∧
  (∀ k : ℤ, ∀ x, f (k * Real.pi / 4 - x) = f (k * Real.pi / 4 + x)) ∧
  (∀ k : ℤ, f (Real.pi / 8 + k * Real.pi / 4) = 5/8) :=
sorry

end f_properties_l3787_378782


namespace average_english_score_of_dropped_students_l3787_378730

/-- Represents the problem of calculating the average English quiz score of dropped students -/
theorem average_english_score_of_dropped_students
  (total_students : ℕ)
  (remaining_students : ℕ)
  (initial_average : ℝ)
  (new_average : ℝ)
  (h1 : total_students = 16)
  (h2 : remaining_students = 13)
  (h3 : initial_average = 62.5)
  (h4 : new_average = 62.0) :
  let dropped_students := total_students - remaining_students
  let total_score := total_students * initial_average
  let remaining_score := remaining_students * new_average
  let dropped_score := total_score - remaining_score
  abs ((dropped_score / dropped_students) - 64.67) < 0.01 := by
  sorry

#check average_english_score_of_dropped_students

end average_english_score_of_dropped_students_l3787_378730


namespace same_color_plate_probability_l3787_378716

theorem same_color_plate_probability : 
  let total_plates : ℕ := 7 + 5
  let red_plates : ℕ := 7
  let blue_plates : ℕ := 5
  let total_combinations : ℕ := Nat.choose total_plates 3
  let red_combinations : ℕ := Nat.choose red_plates 3
  let blue_combinations : ℕ := Nat.choose blue_plates 3
  let same_color_combinations : ℕ := red_combinations + blue_combinations
  (same_color_combinations : ℚ) / total_combinations = 9 / 44 := by
sorry

end same_color_plate_probability_l3787_378716


namespace exists_real_geq_3_is_particular_l3787_378761

-- Define what a particular proposition is
def is_particular_proposition (p : Prop) : Prop :=
  ∃ (x : Type), p = ∃ (y : x), true

-- State the theorem
theorem exists_real_geq_3_is_particular : 
  is_particular_proposition (∃ (x : ℝ), x ≥ 3) :=
sorry

end exists_real_geq_3_is_particular_l3787_378761


namespace tangent_parallel_implies_a_equals_one_l3787_378769

/-- The function f(x) = 3x + ax^3 -/
def f (a : ℝ) (x : ℝ) : ℝ := 3 * x + a * x^3

/-- The derivative of f(x) -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3 + 3 * a * x^2

theorem tangent_parallel_implies_a_equals_one (a : ℝ) :
  f_derivative a 1 = 6 → a = 1 := by sorry

end tangent_parallel_implies_a_equals_one_l3787_378769


namespace product_odd_implies_sum_odd_l3787_378763

theorem product_odd_implies_sum_odd (a b c : ℤ) : 
  Odd (a * b * c) → Odd (a + b + c) := by
  sorry

end product_odd_implies_sum_odd_l3787_378763


namespace area_ratio_midpoint_quadrilateral_l3787_378737

/-- A quadrilateral in a 2D plane --/
structure Quadrilateral where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- The area of a quadrilateral --/
def area (q : Quadrilateral) : ℝ := sorry

/-- The quadrilateral formed by the midpoints of another quadrilateral's sides --/
def midpointQuadrilateral (q : Quadrilateral) : Quadrilateral := sorry

/-- Theorem: The area of a quadrilateral is twice the area of its midpoint quadrilateral --/
theorem area_ratio_midpoint_quadrilateral (q : Quadrilateral) : 
  area q = 2 * area (midpointQuadrilateral q) := by sorry

end area_ratio_midpoint_quadrilateral_l3787_378737


namespace combine_terms_mn_zero_l3787_378717

theorem combine_terms_mn_zero (a b : ℝ) (m n : ℤ) :
  (∃ k : ℝ, ∃ p q : ℤ, -2 * a^m * b^4 + 5 * a^(n+2) * b^(2*m+n) = k * a^p * b^q) →
  m * n = 0 :=
sorry

end combine_terms_mn_zero_l3787_378717


namespace eggs_needed_proof_l3787_378766

def recipe_eggs : ℕ := 2
def recipe_people : ℕ := 4
def target_people : ℕ := 8
def available_eggs : ℕ := 3

theorem eggs_needed_proof : 
  (target_people / recipe_people * recipe_eggs) - available_eggs = 1 := by
sorry

end eggs_needed_proof_l3787_378766


namespace triangle_area_theorem_l3787_378725

theorem triangle_area_theorem (x : ℝ) (h1 : x > 0) : 
  (1/2 : ℝ) * x * (5*x) = 36 → x = 3.6 * Real.sqrt 5 := by
  sorry

end triangle_area_theorem_l3787_378725


namespace isosceles_triangle_leg_length_l3787_378721

/-- Given an isosceles triangle ABC with area √3/2 and sin(A) = √3 * sin(B),
    prove that the length of one of the legs is √2. -/
theorem isosceles_triangle_leg_length 
  (A B C : Real) -- Angles of the triangle
  (a b c : Real) -- Sides of the triangle
  (h : Real) -- Height of the triangle
  (area : Real) -- Area of the triangle
  (is_isosceles : b = c) -- Triangle is isosceles
  (area_value : area = Real.sqrt 3 / 2) -- Area is √3/2
  (sin_relation : Real.sin A = Real.sqrt 3 * Real.sin B) -- sin(A) = √3 * sin(B)
  : b = Real.sqrt 2 := by
  sorry

end isosceles_triangle_leg_length_l3787_378721


namespace binary_addition_subtraction_l3787_378740

/-- Converts a binary number (represented as a list of bits) to its decimal equivalent. -/
def binary_to_decimal (bits : List Bool) : Nat :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The given binary numbers -/
def b1 : List Bool := [true, false, true, true]    -- 1101₂
def b2 : List Bool := [true, true, true]           -- 111₂
def b3 : List Bool := [false, true, true, true]    -- 1110₂
def b4 : List Bool := [true, false, false, true]   -- 1001₂
def b5 : List Bool := [false, true, false, true]   -- 1010₂

/-- The result binary number -/
def result : List Bool := [true, false, false, true, true]  -- 11001₂

theorem binary_addition_subtraction :
  binary_to_decimal b1 + binary_to_decimal b2 - binary_to_decimal b3 + 
  binary_to_decimal b4 + binary_to_decimal b5 = binary_to_decimal result := by
  sorry

end binary_addition_subtraction_l3787_378740


namespace investment_sum_l3787_378779

/-- Proves that given a sum P invested at 18% p.a. for two years generates Rs. 600 more interest
    than if invested at 12% p.a. for the same period, then P = 5000. -/
theorem investment_sum (P : ℚ) : 
  (P * 18 * 2 / 100) - (P * 12 * 2 / 100) = 600 → P = 5000 := by
  sorry

end investment_sum_l3787_378779


namespace cubic_three_roots_range_l3787_378724

/-- The cubic polynomial function -/
def f (x : ℝ) := x^3 - 6*x^2 + 9*x

/-- The derivative of f -/
def f' (x : ℝ) := 3*x^2 - 12*x + 9

/-- Theorem: The range of m for which x^3 - 6x^2 + 9x + m = 0 has exactly three distinct real roots is (-4, 0) -/
theorem cubic_three_roots_range :
  ∀ m : ℝ, (∃! (r₁ r₂ r₃ : ℝ), r₁ ≠ r₂ ∧ r₂ ≠ r₃ ∧ r₁ ≠ r₃ ∧ 
    f r₁ + m = 0 ∧ f r₂ + m = 0 ∧ f r₃ + m = 0) ↔ 
  -4 < m ∧ m < 0 :=
sorry

end cubic_three_roots_range_l3787_378724


namespace book_dimensions_and_area_l3787_378741

/-- Represents the dimensions and surface area of a book. -/
structure Book where
  L : ℝ  -- Length
  W : ℝ  -- Width
  T : ℝ  -- Thickness
  A1 : ℝ  -- Area of front cover
  A2 : ℝ  -- Area of spine
  S : ℝ  -- Total surface area

/-- Theorem stating the width and total surface area of a book with given dimensions. -/
theorem book_dimensions_and_area (b : Book) 
  (hL : b.L = 5)
  (hT : b.T = 2)
  (hA1 : b.A1 = 50)
  (hA1_eq : b.A1 = b.L * b.W)
  (hA2_eq : b.A2 = b.T * b.W)
  (hS_eq : b.S = 2 * b.A1 + b.A2 + 2 * (b.L * b.T)) :
  b.W = 10 ∧ b.S = 140 := by
  sorry

#check book_dimensions_and_area

end book_dimensions_and_area_l3787_378741


namespace ten_thousandths_place_of_seven_fortieths_l3787_378754

theorem ten_thousandths_place_of_seven_fortieths (n : ℕ) : 
  (7 : ℚ) / 40 * 10000 - ((7 : ℚ) / 40 * 10000).floor = (0 : ℚ) / 10 := by
  sorry

end ten_thousandths_place_of_seven_fortieths_l3787_378754


namespace rectangle_max_area_l3787_378772

/-- Given a rectangle with perimeter 30 inches and one side 3 inches longer than the other,
    the maximum possible area is 54 square inches. -/
theorem rectangle_max_area :
  ∀ x : ℝ,
  x > 0 →
  2 * (x + (x + 3)) = 30 →
  x * (x + 3) ≤ 54 :=
by
  sorry

end rectangle_max_area_l3787_378772


namespace valid_triplets_eq_solution_set_l3787_378784

def is_valid_triplet (a b c : ℕ) : Prop :=
  a ≥ 2 ∧ b ≥ 2 ∧ c ≥ 2 ∧ (a * b * c - 1) % ((a - 1) * (b - 1) * (c - 1)) = 0

def valid_triplets : Set (ℕ × ℕ × ℕ) :=
  {t | is_valid_triplet t.1 t.2.1 t.2.2}

def solution_set : Set (ℕ × ℕ × ℕ) :=
  {(3, 5, 15), (3, 15, 5), (2, 4, 8), (2, 8, 4), (2, 2, 4), (2, 4, 2), (2, 2, 2)}

theorem valid_triplets_eq_solution_set : valid_triplets = solution_set := by
  sorry

end valid_triplets_eq_solution_set_l3787_378784


namespace right_triangle_trig_identity_l3787_378749

theorem right_triangle_trig_identity (A B C : Real) : 
  -- ABC is a right-angled triangle with right angle at C
  0 ≤ A ∧ 0 ≤ B ∧ 0 ≤ C ∧ 
  A + B + C = π / 2 ∧ 
  C = π / 2 →
  -- The trigonometric identity
  Real.sin A * Real.sin B * Real.sin (A - B) + 
  Real.sin B * Real.sin C * Real.sin (B - C) + 
  Real.sin C * Real.sin A * Real.sin (C - A) + 
  Real.sin (A - B) * Real.sin (B - C) * Real.sin (C - A) = 0 := by
sorry

end right_triangle_trig_identity_l3787_378749


namespace inequality_system_solution_l3787_378788

-- Define the inequality system
def inequality_system (x : ℝ) : Prop :=
  (2 * x - 1 > 5) ∧ (-x < -6)

-- Define the solution set
def solution_set : Set ℝ :=
  {x | x > 6}

-- Theorem stating that the solution set is correct
theorem inequality_system_solution :
  ∀ x : ℝ, inequality_system x ↔ x ∈ solution_set :=
sorry

end inequality_system_solution_l3787_378788


namespace intersection_of_M_and_N_l3787_378789

def M : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 1}
def N : Set ℝ := {y : ℝ | ∃ x : ℝ, y = 2 * x ∧ -1 ≤ x ∧ x ≤ 1}

theorem intersection_of_M_and_N : M ∩ N = {x : ℝ | -1 ≤ x ∧ x ≤ 1} := by
  sorry

end intersection_of_M_and_N_l3787_378789


namespace range_of_sum_l3787_378753

theorem range_of_sum (x y : ℝ) (h : x^2 + x + y^2 + y = 0) :
  ∃ (z : ℝ), z = x + y ∧ -2 ≤ z ∧ z ≤ 0 :=
by sorry

end range_of_sum_l3787_378753


namespace food_cost_calculation_l3787_378778

def hospital_bill_breakdown (total : ℝ) (medication_percent : ℝ) (overnight_percent : ℝ) (ambulance : ℝ) : ℝ := 
  let medication := medication_percent * total
  let remaining_after_medication := total - medication
  let overnight := overnight_percent * remaining_after_medication
  let food := total - medication - overnight - ambulance
  food

theorem food_cost_calculation :
  hospital_bill_breakdown 5000 0.5 0.25 1700 = 175 := by
  sorry

end food_cost_calculation_l3787_378778


namespace grocer_sales_problem_l3787_378706

theorem grocer_sales_problem (m1 m3 m4 m5 m6 avg : ℕ) (h1 : m1 = 4000) (h3 : m3 = 5689) (h4 : m4 = 7230) (h5 : m5 = 6000) (h6 : m6 = 12557) (havg : avg = 7000) :
  ∃ m2 : ℕ, m2 = 6524 ∧ (m1 + m2 + m3 + m4 + m5 + m6) / 6 = avg :=
sorry

end grocer_sales_problem_l3787_378706


namespace xy_value_l3787_378771

theorem xy_value (x y : ℝ) (h : |x - y + 1| + (y + 5)^2 = 0) : x * y = 30 := by
  sorry

end xy_value_l3787_378771


namespace chess_club_boys_l3787_378731

theorem chess_club_boys (total_members : ℕ) (total_attendees : ℕ) :
  total_members = 30 →
  total_attendees = 18 →
  ∃ (boys girls : ℕ),
    boys + girls = total_members ∧
    boys + (2 * girls / 3) = total_attendees ∧
    boys = 6 := by
  sorry

end chess_club_boys_l3787_378731


namespace middle_managers_sample_size_l3787_378702

/-- Calculates the number of individuals to be selected from a stratum in stratified sampling -/
def stratified_sample_size (total_population : ℕ) (total_sample : ℕ) (stratum_size : ℕ) : ℕ :=
  (total_sample * stratum_size) / total_population

/-- Proves that the number of middle managers to be selected is 6 -/
theorem middle_managers_sample_size :
  stratified_sample_size 160 32 30 = 6 := by
  sorry

#eval stratified_sample_size 160 32 30

end middle_managers_sample_size_l3787_378702


namespace range_of_q_l3787_378768

-- Define the function q(x)
def q (x : ℝ) : ℝ := (x^2 + 2)^3

-- State the theorem
theorem range_of_q : 
  {y : ℝ | ∃ x : ℝ, x ≥ 0 ∧ q x = y} = {y : ℝ | y ≥ 8} := by
  sorry

end range_of_q_l3787_378768
