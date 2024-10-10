import Mathlib

namespace complex_magnitude_problem_l3788_378827

theorem complex_magnitude_problem : ∃ (T : ℂ), 
  T = (1 + Complex.I)^19 + (1 + Complex.I)^19 - (1 - Complex.I)^19 ∧ 
  Complex.abs T = Real.sqrt 5 * 2^(19/2) := by
  sorry

end complex_magnitude_problem_l3788_378827


namespace smallest_addition_for_multiple_of_five_l3788_378847

theorem smallest_addition_for_multiple_of_five : 
  ∃ (n : ℕ), n > 0 ∧ (726 + n) % 5 = 0 ∧ ∀ (m : ℕ), m > 0 ∧ (726 + m) % 5 = 0 → n ≤ m :=
by sorry

end smallest_addition_for_multiple_of_five_l3788_378847


namespace completing_square_equivalence_l3788_378875

theorem completing_square_equivalence :
  ∀ x : ℝ, x^2 + 8*x + 9 = 0 ↔ (x + 4)^2 = 7 :=
by sorry

end completing_square_equivalence_l3788_378875


namespace quadratic_minimum_minimum_value_l3788_378890

theorem quadratic_minimum (x y : ℝ) :
  let M := 4 * x^2 - 12 * x * y + 10 * y^2 + 4 * y + 9
  ∀ a b, M ≥ (4 * a^2 - 12 * a * b + 10 * b^2 + 4 * b + 9) → a = -3 ∧ b = -2 :=
by
  sorry

theorem minimum_value :
  ∃ x y, 4 * x^2 - 12 * x * y + 10 * y^2 + 4 * y + 9 = 5 :=
by
  sorry

end quadratic_minimum_minimum_value_l3788_378890


namespace additional_members_needed_club_membership_increase_l3788_378819

/-- The number of additional members needed for a club to reach its desired membership. -/
theorem additional_members_needed (current_members : ℕ) (desired_members : ℕ) : ℕ :=
  desired_members - current_members

/-- Proof that the club needs 15 additional members. -/
theorem club_membership_increase : additional_members_needed 10 25 = 15 := by
  -- The proof goes here
  sorry

#check club_membership_increase

end additional_members_needed_club_membership_increase_l3788_378819


namespace square_root_equation_l3788_378814

theorem square_root_equation (n : ℝ) : 3 * Real.sqrt (8 + n) = 15 → n = 17 := by
  sorry

end square_root_equation_l3788_378814


namespace mall_meal_pairs_l3788_378893

/-- The number of distinct pairs of meals for two people, given the number of options for each meal component. -/
def distinct_meal_pairs (num_entrees num_drinks num_desserts : ℕ) : ℕ :=
  let total_meals := num_entrees * num_drinks * num_desserts
  total_meals * (total_meals - 1)

/-- Theorem stating that the number of distinct meal pairs is 1260 given the specific options. -/
theorem mall_meal_pairs :
  distinct_meal_pairs 4 3 3 = 1260 := by
  sorry

end mall_meal_pairs_l3788_378893


namespace tv_sale_increase_l3788_378801

theorem tv_sale_increase (original_price original_quantity : ℝ) 
  (h_price_reduction : ℝ) (h_net_effect : ℝ) :
  h_price_reduction = 0.2 →
  h_net_effect = 0.44000000000000014 →
  ∃ (new_quantity : ℝ),
    (1 - h_price_reduction) * original_price * new_quantity = 
      (1 + h_net_effect) * original_price * original_quantity ∧
    (new_quantity / original_quantity - 1) * 100 = 80 :=
by sorry

end tv_sale_increase_l3788_378801


namespace last_locker_opened_l3788_378861

/-- Represents the locker opening pattern described in the problem -/
def lockerOpeningPattern (n : ℕ) : ℕ → Prop :=
  sorry

/-- The number of lockers -/
def totalLockers : ℕ := 2048

/-- Theorem stating that the last locker opened is number 2046 -/
theorem last_locker_opened :
  ∃ (last : ℕ), last = 2046 ∧ 
  (∀ (k : ℕ), k ≤ totalLockers → lockerOpeningPattern totalLockers k → k ≤ last) ∧
  lockerOpeningPattern totalLockers last :=
sorry

end last_locker_opened_l3788_378861


namespace rectangle_properties_l3788_378834

/-- The equation representing the roots of the rectangle's sides -/
def side_equation (m x : ℝ) : Prop := x^2 - m*x + m/2 - 1/4 = 0

/-- The condition for the rectangle to be a square -/
def is_square (m : ℝ) : Prop := ∃ x : ℝ, side_equation m x ∧ ∀ y : ℝ, side_equation m y → y = x

/-- The perimeter of the rectangle given one side length -/
def perimeter (ab bc : ℝ) : ℝ := 2 * (ab + bc)

theorem rectangle_properties :
  (∃ m : ℝ, is_square m ∧ ∃ x : ℝ, side_equation m x ∧ x = 1/2) ∧
  (∃ m : ℝ, side_equation m 2 ∧ ∃ bc : ℝ, side_equation m bc ∧ perimeter 2 bc = 5) :=
by sorry

end rectangle_properties_l3788_378834


namespace division_problem_l3788_378830

theorem division_problem : (64 : ℝ) / 0.08 = 800 := by sorry

end division_problem_l3788_378830


namespace train_length_proof_l3788_378825

/-- Given a train with a speed of 40 km/hr that crosses a post in 18 seconds,
    prove that its length is approximately 200 meters. -/
theorem train_length_proof (speed : ℝ) (time : ℝ) (length : ℝ) : 
  speed = 40 → -- speed in km/hr
  time = 18 → -- time in seconds
  length = speed * (1000 / 3600) * time →
  ∃ ε > 0, |length - 200| < ε :=
by
  sorry

end train_length_proof_l3788_378825


namespace trig_equation_solution_l3788_378851

theorem trig_equation_solution (x : Real) :
  0 < x ∧ x < 180 →
  Real.tan ((150 - x) * Real.pi / 180) = 
    (Real.sin (150 * Real.pi / 180) - Real.sin (x * Real.pi / 180)) / 
    (Real.cos (150 * Real.pi / 180) - Real.cos (x * Real.pi / 180)) →
  x = 100 := by
  sorry

end trig_equation_solution_l3788_378851


namespace macaroon_packing_l3788_378895

/-- The number of brown bags used to pack macaroons -/
def number_of_bags : ℕ := 4

/-- The total number of macaroons -/
def total_macaroons : ℕ := 12

/-- The weight of each macaroon in ounces -/
def macaroon_weight : ℕ := 5

/-- The remaining weight of macaroons after one bag is eaten, in ounces -/
def remaining_weight : ℕ := 45

theorem macaroon_packing :
  (total_macaroons % number_of_bags = 0) ∧
  (total_macaroons / number_of_bags * macaroon_weight = 
   total_macaroons * macaroon_weight - remaining_weight) →
  number_of_bags = 4 := by
sorry

end macaroon_packing_l3788_378895


namespace parallel_segment_length_l3788_378822

theorem parallel_segment_length (a b c d : ℝ) (h1 : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h2 : a = 300) (h3 : b = 320) (h4 : c = 400) :
  let s := (a + b + c) / 2
  let area_abc := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let area_dpd := area_abc / 4
  ∃ d : ℝ, d > 0 ∧ d^2 / a^2 = area_dpd / area_abc ∧ d = 150 := by
  sorry

end parallel_segment_length_l3788_378822


namespace johns_family_ages_l3788_378828

/-- Given information about John's family ages, prove John's and his sibling's ages -/
theorem johns_family_ages :
  ∀ (john_age dad_age sibling_age : ℕ),
  john_age + 30 = dad_age →
  john_age + dad_age = 90 →
  sibling_age = john_age + 5 →
  john_age = 30 ∧ sibling_age = 35 := by
  sorry

end johns_family_ages_l3788_378828


namespace trululu_nonexistence_l3788_378824

structure Individual where
  statement : Prop

def is_weekday (day : Nat) : Prop :=
  1 ≤ day ∧ day ≤ 5

def Barmaglot_lies (day : Nat) : Prop :=
  1 ≤ day ∧ day ≤ 3

theorem trululu_nonexistence (day : Nat) 
  (h1 : is_weekday day)
  (h2 : ∃ (i1 i2 : Individual), i1.statement = (∃ Trululu : Type, Nonempty Trululu) ∧ i2.statement = True)
  (h3 : ∀ (i : Individual), i.statement = True → i.statement)
  (h4 : Barmaglot_lies day → ¬(∃ Trululu : Type, Nonempty Trululu))
  (h5 : ¬(Barmaglot_lies day))
  : ¬(∃ Trululu : Type, Nonempty Trululu) := by
  sorry

#check trululu_nonexistence

end trululu_nonexistence_l3788_378824


namespace stream_current_rate_l3788_378856

/-- Represents the problem of finding the stream's current rate given rowing conditions. -/
theorem stream_current_rate
  (distance : ℝ)
  (normal_time_diff : ℝ)
  (triple_speed_time_diff : ℝ)
  (h1 : distance = 18)
  (h2 : normal_time_diff = 4)
  (h3 : triple_speed_time_diff = 2)
  (h4 : ∀ (r w : ℝ),
    (distance / (r + w) + normal_time_diff = distance / (r - w)) →
    (distance / (3 * r + w) + triple_speed_time_diff = distance / (3 * r - w)) →
    w = 9 / 8) :
  ∃ (w : ℝ), w = 9 / 8 ∧ 
    (∃ (r : ℝ), 
      (distance / (r + w) + normal_time_diff = distance / (r - w)) ∧
      (distance / (3 * r + w) + triple_speed_time_diff = distance / (3 * r - w))) :=
by
  sorry

end stream_current_rate_l3788_378856


namespace edward_money_proof_l3788_378835

/-- The amount of money Edward had before spending, given his expenses and remaining money. -/
def edward_initial_money (books_cost pens_cost remaining : ℕ) : ℕ :=
  books_cost + pens_cost + remaining

/-- Theorem stating that Edward's initial money was $41 given the problem conditions. -/
theorem edward_money_proof :
  edward_initial_money 6 16 19 = 41 := by
  sorry

end edward_money_proof_l3788_378835


namespace plane_speed_with_wind_l3788_378873

theorem plane_speed_with_wind (distance : ℝ) (wind_speed : ℝ) (time_with_wind : ℝ) (time_against_wind : ℝ) :
  wind_speed = 24 ∧ time_with_wind = 5.5 ∧ time_against_wind = 6 →
  ∃ (plane_speed : ℝ),
    distance / time_with_wind = plane_speed + wind_speed ∧
    distance / time_against_wind = plane_speed - wind_speed ∧
    plane_speed + wind_speed = 576 ∧
    plane_speed - wind_speed = 528 := by
  sorry

end plane_speed_with_wind_l3788_378873


namespace unique_products_count_l3788_378870

def set_a : Finset ℕ := {2, 3, 5, 7, 11}
def set_b : Finset ℕ := {2, 4, 6, 19}

theorem unique_products_count : 
  Finset.card ((set_a.product set_b).image (λ (x : ℕ × ℕ) => x.1 * x.2)) = 19 := by
  sorry

end unique_products_count_l3788_378870


namespace catherine_friends_count_l3788_378869

def total_bottle_caps : ℕ := 18
def caps_per_friend : ℕ := 3

theorem catherine_friends_count : 
  total_bottle_caps / caps_per_friend = 6 := by sorry

end catherine_friends_count_l3788_378869


namespace point_sum_coordinates_l3788_378849

/-- Given that (3, 8) is on the graph of y = g(x), prove that the sum of the coordinates
    of the point on the graph of 5y = 4g(2x) + 6 is 9.1 -/
theorem point_sum_coordinates (g : ℝ → ℝ) (h : g 3 = 8) :
  ∃ x y : ℝ, 5 * y = 4 * g (2 * x) + 6 ∧ x + y = 9.1 := by
  sorry

end point_sum_coordinates_l3788_378849


namespace sin_585_degrees_l3788_378862

theorem sin_585_degrees : Real.sin (585 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end sin_585_degrees_l3788_378862


namespace perfect_square_trinomial_l3788_378817

theorem perfect_square_trinomial (a b : ℝ) : 
  (b - a = -7) → 
  (∃ k : ℝ, ∀ x : ℝ, 16 * x^2 + 144 * x + (a + b) = (k * x + (a + b) / (2 * k))^2) ↔ 
  (a = 165.5 ∧ b = 158.5) := by
sorry

end perfect_square_trinomial_l3788_378817


namespace max_distinct_permutations_eight_points_l3788_378836

-- Define a type for points in a plane
def Point : Type := ℝ × ℝ

-- Define a type for directed lines in a plane
def DirectedLine : Type := ℝ × ℝ × ℝ  -- ax + by + c = 0, with (a,b) ≠ (0,0)

-- Define a function to project a point onto a directed line
def project (p : Point) (l : DirectedLine) : ℝ := sorry

-- Define a function to get the permutation from projections
def getPermutation (points : List Point) (l : DirectedLine) : List ℕ := sorry

-- Define a function to count distinct permutations
def countDistinctPermutations (points : List Point) : ℕ := sorry

theorem max_distinct_permutations_eight_points :
  ∀ (points : List Point),
    points.length = 8 →
    points.Nodup →
    (∀ (l : DirectedLine), (getPermutation points l).Nodup) →
    countDistinctPermutations points ≤ 56 ∧
    ∃ (points' : List Point),
      points'.length = 8 ∧
      points'.Nodup ∧
      (∀ (l : DirectedLine), (getPermutation points' l).Nodup) ∧
      countDistinctPermutations points' = 56 := by
  sorry

end max_distinct_permutations_eight_points_l3788_378836


namespace daily_average_is_40_l3788_378806

/-- Represents the daily average of borrowed books -/
def daily_average : ℝ := 40

/-- Represents the total number of books borrowed in a week -/
def total_weekly_books : ℕ := 216

/-- Represents the borrowing rate on Friday as a multiplier of the daily average -/
def friday_rate : ℝ := 1.4

/-- Theorem stating that given the conditions, the daily average of borrowed books is 40 -/
theorem daily_average_is_40 :
  daily_average * 4 + daily_average * friday_rate = total_weekly_books :=
by sorry

end daily_average_is_40_l3788_378806


namespace log_equation_roots_range_l3788_378841

-- Define the logarithmic equation
def log_equation (x a : ℝ) : Prop :=
  Real.log (x - 1) + Real.log (3 - x) = Real.log (a - x)

-- Define the condition for two distinct real roots
def has_two_distinct_roots (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 1 < x₁ ∧ x₁ < 3 ∧ 1 < x₂ ∧ x₂ < 3 ∧
  log_equation x₁ a ∧ log_equation x₂ a

-- Theorem statement
theorem log_equation_roots_range :
  ∀ a : ℝ, has_two_distinct_roots a ↔ 3 < a ∧ a < 13/4 :=
by sorry

end log_equation_roots_range_l3788_378841


namespace combined_boys_avg_is_24_58_l3788_378858

/-- Represents a high school with average test scores -/
structure School where
  boys_avg : ℝ
  girls_avg : ℝ
  total_avg : ℝ

/-- Calculates the combined average score for boys given two schools and the combined girls' average -/
def combined_boys_avg (lincoln : School) (madison : School) (combined_girls_avg : ℝ) : ℝ :=
  sorry

/-- Theorem stating that the combined boys' average is approximately 24.58 -/
theorem combined_boys_avg_is_24_58 
  (lincoln : School)
  (madison : School)
  (combined_girls_avg : ℝ)
  (h1 : lincoln.boys_avg = 65)
  (h2 : lincoln.girls_avg = 70)
  (h3 : lincoln.total_avg = 68)
  (h4 : madison.boys_avg = 75)
  (h5 : madison.girls_avg = 85)
  (h6 : madison.total_avg = 78)
  (h7 : combined_girls_avg = 80) :
  ∃ ε > 0, |combined_boys_avg lincoln madison combined_girls_avg - 24.58| < ε :=
by sorry

end combined_boys_avg_is_24_58_l3788_378858


namespace condition_necessity_sufficiency_l3788_378840

theorem condition_necessity_sufficiency : 
  (∀ x : ℝ, (x + 1) * (x^2 + 2) > 0 → (x + 1) * (x + 2) > 0) ∧ 
  (∃ x : ℝ, (x + 1) * (x + 2) > 0 ∧ (x + 1) * (x^2 + 2) ≤ 0) := by
  sorry

end condition_necessity_sufficiency_l3788_378840


namespace hamburger_cost_calculation_l3788_378837

/-- Represents the cost calculation for hamburgers with higher quality meat -/
theorem hamburger_cost_calculation 
  (original_meat_pounds : ℝ) 
  (original_cost_per_pound : ℝ) 
  (original_hamburger_count : ℝ) 
  (new_hamburger_count : ℝ) 
  (cost_increase_percentage : ℝ) :
  original_meat_pounds = 5 →
  original_cost_per_pound = 4 →
  original_hamburger_count = 10 →
  new_hamburger_count = 30 →
  cost_increase_percentage = 0.25 →
  (original_meat_pounds / original_hamburger_count) * new_hamburger_count * 
  (original_cost_per_pound * (1 + cost_increase_percentage)) = 75 := by
  sorry

end hamburger_cost_calculation_l3788_378837


namespace quarters_fraction_is_three_fifths_l3788_378805

/-- The total number of quarters Ella has -/
def total_quarters : ℕ := 30

/-- The number of quarters representing states that joined between 1790 and 1809 -/
def quarters_1790_1809 : ℕ := 18

/-- The fraction of quarters representing states that joined between 1790 and 1809 -/
def fraction_1790_1809 : ℚ := quarters_1790_1809 / total_quarters

theorem quarters_fraction_is_three_fifths : 
  fraction_1790_1809 = 3 / 5 := by sorry

end quarters_fraction_is_three_fifths_l3788_378805


namespace cone_lateral_surface_area_l3788_378848

/-- Given a cone with base radius 6 and volume 30π, its lateral surface area is 39π -/
theorem cone_lateral_surface_area (r h l : ℝ) : 
  r = 6 → 
  (1/3) * π * r^2 * h = 30*π → 
  l^2 = r^2 + h^2 → 
  π * r * l = 39*π := by
  sorry

end cone_lateral_surface_area_l3788_378848


namespace parallelogram_perimeter_18_12_l3788_378816

/-- Perimeter of a parallelogram -/
def parallelogram_perimeter (side1 : ℝ) (side2 : ℝ) : ℝ :=
  2 * (side1 + side2)

/-- Theorem: The perimeter of a parallelogram with sides 18 cm and 12 cm is 60 cm -/
theorem parallelogram_perimeter_18_12 :
  parallelogram_perimeter 18 12 = 60 := by
  sorry

end parallelogram_perimeter_18_12_l3788_378816


namespace cos_alpha_value_l3788_378813

theorem cos_alpha_value (α : ℝ) (h : Real.sin (5 * Real.pi / 2 + α) = 1 / 5) : 
  Real.cos α = 1 / 5 := by
  sorry

end cos_alpha_value_l3788_378813


namespace multiple_of_nine_in_range_l3788_378886

theorem multiple_of_nine_in_range (y : ℕ) :
  y > 0 ∧ 
  ∃ k : ℕ, y = 9 * k ∧ 
  y^2 > 225 ∧ 
  y < 30 →
  y = 18 ∨ y = 27 := by
sorry

end multiple_of_nine_in_range_l3788_378886


namespace circle_radius_from_area_l3788_378884

theorem circle_radius_from_area (r : ℝ) : r > 0 → π * r^2 = 9 * π → r = 3 := by
  sorry

end circle_radius_from_area_l3788_378884


namespace roses_count_l3788_378867

theorem roses_count (vase_capacity : ℕ) (carnations : ℕ) (vases : ℕ) : 
  vase_capacity = 9 → carnations = 4 → vases = 3 → 
  vases * vase_capacity - carnations = 23 := by
  sorry

end roses_count_l3788_378867


namespace broken_line_length_bound_l3788_378852

/-- Represents a chessboard -/
structure Chessboard :=
  (size : ℕ)

/-- Represents a broken line on a chessboard -/
structure BrokenLine :=
  (board : Chessboard)
  (isClosed : Bool)
  (noSelfIntersections : Bool)
  (joinsAdjacentCells : Bool)
  (isSymmetricToDiagonal : Bool)

/-- Calculates the length of a broken line -/
def brokenLineLength (line : BrokenLine) : ℝ :=
  sorry

/-- Theorem: The length of a specific broken line on a 15x15 chessboard is at most 200 -/
theorem broken_line_length_bound (line : BrokenLine) :
  line.board.size = 15 →
  line.isClosed = true →
  line.noSelfIntersections = true →
  line.joinsAdjacentCells = true →
  line.isSymmetricToDiagonal = true →
  brokenLineLength line ≤ 200 :=
by sorry

end broken_line_length_bound_l3788_378852


namespace quadratic_inequality_solution_l3788_378839

theorem quadratic_inequality_solution (x : ℝ) :
  -3 * x^2 + 8 * x + 3 > 0 ↔ x < -1/3 ∨ x > 3 :=
by sorry

end quadratic_inequality_solution_l3788_378839


namespace bond_interest_rate_l3788_378803

/-- Represents the annual interest rate as a real number between 0 and 1 -/
def annual_interest_rate : ℝ := 0.04

/-- The initial investment amount in yuan -/
def initial_investment : ℝ := 1000

/-- The amount spent after the first maturity in yuan -/
def spent_amount : ℝ := 440

/-- The final amount received after the second maturity in yuan -/
def final_amount : ℝ := 624

/-- Theorem stating that the annual interest rate is 4% given the problem conditions -/
theorem bond_interest_rate :
  (initial_investment * (1 + annual_interest_rate) - spent_amount) * (1 + annual_interest_rate) = final_amount :=
by sorry

end bond_interest_rate_l3788_378803


namespace volume_of_specific_tetrahedron_l3788_378885

/-- The volume of a tetrahedron given its edge lengths -/
def tetrahedron_volume (pq pr ps qr qs rs : ℝ) : ℝ := sorry

/-- Theorem: The volume of tetrahedron PQRS with given edge lengths is 140/9 -/
theorem volume_of_specific_tetrahedron :
  let pq : ℝ := 6
  let pr : ℝ := 5
  let ps : ℝ := 4 * Real.sqrt 2
  let qr : ℝ := 3 * Real.sqrt 2
  let qs : ℝ := 5
  let rs : ℝ := 4
  tetrahedron_volume pq pr ps qr qs rs = 140 / 9 := by sorry

end volume_of_specific_tetrahedron_l3788_378885


namespace quadratic_coefficient_is_one_l3788_378808

/-- The quadratic equation x^2 - 2x + 1 = 0 -/
def quadratic_equation (x : ℝ) : Prop := x^2 - 2*x + 1 = 0

/-- The coefficient of the quadratic term in the equation x^2 - 2x + 1 = 0 -/
def quadratic_coefficient : ℝ := 1

theorem quadratic_coefficient_is_one : 
  quadratic_coefficient = 1 := by sorry

end quadratic_coefficient_is_one_l3788_378808


namespace smallest_number_l3788_378874

theorem smallest_number (A B C : ℤ) : 
  A = 18 + 38 →
  B = A - 26 →
  C = B / 3 →
  C < A ∧ C < B :=
by sorry

end smallest_number_l3788_378874


namespace restaurant_menu_combinations_l3788_378880

theorem restaurant_menu_combinations (menu_size : ℕ) (yann_order camille_order : ℕ) : 
  menu_size = 12 →
  yann_order ≠ camille_order →
  yann_order ≤ menu_size ∧ camille_order ≤ menu_size →
  (menu_size * (menu_size - 1) : ℕ) = 132 :=
by sorry

end restaurant_menu_combinations_l3788_378880


namespace log_always_defined_range_log_sometimes_undefined_range_l3788_378802

-- Define the function f(m, x)
def f (m x : ℝ) : ℝ := m * x^2 - 4 * m * x + m + 3

-- Theorem 1: Range of m for which the logarithm is always defined
theorem log_always_defined_range (m : ℝ) :
  (∀ x : ℝ, f m x > 0) ↔ m ∈ Set.Ici 0 ∩ Set.Iio 1 :=
sorry

-- Theorem 2: Range of m for which the logarithm is undefined for some x
theorem log_sometimes_undefined_range (m : ℝ) :
  (∃ x : ℝ, f m x ≤ 0) ↔ m ∈ Set.Iio 0 ∪ Set.Ici 1 :=
sorry

end log_always_defined_range_log_sometimes_undefined_range_l3788_378802


namespace gcd_lcm_equality_implies_equal_l3788_378879

theorem gcd_lcm_equality_implies_equal (a b c : ℕ+) :
  (Nat.gcd a b + Nat.lcm a b = Nat.gcd a c + Nat.lcm a c) → b = c := by
  sorry

end gcd_lcm_equality_implies_equal_l3788_378879


namespace soccer_league_female_fraction_l3788_378859

theorem soccer_league_female_fraction :
  -- Last year's male participants
  ∀ (male_last_year : ℕ),
  male_last_year = 30 →
  -- Male increase rate
  ∀ (male_increase_rate : ℝ),
  male_increase_rate = 0.1 →
  -- Female increase rate
  ∀ (female_increase_rate : ℝ),
  female_increase_rate = 0.25 →
  -- Overall increase rate
  ∀ (total_increase_rate : ℝ),
  total_increase_rate = 0.1 →
  -- This year's female participants fraction
  ∃ (female_fraction : ℚ),
  female_fraction = 50 / 83 :=
by sorry

end soccer_league_female_fraction_l3788_378859


namespace equation_solution_l3788_378860

theorem equation_solution : ∃! x : ℝ, (x - 12) / 3 = (3 * x + 9) / 8 ∧ x = -123 := by sorry

end equation_solution_l3788_378860


namespace shelter_dogs_l3788_378868

theorem shelter_dogs (cat_count : ℕ) (cat_ratio : ℕ) (dog_ratio : ℕ) : 
  cat_count = 21 → cat_ratio = 7 → dog_ratio = 5 → 
  (cat_count * dog_ratio) / cat_ratio = 15 :=
by sorry

end shelter_dogs_l3788_378868


namespace third_factor_proof_l3788_378845

theorem third_factor_proof (w : ℕ) (h1 : w = 168) (h2 : 2^5 ∣ (936 * w)) (h3 : 3^3 ∣ (936 * w)) :
  (936 * w) / (2^5 * 3^3) = 182 := by
  sorry

end third_factor_proof_l3788_378845


namespace intersection_and_union_when_a_is_2_subset_condition_l3788_378838

-- Define sets M and N
def M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def N (a : ℝ) : Set ℝ := {x | a + 1 ≤ x ∧ x ≤ 2*a - 1}

-- Theorem for part (1)
theorem intersection_and_union_when_a_is_2 :
  (M ∩ N 2 = {3}) ∧ (M ∪ N 2 = M) := by sorry

-- Theorem for part (2)
theorem subset_condition :
  ∀ a : ℝ, (M ⊇ N a) ↔ a ≤ 3 := by sorry

end intersection_and_union_when_a_is_2_subset_condition_l3788_378838


namespace sum_of_squared_coefficients_l3788_378864

def expression (x : ℝ) : ℝ := 5 * (x^2 - 3*x + 4) - 9 * (x^3 - 2*x^2 + x - 1)

theorem sum_of_squared_coefficients :
  ∃ (a b c d : ℝ),
    (∀ x, expression x = a*x^3 + b*x^2 + c*x + d) ∧
    a^2 + b^2 + c^2 + d^2 = 2027 := by sorry

end sum_of_squared_coefficients_l3788_378864


namespace more_cylindrical_sandcastles_l3788_378829

/-- Represents the sandbox and sandcastle properties -/
structure Sandbox :=
  (base_area : ℝ)
  (sand_height : ℝ)
  (bucket_height : ℝ)
  (cylinder_base_area : ℝ)
  (m : ℕ)  -- number of cylindrical sandcastles
  (n : ℕ)  -- number of conical sandcastles

/-- Theorem stating that Masha's cylindrical sandcastles are more numerous -/
theorem more_cylindrical_sandcastles (sb : Sandbox) 
  (h1 : sb.sand_height = 1)
  (h2 : sb.bucket_height = 2)
  (h3 : sb.base_area = sb.cylinder_base_area * (sb.m + sb.n))
  (h4 : sb.base_area * sb.sand_height = 
        sb.cylinder_base_area * sb.bucket_height * sb.m + 
        (1/3) * sb.cylinder_base_area * sb.bucket_height * sb.n) :
  sb.m > sb.n := by
  sorry

end more_cylindrical_sandcastles_l3788_378829


namespace gcd_digits_bound_l3788_378853

theorem gcd_digits_bound (a b : ℕ) : 
  1000000 ≤ a ∧ a < 10000000 ∧ 
  1000000 ≤ b ∧ b < 10000000 ∧ 
  10000000000 ≤ Nat.lcm a b ∧ Nat.lcm a b < 100000000000 →
  Nat.gcd a b < 10000 := by
sorry

end gcd_digits_bound_l3788_378853


namespace fraction_power_product_l3788_378846

theorem fraction_power_product : (8 / 9 : ℚ)^3 * (1 / 3 : ℚ)^3 = 512 / 19683 := by
  sorry

end fraction_power_product_l3788_378846


namespace line_l_equation_l3788_378878

-- Define the ellipse E
def ellipse (t : ℝ) (x y : ℝ) : Prop := x^2 / t + y^2 = 1

-- Define the parabola C
def parabola (x y : ℝ) : Prop := x^2 = 2 * Real.sqrt 2 * y

-- Define the point H
def H : ℝ × ℝ := (2, 0)

-- Define the condition for line l
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k * (x - 2)

-- Define the condition for tangent lines being perpendicular
def perpendicular_tangents (x₁ x₂ : ℝ) : Prop := 
  (Real.sqrt 2 / 2 * x₁) * (Real.sqrt 2 / 2 * x₂) = -1

theorem line_l_equation :
  ∃ (t k : ℝ) (A B M N : ℝ × ℝ),
    -- Conditions
    (ellipse t A.1 A.2) ∧
    (ellipse t B.1 B.2) ∧
    (parabola A.1 A.2) ∧
    (parabola B.1 B.2) ∧
    (parabola M.1 M.2) ∧
    (parabola N.1 N.2) ∧
    (line_l k M.1 M.2) ∧
    (line_l k N.1 N.2) ∧
    (perpendicular_tangents M.1 N.1) →
    -- Conclusion
    k = -Real.sqrt 2 / 4 ∧ 
    ∀ (x y : ℝ), line_l k x y ↔ x + 2 * Real.sqrt 2 * y - 2 = 0 :=
sorry

end line_l_equation_l3788_378878


namespace infinite_primes_quadratic_equation_l3788_378897

theorem infinite_primes_quadratic_equation :
  ∀ (S : Finset Nat), ∃ (p : Nat), Prime p ∧ p ∉ S ∧ ∃ (x y : ℤ), x^2 + x + 1 = p * y := by
  sorry

end infinite_primes_quadratic_equation_l3788_378897


namespace quadratic_polynomial_from_sum_and_product_l3788_378892

theorem quadratic_polynomial_from_sum_and_product (x y : ℝ) 
  (sum_condition : x + y = 15) 
  (product_condition : x * y = 36) : 
  (fun z : ℝ => z^2 - 15*z + 36) = (fun z : ℝ => (z - x) * (z - y)) := by
  sorry

end quadratic_polynomial_from_sum_and_product_l3788_378892


namespace amanda_keeps_121_candy_bars_l3788_378821

/-- The number of candy bars Amanda keeps for herself after four days of transactions --/
def amanda_candy_bars : ℕ :=
  let initial := 7
  let day1_remaining := initial - (initial / 3)
  let day2_total := day1_remaining + 30
  let day2_remaining := day2_total - (day2_total / 4)
  let day3_gift := day2_remaining * 3
  let day3_remaining := day2_remaining + (day3_gift / 2)
  let day4_bought := 20
  let day4_remaining := day3_remaining + (day4_bought / 3)
  day4_remaining

/-- Theorem stating that Amanda keeps 121 candy bars for herself --/
theorem amanda_keeps_121_candy_bars : amanda_candy_bars = 121 := by
  sorry

end amanda_keeps_121_candy_bars_l3788_378821


namespace max_value_expression_l3788_378804

theorem max_value_expression (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (sum_eq : a + b + c = 3) :
  a + Real.sqrt (a * b) + (a * b * c) ^ (1/4) ≤ 10/3 ∧
  ∃ (a' b' c' : ℝ), 0 ≤ a' ∧ 0 ≤ b' ∧ 0 ≤ c' ∧ a' + b' + c' = 3 ∧
    a' + Real.sqrt (a' * b') + (a' * b' * c') ^ (1/4) = 10/3 :=
by sorry

end max_value_expression_l3788_378804


namespace factorization_proof_l3788_378872

theorem factorization_proof (a b : ℝ) : 4 * a^3 * b - a * b = a * b * (2 * a + 1) * (2 * a - 1) := by
  sorry

end factorization_proof_l3788_378872


namespace multiplicative_inverse_modulo_l3788_378877

theorem multiplicative_inverse_modulo (A' B' : Nat) (m : Nat) (h : m = 2000000) :
  A' = 222222 →
  B' = 285714 →
  (1500000 * (A' * B')) % m = 1 :=
by sorry

end multiplicative_inverse_modulo_l3788_378877


namespace equation_describes_cone_l3788_378882

/-- Spherical coordinates -/
structure SphericalCoord where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

/-- Definition of a cone in spherical coordinates -/
def IsCone (c : ℝ) (f : SphericalCoord → Prop) : Prop :=
  c > 0 ∧ ∀ p : SphericalCoord, f p ↔ p.ρ = c * Real.sin p.φ

/-- The main theorem: the equation ρ = c sin(φ) describes a cone -/
theorem equation_describes_cone (c : ℝ) :
  IsCone c (fun p => p.ρ = c * Real.sin p.φ) := by
  sorry


end equation_describes_cone_l3788_378882


namespace negative_six_div_three_l3788_378831

theorem negative_six_div_three : (-6) / 3 = -2 := by
  sorry

end negative_six_div_three_l3788_378831


namespace mary_final_weight_l3788_378800

def weight_change (initial_weight : ℕ) : ℕ :=
  let first_loss := 12
  let second_gain := 2 * first_loss
  let third_loss := 3 * first_loss
  let final_gain := 6
  initial_weight - first_loss + second_gain - third_loss + final_gain

theorem mary_final_weight (initial_weight : ℕ) (h : initial_weight = 99) :
  weight_change initial_weight = 81 := by
  sorry

end mary_final_weight_l3788_378800


namespace lesser_solution_quadratic_l3788_378855

theorem lesser_solution_quadratic (x : ℝ) : 
  x^2 + 10*x - 75 = 0 → (∃ y : ℝ, y^2 + 10*y - 75 = 0 ∧ y ≤ x) → x = -15 :=
sorry

end lesser_solution_quadratic_l3788_378855


namespace parallel_planes_normal_vectors_l3788_378899

/-- Given two vectors that are normal vectors of parallel planes, prove that their specific components multiply to -3 -/
theorem parallel_planes_normal_vectors (m n : ℝ) : 
  let a : Fin 3 → ℝ := ![0, 1, m]
  let b : Fin 3 → ℝ := ![0, n, -3]
  (∃ (k : ℝ), a = k • b) →  -- Parallel planes condition
  m * n = -3 := by
  sorry

end parallel_planes_normal_vectors_l3788_378899


namespace negation_of_universal_proposition_l3788_378898

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 ≠ x) ↔ (∃ x : ℝ, x^2 = x) := by
  sorry

end negation_of_universal_proposition_l3788_378898


namespace midpoint_x_coordinate_constant_l3788_378863

/-- Parabola structure -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  focus : ℝ × ℝ

/-- Line structure -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- Defines a point on a parabola -/
def PointOnParabola (p : Parabola) (x y : ℝ) : Prop :=
  p.equation x y

/-- Defines the perpendicular bisector of a line segment -/
def PerpendicularBisector (A B M : ℝ × ℝ) : Prop :=
  -- Definition of perpendicular bisector passing through M
  sorry

/-- Main theorem -/
theorem midpoint_x_coordinate_constant
  (p : Parabola)
  (A B : ℝ × ℝ)
  (hA : PointOnParabola p A.1 A.2)
  (hB : PointOnParabola p B.1 B.2)
  (hAB : A.2 - B.2 ≠ 0)  -- AB not perpendicular to x-axis
  (hM : PerpendicularBisector A B (4, 0)) :
  (A.1 + B.1) / 2 = 2 :=
sorry

/-- Setup for the specific problem -/
def problem_setup : Parabola :=
  { equation := fun x y => y^2 = 4*x
  , focus := (1, 0) }

#check midpoint_x_coordinate_constant problem_setup

end midpoint_x_coordinate_constant_l3788_378863


namespace exam_students_count_l3788_378854

theorem exam_students_count :
  ∀ (N : ℕ) (T : ℝ),
  T = N * 80 →
  (T - 100) / (N - 5 : ℝ) = 95 →
  N = 25 :=
by
  sorry

end exam_students_count_l3788_378854


namespace trig_simplification_l3788_378850

theorem trig_simplification :
  (Real.sin (30 * π / 180) + Real.sin (60 * π / 180)) /
  (Real.cos (30 * π / 180) + Real.cos (60 * π / 180)) =
  Real.tan (45 * π / 180) := by
  sorry

end trig_simplification_l3788_378850


namespace andreas_erasers_l3788_378826

theorem andreas_erasers (andrea_erasers : ℕ) : 
  (4 * andrea_erasers = andrea_erasers + 12) → andrea_erasers = 4 := by
  sorry

end andreas_erasers_l3788_378826


namespace only_third_proposition_true_l3788_378883

theorem only_third_proposition_true :
  ∃ (a b c d : ℝ),
    (∃ c, a > b ∧ c ≠ 0 ∧ ¬(a * c > b * c)) ∧
    (∃ c, a > b ∧ ¬(a * c^2 > b * c^2)) ∧
    (∀ a b c : ℝ, a * c^2 > b * c^2 → a > b) ∧
    (∃ a b, a > b ∧ ¬(1 / a < 1 / b)) ∧
    (∃ a b c d, a > b ∧ b > 0 ∧ c > d ∧ ¬(a * c > b * d)) :=
by sorry

end only_third_proposition_true_l3788_378883


namespace incircle_tangents_concurrent_l3788_378866

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a line in 2D space -/
structure Line :=
  (a : ℝ) (b : ℝ) (c : ℝ)

/-- Represents a circle in 2D space -/
structure Circle :=
  (center : Point) (radius : ℝ)

/-- Represents a triangle -/
structure Triangle :=
  (a : Point) (b : Point) (c : Point)

/-- Checks if three lines are concurrent -/
def are_concurrent (l1 l2 l3 : Line) : Prop := sorry

/-- Returns the tangent line to a circle at a given point -/
def tangent_line (c : Circle) (p : Point) : Line := sorry

/-- Returns the line passing through two points -/
def line_through_points (p1 p2 : Point) : Line := sorry

/-- Checks if a point lies on a line -/
def point_on_line (p : Point) (l : Line) : Prop := sorry

/-- Checks if a point lies on a circle -/
def point_on_circle (p : Point) (c : Circle) : Prop := sorry

theorem incircle_tangents_concurrent 
  (t : Triangle) 
  (incircle : Circle) 
  (m n k l : Point) :
  point_on_circle m incircle →
  point_on_circle n incircle →
  point_on_circle k incircle →
  point_on_circle l incircle →
  point_on_line m (line_through_points t.a t.b) →
  point_on_line n (line_through_points t.b t.c) →
  point_on_line k (line_through_points t.c t.a) →
  point_on_line l (line_through_points t.a t.c) →
  are_concurrent 
    (line_through_points m n)
    (line_through_points k l)
    (tangent_line incircle t.a) :=
by
  sorry

end incircle_tangents_concurrent_l3788_378866


namespace minimum_value_theorem_l3788_378876

theorem minimum_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 2) :
  2 / x + 1 / y ≥ 9 / 2 ∧ (2 / x + 1 / y = 9 / 2 ↔ x = 2 / 3 ∧ y = 2 / 3) := by
  sorry

end minimum_value_theorem_l3788_378876


namespace inverse_proportion_problem_l3788_378865

/-- Given two real numbers x and y that are inversely proportional,
    prove that if x + y = 20 and x - y = 4, then y = 24 when x = 4. -/
theorem inverse_proportion_problem (x y : ℝ) (h1 : ∃ k : ℝ, x * y = k) 
    (h2 : x + y = 20) (h3 : x - y = 4) : 
    x = 4 → y = 24 := by
  sorry

end inverse_proportion_problem_l3788_378865


namespace equation_solution_l3788_378833

theorem equation_solution : 
  ∀ m n : ℕ, 19 * m + 84 * n = 1984 ↔ (m = 100 ∧ n = 1) ∨ (m = 16 ∧ n = 20) :=
by sorry

end equation_solution_l3788_378833


namespace xy_max_and_x2_4y2_min_l3788_378842

theorem xy_max_and_x2_4y2_min (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_sum : x + 2*y = 3) :
  (∀ a b : ℝ, a > 0 ∧ b > 0 ∧ a + 2*b = 3 → x*y ≥ a*b) ∧
  x*y = 9/8 ∧
  (∀ a b : ℝ, a > 0 ∧ b > 0 ∧ a + 2*b = 3 → x^2 + 4*y^2 ≤ a^2 + 4*b^2) ∧
  x^2 + 4*y^2 = 9/2 :=
sorry

end xy_max_and_x2_4y2_min_l3788_378842


namespace max_distance_between_sets_l3788_378820

theorem max_distance_between_sets : ∃ (a b : ℂ),
  (a^4 - 16 = 0) ∧
  (b^4 - 16*b^3 - 16*b + 256 = 0) ∧
  (∀ (x y : ℂ), (x^4 - 16 = 0) → (y^4 - 16*y^3 - 16*y + 256 = 0) →
    Complex.abs (x - y) ≤ Complex.abs (a - b)) ∧
  Complex.abs (a - b) = 2 * Real.sqrt 65 :=
sorry

end max_distance_between_sets_l3788_378820


namespace simplify_polynomial_l3788_378815

theorem simplify_polynomial (x : ℝ) : 
  2*x*(4*x^3 - 3*x + 1) - 4*(2*x^3 - x^2 + 3*x - 5) = 
  8*x^4 - 8*x^3 - 2*x^2 - 10*x + 20 := by sorry

end simplify_polynomial_l3788_378815


namespace child_ticket_price_l3788_378843

theorem child_ticket_price 
  (total_tickets : ℕ)
  (total_receipts : ℕ)
  (adult_price : ℕ)
  (child_tickets : ℕ)
  (h1 : total_tickets = 130)
  (h2 : total_receipts = 840)
  (h3 : adult_price = 12)
  (h4 : child_tickets = 90) :
  (total_receipts - (total_tickets - child_tickets) * adult_price) / child_tickets = 4 :=
by
  sorry

end child_ticket_price_l3788_378843


namespace chess_games_ratio_l3788_378809

theorem chess_games_ratio (total_games won_games : ℕ) 
  (h1 : total_games = 44)
  (h2 : won_games = 16) :
  let lost_games := total_games - won_games
  Nat.gcd lost_games won_games = 4 ∧ 
  lost_games / 4 = 7 ∧ 
  won_games / 4 = 4 := by
sorry

end chess_games_ratio_l3788_378809


namespace cos_225_degrees_l3788_378810

theorem cos_225_degrees : Real.cos (225 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end cos_225_degrees_l3788_378810


namespace largest_complete_graph_with_arithmetic_progression_edges_l3788_378887

/-- A function that assigns non-negative integers to edges of a complete graph -/
def EdgeAssignment (n : ℕ) := Fin n → Fin n → ℕ

/-- Predicate to check if three numbers form an arithmetic progression -/
def IsArithmeticProgression (a b c : ℕ) : Prop := 2 * b = a + c

/-- Predicate to check if all edges of a triangle form an arithmetic progression -/
def TriangleIsArithmeticProgression (f : EdgeAssignment n) (i j k : Fin n) : Prop :=
  IsArithmeticProgression (f i j) (f i k) (f j k) ∧
  IsArithmeticProgression (f i j) (f j k) (f i k) ∧
  IsArithmeticProgression (f i k) (f j k) (f i j)

/-- Predicate to check if the edge assignment is valid -/
def ValidAssignment (n : ℕ) (f : EdgeAssignment n) : Prop :=
  (∀ i j : Fin n, i ≠ j → f i j = f j i) ∧ 
  (∀ i j k : Fin n, i ≠ j → j ≠ k → i ≠ k → TriangleIsArithmeticProgression f i j k) ∧
  (∀ i j k l : Fin n, i ≠ j → i ≠ k → i ≠ l → j ≠ k → j ≠ l → k ≠ l → 
    f i j ≠ f i k ∧ f i j ≠ f i l ∧ f i j ≠ f j k ∧ f i j ≠ f j l ∧ f i j ≠ f k l ∧
    f i k ≠ f i l ∧ f i k ≠ f j k ∧ f i k ≠ f j l ∧ f i k ≠ f k l ∧
    f i l ≠ f j k ∧ f i l ≠ f j l ∧ f i l ≠ f k l ∧
    f j k ≠ f j l ∧ f j k ≠ f k l ∧
    f j l ≠ f k l)

theorem largest_complete_graph_with_arithmetic_progression_edges :
  (∃ f : EdgeAssignment 4, ValidAssignment 4 f) ∧
  (∀ n : ℕ, n > 4 → ¬∃ f : EdgeAssignment n, ValidAssignment n f) :=
sorry

end largest_complete_graph_with_arithmetic_progression_edges_l3788_378887


namespace bobs_first_lap_time_l3788_378811

/-- Proves that the time for the first lap is 70 seconds given the conditions of Bob's run --/
theorem bobs_first_lap_time (track_length : ℝ) (num_laps : ℕ) (time_second_lap : ℝ) (time_third_lap : ℝ) (average_speed : ℝ) :
  track_length = 400 →
  num_laps = 3 →
  time_second_lap = 85 →
  time_third_lap = 85 →
  average_speed = 5 →
  (track_length * num_laps) / average_speed - (time_second_lap + time_third_lap) = 70 :=
by sorry

end bobs_first_lap_time_l3788_378811


namespace system_solution_condition_l3788_378881

-- Define the system of equations
def equation1 (x y a : ℝ) : Prop := Real.arccos ((4 + y) / 4) = Real.arccos (x - a)
def equation2 (x y b : ℝ) : Prop := x^2 + y^2 - 4*x + 8*y = b

-- Define the condition for no more than one solution
def atMostOneSolution (a : ℝ) : Prop :=
  ∀ b : ℝ, ∃! (x y : ℝ), equation1 x y a ∧ equation2 x y b

-- Theorem statement
theorem system_solution_condition (a : ℝ) :
  atMostOneSolution a ↔ a ≤ -15 ∨ a ≥ 19 := by sorry

end system_solution_condition_l3788_378881


namespace stratified_sampling_third_grade_l3788_378832

/-- Represents the number of students to be sampled from each grade in a stratified sampling -/
structure StratifiedSample where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Calculates the stratified sample given the total sample size and the ratio of students in each grade -/
def calculateStratifiedSample (totalSample : ℕ) (ratio1 : ℕ) (ratio2 : ℕ) (ratio3 : ℕ) : StratifiedSample :=
  let totalRatio := ratio1 + ratio2 + ratio3
  { first := (ratio1 * totalSample) / totalRatio,
    second := (ratio2 * totalSample) / totalRatio,
    third := (ratio3 * totalSample) / totalRatio }

theorem stratified_sampling_third_grade 
  (totalSample : ℕ) (ratio1 ratio2 ratio3 : ℕ) 
  (h1 : totalSample = 50)
  (h2 : ratio1 = 3)
  (h3 : ratio2 = 3)
  (h4 : ratio3 = 4) :
  (calculateStratifiedSample totalSample ratio1 ratio2 ratio3).third = 20 := by
  sorry

end stratified_sampling_third_grade_l3788_378832


namespace toms_gas_expense_l3788_378812

/-- Proves that given the conditions of Tom's lawn mowing business,
    his monthly gas expense is $17. -/
theorem toms_gas_expense (lawns_mowed : ℕ) (price_per_lawn : ℕ) (extra_income : ℕ) (profit : ℕ) 
    (h1 : lawns_mowed = 3)
    (h2 : price_per_lawn = 12)
    (h3 : extra_income = 10)
    (h4 : profit = 29) :
  lawns_mowed * price_per_lawn + extra_income - profit = 17 := by
  sorry

end toms_gas_expense_l3788_378812


namespace bug_visits_24_tiles_l3788_378889

/-- The number of tiles a bug visits when walking diagonally across a rectangular floor -/
def tilesVisited (width length : ℕ) : ℕ :=
  width + length - Nat.gcd width length

/-- The floor dimensions -/
def floorWidth : ℕ := 12
def floorLength : ℕ := 18

/-- Theorem: A bug walking diagonally across the given rectangular floor visits 24 tiles -/
theorem bug_visits_24_tiles :
  tilesVisited floorWidth floorLength = 24 := by
  sorry


end bug_visits_24_tiles_l3788_378889


namespace ways_to_fifth_floor_l3788_378894

/-- Represents a building with a specified number of floors and staircases between each floor. -/
structure Building where
  floors : ℕ
  staircases : ℕ

/-- Calculates the number of different ways to go from the first floor to the top floor. -/
def waysToTopFloor (b : Building) : ℕ :=
  b.staircases ^ (b.floors - 1)

/-- Theorem stating that in a 5-floor building with 2 staircases between each pair of consecutive floors,
    the number of different ways to go from the first floor to the fifth floor is 2^4. -/
theorem ways_to_fifth_floor :
  let b : Building := { floors := 5, staircases := 2 }
  waysToTopFloor b = 2^4 := by
  sorry

end ways_to_fifth_floor_l3788_378894


namespace cost_at_two_l3788_378888

/-- The cost function for a product -/
def cost (q : ℝ) : ℝ := q^3 + q - 1

/-- Theorem: The cost is 9 when the quantity is 2 -/
theorem cost_at_two : cost 2 = 9 := by
  sorry

end cost_at_two_l3788_378888


namespace unique_solution_l3788_378844

def problem (a : ℕ) (x : ℕ) : Prop :=
  a > 0 ∧
  x > 0 ∧
  x < a ∧
  71 * x + 69 * (a - x) = 3480

theorem unique_solution :
  ∃! a x, problem a x ∧ a = 50 ∧ x = 15 :=
sorry

end unique_solution_l3788_378844


namespace banana_cream_pie_degrees_is_44_l3788_378891

/-- The number of degrees in a pie chart slice for banana cream pie preference --/
def banana_cream_pie_degrees (total_students : ℕ) 
                              (strawberry_pref : ℕ) 
                              (pecan_pref : ℕ) 
                              (pumpkin_pref : ℕ) : ℚ :=
  let remaining_students := total_students - (strawberry_pref + pecan_pref + pumpkin_pref)
  let banana_cream_pref := remaining_students / 2
  (banana_cream_pref / total_students) * 360

/-- Theorem stating that the number of degrees for banana cream pie preference is 44 --/
theorem banana_cream_pie_degrees_is_44 : 
  banana_cream_pie_degrees 45 15 10 9 = 44 := by
  sorry

end banana_cream_pie_degrees_is_44_l3788_378891


namespace one_positive_root_l3788_378857

def f (x : ℝ) : ℝ := x^4 + 10*x^3 - 2*x^2 + 12*x - 9

theorem one_positive_root :
  ∃! x : ℝ, x > 0 ∧ x < 1 ∧ f x = 0 :=
sorry

end one_positive_root_l3788_378857


namespace odd_function_sum_zero_l3788_378896

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x ∈ ℝ -/
def OddFunction (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

theorem odd_function_sum_zero (f : ℝ → ℝ) (h : OddFunction f) :
  f (-2012) + f (-2011) + f 0 + f 2011 + f 2012 = 0 := by
  sorry

end odd_function_sum_zero_l3788_378896


namespace min_skilled_players_exists_tournament_with_three_skilled_l3788_378871

/-- Represents a player in the tournament -/
def Player := Fin 2023

/-- Represents the result of a match between two players -/
def MatchResult := Player → Player → Prop

/-- A player is skilled if for every player who defeats them, there exists another player who defeats that player and loses to the skilled player -/
def IsSkilled (result : MatchResult) (p : Player) : Prop :=
  ∀ q, result q p → ∃ r, result p r ∧ result r q

/-- The tournament satisfies the given conditions -/
def ValidTournament (result : MatchResult) : Prop :=
  (∀ p q, p ≠ q → (result p q ∨ result q p)) ∧
  (∀ p, ¬(∀ q, p ≠ q → result p q))

/-- The main theorem: there are at least 3 skilled players in any valid tournament -/
theorem min_skilled_players (result : MatchResult) (h : ValidTournament result) :
  ∃ a b c : Player, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ IsSkilled result a ∧ IsSkilled result b ∧ IsSkilled result c :=
sorry

/-- There exists a valid tournament with exactly 3 skilled players -/
theorem exists_tournament_with_three_skilled :
  ∃ result : MatchResult, ValidTournament result ∧
  (∃ a b c : Player, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    IsSkilled result a ∧ IsSkilled result b ∧ IsSkilled result c ∧
    (∀ p, IsSkilled result p → p = a ∨ p = b ∨ p = c)) :=
sorry

end min_skilled_players_exists_tournament_with_three_skilled_l3788_378871


namespace cos_sin_sum_zero_implies_double_angle_sum_zero_l3788_378818

theorem cos_sin_sum_zero_implies_double_angle_sum_zero 
  (x y z : ℝ) 
  (h1 : Real.cos x + Real.cos y + Real.cos z = 0)
  (h2 : Real.sin x + Real.sin y + Real.sin z = 0) : 
  Real.cos (2*x) + Real.cos (2*y) + Real.cos (2*z) = 0 := by
  sorry

end cos_sin_sum_zero_implies_double_angle_sum_zero_l3788_378818


namespace x_value_l3788_378807

theorem x_value (x y : ℝ) (hx : x ≠ 0) (h1 : x/3 = y^3) (h2 : x/9 = 9*y) :
  x = 243 * Real.sqrt 3 ∨ x = -243 * Real.sqrt 3 := by
  sorry

end x_value_l3788_378807


namespace jane_donuts_problem_l3788_378823

theorem jane_donuts_problem :
  ∀ (d c : ℕ),
  d + c = 6 →
  90 * d + 60 * c = 450 →
  d = 3 :=
by
  sorry

end jane_donuts_problem_l3788_378823
