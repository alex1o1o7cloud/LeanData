import Mathlib

namespace average_of_series_l1665_166598

/-- The average of a series z, 3z, 5z, 9z, and 17z is 7z -/
theorem average_of_series (z : ℝ) : (z + 3*z + 5*z + 9*z + 17*z) / 5 = 7*z := by
  sorry

end average_of_series_l1665_166598


namespace boat_distance_theorem_l1665_166506

/-- The distance traveled by a boat against a water flow -/
def distance_traveled (a : ℝ) : ℝ :=
  3 * (a - 3)

/-- Theorem: The distance traveled by a boat against a water flow in 3 hours
    is 3(a-3) km, given that the boat's speed in still water is a km/h
    and the water flow speed is 3 km/h. -/
theorem boat_distance_theorem (a : ℝ) :
  let boat_speed := a
  let water_flow_speed := (3 : ℝ)
  let travel_time := (3 : ℝ)
  distance_traveled a = travel_time * (boat_speed - water_flow_speed) :=
by
  sorry


end boat_distance_theorem_l1665_166506


namespace mean_proportional_sum_l1665_166558

/-- Mean proportional of two numbers -/
def mean_proportional (a b c : ℝ) : Prop := a / b = b / c

/-- Find x such that 0.9 : 0.6 = 0.6 : x -/
def find_x : ℝ := 0.4

/-- Find y such that 1/2 : 1/5 = 1/5 : y -/
def find_y : ℝ := 0.08

theorem mean_proportional_sum :
  mean_proportional 0.9 0.6 find_x ∧ 
  mean_proportional (1/2) (1/5) find_y ∧
  find_x + find_y = 0.48 := by sorry

end mean_proportional_sum_l1665_166558


namespace min_value_inequality_l1665_166509

theorem min_value_inequality (x y z : ℝ) (h : x + 2*y + 3*z = 1) : 
  x^2 + 2*y^2 + 3*z^2 ≥ 1/3 := by
  sorry

#check min_value_inequality

end min_value_inequality_l1665_166509


namespace transylvanian_logic_l1665_166557

/-- Represents the possible types of beings in Transylvania -/
inductive Being
| Human
| Vampire

/-- Represents the possible responses to questions -/
inductive Response
| Yes
| No

/-- A function that determines how a being responds to a question about another being's type -/
def respond (respondent : Being) (subject : Being) : Response :=
  match respondent, subject with
  | Being.Human, Being.Human => Response.Yes
  | Being.Human, Being.Vampire => Response.No
  | Being.Vampire, Being.Human => Response.No
  | Being.Vampire, Being.Vampire => Response.Yes

theorem transylvanian_logic (A B : Being) 
  (h1 : respond A B = Response.Yes) : 
  respond B A = Response.Yes := by
  sorry

end transylvanian_logic_l1665_166557


namespace f_positive_implies_a_greater_than_half_open_l1665_166538

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2 * x + 2

-- State the theorem
theorem f_positive_implies_a_greater_than_half_open :
  ∀ a : ℝ, (∀ x : ℝ, x ∈ Set.Icc 1 4 → f a x > 0) → a > 1/2 := by
  sorry

end f_positive_implies_a_greater_than_half_open_l1665_166538


namespace negation_of_proposition_l1665_166551

theorem negation_of_proposition (p : Prop) :
  (p ↔ ∃ x, x < 1 ∧ x^2 ≤ 1) →
  (¬p ↔ ∀ x, x < 1 → x^2 > 1) :=
by sorry

end negation_of_proposition_l1665_166551


namespace unique_number_with_divisor_sum_power_of_ten_l1665_166570

theorem unique_number_with_divisor_sum_power_of_ten (N : ℕ) : 
  (∃ m : ℕ, m < N ∧ m ∣ N ∧ (∀ d : ℕ, d < N → d ∣ N → d ≤ m) ∧ 
   (∃ k : ℕ, N + m = 10^k)) → N = 75 := by
sorry

end unique_number_with_divisor_sum_power_of_ten_l1665_166570


namespace room_occupancy_l1665_166518

theorem room_occupancy (total_chairs : ℕ) (total_people : ℕ) : 
  (3 * total_chairs / 4 = total_chairs - 6) →  -- Three-fourths of chairs are occupied
  (2 * total_people / 3 = 3 * total_chairs / 4) →  -- Two-thirds of people are seated
  total_people = 27 := by
sorry

end room_occupancy_l1665_166518


namespace amanda_ticket_sales_l1665_166567

/-- The number of tickets Amanda needs to sell on the third day -/
def tickets_to_sell_on_third_day (total_tickets : ℕ) (friends : ℕ) (tickets_per_friend : ℕ) (second_day_sales : ℕ) : ℕ :=
  total_tickets - (friends * tickets_per_friend + second_day_sales)

/-- Theorem stating the number of tickets Amanda needs to sell on the third day -/
theorem amanda_ticket_sales : tickets_to_sell_on_third_day 80 5 4 32 = 28 := by
  sorry

end amanda_ticket_sales_l1665_166567


namespace polynomial_perfect_square_l1665_166528

/-- The polynomial (x-1)(x+3)(x-4)(x-8)+m is a perfect square if and only if m = 196 -/
theorem polynomial_perfect_square (x m : ℝ) : 
  ∃ y : ℝ, (x - 1) * (x + 3) * (x - 4) * (x - 8) + m = y^2 ↔ m = 196 := by
  sorry

end polynomial_perfect_square_l1665_166528


namespace like_terms_exponent_sum_l1665_166564

/-- Given two terms 3x^m*y and -5x^2*y^n that are like terms, prove that m + n = 3 -/
theorem like_terms_exponent_sum (m n : ℕ) : 
  (∃ (x y : ℝ), 3 * x^m * y = -5 * x^2 * y^n) → m + n = 3 :=
by sorry

end like_terms_exponent_sum_l1665_166564


namespace planting_schemes_count_l1665_166576

def number_of_seeds : ℕ := 6
def number_of_plots : ℕ := 4
def number_of_first_plot_options : ℕ := 2

def planting_schemes : ℕ :=
  number_of_first_plot_options * (number_of_seeds - 1).factorial / (number_of_seeds - number_of_plots).factorial

theorem planting_schemes_count : planting_schemes = 120 := by
  sorry

end planting_schemes_count_l1665_166576


namespace root_equation_solution_l1665_166582

theorem root_equation_solution (a : ℚ) : 
  ((-2 : ℚ)^2 - a * (-2) + 7 = 0) → a = -11/2 := by
sorry

end root_equation_solution_l1665_166582


namespace converse_and_inverse_false_l1665_166543

-- Define what it means for a triangle to be equilateral
def is_equilateral (triangle : Type) : Prop := sorry

-- Define what it means for a triangle to be isosceles
def is_isosceles (triangle : Type) : Prop := sorry

-- The original statement (given as true)
axiom original_statement : ∀ (triangle : Type), is_equilateral triangle → is_isosceles triangle

-- Theorem to prove
theorem converse_and_inverse_false :
  (∃ (triangle : Type), is_isosceles triangle ∧ ¬is_equilateral triangle) ∧
  (∃ (triangle : Type), ¬is_equilateral triangle ∧ is_isosceles triangle) :=
by sorry

end converse_and_inverse_false_l1665_166543


namespace street_length_proof_l1665_166580

/-- Proves that the length of a street is 1440 meters, given that a person crosses it in 12 minutes at a speed of 7.2 km per hour. -/
theorem street_length_proof (time : ℝ) (speed : ℝ) (length : ℝ) : 
  time = 12 →
  speed = 7.2 →
  length = speed * 1000 / 60 * time →
  length = 1440 := by
sorry

end street_length_proof_l1665_166580


namespace star_properties_l1665_166549

def star (x y : ℝ) : ℝ := (x - 1) * (y - 1) - 1

theorem star_properties :
  (∀ x y : ℝ, star x y = star y x) ∧
  (∃ x y z : ℝ, star x (y + z) ≠ star x y + star x z) ∧
  (∀ x : ℝ, star (x + 1) (x - 1) = star x x - 1) ∧
  (∀ e : ℝ, ∃ x : ℝ, star x e ≠ x) ∧
  (∃ x y z : ℝ, star (star x y) z ≠ star x (star y z)) := by
  sorry

end star_properties_l1665_166549


namespace files_left_theorem_l1665_166519

/-- Calculates the number of files left after deletion -/
def files_left (initial_files : ℕ) (deleted_files : ℕ) : ℕ :=
  initial_files - deleted_files

/-- Theorem: The number of files left is the difference between initial files and deleted files -/
theorem files_left_theorem (initial_files deleted_files : ℕ) 
  (h : deleted_files ≤ initial_files) : 
  files_left initial_files deleted_files = initial_files - deleted_files :=
by
  sorry

#eval files_left 21 14  -- Should output 7

end files_left_theorem_l1665_166519


namespace mike_ride_distance_l1665_166585

-- Define the taxi fare structure for each route
structure TaxiFare :=
  (initial_fee : ℚ)
  (per_mile_rate : ℚ)
  (extra_fee : ℚ)

-- Define the routes
def route_a : TaxiFare := ⟨2.5, 0.25, 3⟩
def route_b : TaxiFare := ⟨2.5, 0.3, 4⟩
def route_c : TaxiFare := ⟨2.5, 0.25, 9⟩ -- Combined bridge toll and traffic surcharge

-- Calculate the fare for a given route and distance
def calculate_fare (route : TaxiFare) (miles : ℚ) : ℚ :=
  route.initial_fee + route.per_mile_rate * miles + route.extra_fee

-- Theorem statement
theorem mike_ride_distance :
  let annie_miles : ℚ := 14
  let annie_fare := calculate_fare route_c annie_miles
  ∃ (mike_miles : ℚ), 
    (calculate_fare route_a mike_miles = annie_fare) ∧
    (mike_miles = 38) :=
by
  sorry


end mike_ride_distance_l1665_166585


namespace det_3A_eq_96_l1665_166563

def A : Matrix (Fin 2) (Fin 2) ℝ := !![7, -2; -5, 6]

theorem det_3A_eq_96 : Matrix.det (3 • A) = 96 := by
  sorry

end det_3A_eq_96_l1665_166563


namespace morning_eggs_count_l1665_166579

/-- The number of eggs used in a day at the Wafting Pie Company -/
def total_eggs : ℕ := 1339

/-- The number of eggs used in the afternoon at the Wafting Pie Company -/
def afternoon_eggs : ℕ := 523

/-- The number of eggs used in the morning at the Wafting Pie Company -/
def morning_eggs : ℕ := total_eggs - afternoon_eggs

theorem morning_eggs_count : morning_eggs = 816 := by sorry

end morning_eggs_count_l1665_166579


namespace prime_sum_theorem_l1665_166574

theorem prime_sum_theorem (p q r : ℕ) : 
  Prime p → Prime q → Prime r → p + q = r → p < q → p = 2 := by
  sorry

end prime_sum_theorem_l1665_166574


namespace infinitely_many_non_squares_l1665_166520

theorem infinitely_many_non_squares (a b c : ℕ+) :
  Set.Infinite {n : ℕ+ | ∃ k : ℕ, (n.val : ℤ)^3 + (a.val : ℤ) * (n.val : ℤ)^2 + (b.val : ℤ) * (n.val : ℤ) + (c.val : ℤ) ≠ (k : ℤ)^2} :=
sorry

end infinitely_many_non_squares_l1665_166520


namespace diana_hits_eight_l1665_166514

structure Friend where
  name : String
  score : Nat

def target_scores : List Nat := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

def friends : List Friend := [
  { name := "Alex", score := 18 },
  { name := "Betsy", score := 5 },
  { name := "Carlos", score := 12 },
  { name := "Diana", score := 14 },
  { name := "Edward", score := 19 },
  { name := "Fiona", score := 11 }
]

theorem diana_hits_eight :
  ∀ (assignments : List (List Nat)),
    (∀ f : Friend, f ∈ friends → 
      ∃! pair : List Nat, pair ∈ assignments ∧ pair.length = 2 ∧ pair.sum = f.score) →
    (∀ pair : List Nat, pair ∈ assignments → 
      pair.length = 2 ∧ pair.toFinset ⊆ target_scores.toFinset) →
    (∀ n : Nat, n ∈ target_scores → 
      (assignments.join.count n ≤ 1)) →
    ∃ pair : List Nat, pair ∈ assignments ∧ 
      pair.length = 2 ∧ 
      pair.sum = 14 ∧ 
      8 ∈ pair :=
by sorry

end diana_hits_eight_l1665_166514


namespace multiple_births_l1665_166595

theorem multiple_births (total_babies : ℕ) (twins triplets quintuplets : ℕ) : 
  total_babies = 1200 →
  triplets = 2 * quintuplets →
  twins = 2 * triplets →
  2 * twins + 3 * triplets + 5 * quintuplets = total_babies →
  5 * quintuplets = 316 :=
by
  sorry

end multiple_births_l1665_166595


namespace cylinder_minus_cones_volume_l1665_166536

/-- The volume of a cylinder minus the volume of two congruent cones --/
theorem cylinder_minus_cones_volume 
  (r : ℝ) -- radius of cylinder and cones
  (h_cylinder : ℝ) -- height of cylinder
  (h_cone : ℝ) -- height of each cone
  (h_cylinder_eq : h_cylinder = 2 * h_cone) -- cylinder height is twice the cone height
  (r_eq : r = 10) -- radius is 10 cm
  (h_cone_eq : h_cone = 15) -- cone height is 15 cm
  : π * r^2 * h_cylinder - 2 * (1/3 * π * r^2 * h_cone) = 2000 * π := by
  sorry

end cylinder_minus_cones_volume_l1665_166536


namespace simplify_fraction_l1665_166589

theorem simplify_fraction (b : ℚ) (h : b = 2) : (15 * b^4) / (75 * b^3) = 2 / 5 := by
  sorry

end simplify_fraction_l1665_166589


namespace complex_fraction_calculations_l1665_166512

theorem complex_fraction_calculations :
  (1 / 60) / ((1 / 3) - (1 / 4) + (1 / 12)) = 1 / 10 ∧
  -(1 / 42) / ((3 / 7) - (5 / 14) + (2 / 3) - (1 / 6)) = -(1 / 24) := by
  sorry

end complex_fraction_calculations_l1665_166512


namespace truck_tunnel_height_l1665_166577

theorem truck_tunnel_height (tunnel_radius : ℝ) (truck_width : ℝ) 
  (h_radius : tunnel_radius = 4.5)
  (h_width : truck_width = 2.7) :
  Real.sqrt (tunnel_radius^2 - (truck_width/2)^2) = 3.6 := by
sorry

end truck_tunnel_height_l1665_166577


namespace fraction_subtraction_simplification_l1665_166503

theorem fraction_subtraction_simplification :
  (9 : ℚ) / 19 - 3 / 57 - 1 / 3 = 5 / 57 := by
  sorry

end fraction_subtraction_simplification_l1665_166503


namespace only_one_chooses_course_a_l1665_166524

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- The total number of course selection combinations -/
def total_combinations (n k : ℕ) : ℕ := (choose n k) * (choose n k)

/-- The number of combinations where both people choose course A -/
def both_choose_a (n k : ℕ) : ℕ := (choose (n - 1) (k - 1)) * (choose (n - 1) (k - 1))

/-- The number of ways in which only one person chooses course A -/
def only_one_chooses_a (n k : ℕ) : ℕ := (total_combinations n k) - (both_choose_a n k)

theorem only_one_chooses_course_a :
  only_one_chooses_a 4 2 = 27 := by sorry

end only_one_chooses_course_a_l1665_166524


namespace segment_length_is_twenty_l1665_166537

/-- The volume of a geometric body formed by points whose distance to a line segment
    is no greater than r units -/
noncomputable def geometricBodyVolume (r : ℝ) (segmentLength : ℝ) : ℝ :=
  (4/3) * Real.pi * r^3 + Real.pi * r^2 * segmentLength

/-- Theorem stating that if the volume of the geometric body with radius 3
    is 216π, then the segment length is 20 -/
theorem segment_length_is_twenty (segmentLength : ℝ) :
  geometricBodyVolume 3 segmentLength = 216 * Real.pi → segmentLength = 20 := by
  sorry

#check segment_length_is_twenty

end segment_length_is_twenty_l1665_166537


namespace min_value_sum_reciprocals_l1665_166586

theorem min_value_sum_reciprocals (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 2 * a + b = 2) :
  (1 / a + 2 / b) ≥ 3 + 2 * Real.sqrt 2 := by
  sorry

end min_value_sum_reciprocals_l1665_166586


namespace distribute_seven_balls_two_boxes_l1665_166511

/-- The number of ways to distribute distinguishable balls into distinguishable boxes -/
def distribute_balls (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  num_boxes ^ num_balls

/-- Theorem: The number of ways to distribute 7 distinguishable balls into 2 distinguishable boxes is 128 -/
theorem distribute_seven_balls_two_boxes : 
  distribute_balls 7 2 = 128 := by
  sorry

end distribute_seven_balls_two_boxes_l1665_166511


namespace point_on_line_l1665_166516

/-- A point in the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of a line passing through two points -/
def Line (p1 p2 : Point) :=
  {p : Point | (p.y - p1.y) * (p2.x - p1.x) = (p2.y - p1.y) * (p.x - p1.x)}

/-- The problem statement -/
theorem point_on_line :
  let p1 : Point := ⟨0, 10⟩
  let p2 : Point := ⟨5, 0⟩
  let p3 : Point := ⟨x, -5⟩
  p3 ∈ Line p1 p2 → x = 7.5 := by
  sorry

end point_on_line_l1665_166516


namespace function_cycle_existence_l1665_166562

theorem function_cycle_existence :
  ∃ (f : ℝ → ℝ) (x₁ x₂ x₃ x₄ x₅ : ℝ),
    (∃ (a b c d : ℝ), ∀ x, f x = (a * x + b) / (c * x + d)) ∧
    x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₁ ≠ x₅ ∧
    x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₂ ≠ x₅ ∧
    x₃ ≠ x₄ ∧ x₃ ≠ x₅ ∧
    x₄ ≠ x₅ ∧
    f x₁ = x₂ ∧ f x₂ = x₃ ∧ f x₃ = x₄ ∧ f x₄ = x₅ ∧ f x₅ = x₁ := by
  sorry

end function_cycle_existence_l1665_166562


namespace empty_carton_weight_l1665_166540

/-- Given the weights of a half-full and full milk carton, calculate the weight of an empty carton -/
theorem empty_carton_weight (half_full_weight full_weight : ℝ) :
  half_full_weight = 5 →
  full_weight = 8 →
  full_weight - 2 * (full_weight - half_full_weight) = 2 := by
  sorry

end empty_carton_weight_l1665_166540


namespace classroom_discussion_group_l1665_166547

def group_sizes : List Nat := [2, 3, 5, 6, 7, 8, 11, 12, 13, 17, 20, 22, 24]

theorem classroom_discussion_group (
  total_groups : Nat) 
  (lecture_groups : Nat) 
  (chinese_lecture_ratio : Nat) 
  (h1 : total_groups = 13)
  (h2 : lecture_groups = 12)
  (h3 : chinese_lecture_ratio = 6)
  (h4 : group_sizes.length = total_groups)
  (h5 : group_sizes.sum = 150) :
  ∃ x : Nat, x ∈ group_sizes ∧ x % 7 = 4 := by
  sorry

end classroom_discussion_group_l1665_166547


namespace sqrt_equation_solution_l1665_166593

theorem sqrt_equation_solution :
  ∀ x : ℚ, (x > 2) → (Real.sqrt (8 * x) / Real.sqrt (5 * (x - 2)) = 3) → x = 90 / 37 := by
sorry

end sqrt_equation_solution_l1665_166593


namespace line_passes_through_fixed_point_l1665_166517

-- Define the curve C
def C (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define point A
def A : ℝ × ℝ := (0, 1)

-- Define the condition for a point to be on a line passing through A
def on_line_through_A (k b x y : ℝ) : Prop := y = k * x + b ∧ b ≠ 1

-- Define the perpendicular condition
def perpendicular_condition (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  (x₁ - 0) * (x₂ - 0) + (y₁ - 1) * (y₂ - 1) = 0

-- Main theorem
theorem line_passes_through_fixed_point 
  (k b x₁ y₁ x₂ y₂ : ℝ) :
  C x₁ y₁ → C x₂ y₂ → 
  on_line_through_A k b x₁ y₁ → on_line_through_A k b x₂ y₂ →
  perpendicular_condition x₁ y₁ x₂ y₂ →
  b = -3/5 :=
sorry

end line_passes_through_fixed_point_l1665_166517


namespace elephant_count_l1665_166590

/-- The number of elephants at We Preserve For Future park -/
def W : ℕ := 70

/-- The number of elephants at Gestures For Good park -/
def G : ℕ := 3 * W

/-- The total number of elephants in both parks -/
def total_elephants : ℕ := W + G

theorem elephant_count : total_elephants = 280 := by
  sorry

end elephant_count_l1665_166590


namespace height_weight_relationship_l1665_166573

/-- Represents the coefficient of determination (R²) in a linear regression model -/
def R_squared : ℝ := 0.64

/-- The proportion of variation explained by the model -/
def variation_explained : ℝ := R_squared

/-- The proportion of variation not explained by the model (random error) -/
def variation_unexplained : ℝ := 1 - R_squared

theorem height_weight_relationship :
  variation_explained = 0.64 ∧
  variation_unexplained = 0.36 ∧
  variation_explained + variation_unexplained = 1 := by
  sorry

#eval R_squared
#eval variation_explained
#eval variation_unexplained

end height_weight_relationship_l1665_166573


namespace school_trip_combinations_l1665_166510

/-- The number of different combinations of riding groups and ride choices -/
def ride_combinations (total_people : ℕ) (group_size : ℕ) (ride_choices : ℕ) : ℕ :=
  Nat.choose total_people group_size * ride_choices

/-- Theorem: Given 8 people, rides of 4, and 2 choices, there are 140 combinations -/
theorem school_trip_combinations :
  ride_combinations 8 4 2 = 140 := by
  sorry

end school_trip_combinations_l1665_166510


namespace union_cardinality_l1665_166525

def A : Finset ℕ := {4, 5, 7, 9}
def B : Finset ℕ := {3, 4, 7, 8, 9}

theorem union_cardinality : (A ∪ B).card = 6 := by
  sorry

end union_cardinality_l1665_166525


namespace darryl_honeydew_price_l1665_166548

/-- The price of a honeydew given Darryl's sales data -/
def honeydew_price (cantaloupe_price : ℚ) (initial_cantaloupes : ℕ) (initial_honeydews : ℕ)
  (dropped_cantaloupes : ℕ) (rotten_honeydews : ℕ) (final_cantaloupes : ℕ) (final_honeydews : ℕ)
  (total_revenue : ℚ) : ℚ :=
  let sold_cantaloupes := initial_cantaloupes - final_cantaloupes - dropped_cantaloupes
  let sold_honeydews := initial_honeydews - final_honeydews - rotten_honeydews
  let cantaloupe_revenue := cantaloupe_price * sold_cantaloupes
  let honeydew_revenue := total_revenue - cantaloupe_revenue
  honeydew_revenue / sold_honeydews

theorem darryl_honeydew_price :
  honeydew_price 2 30 27 2 3 8 9 85 = 3 := by
  sorry

#eval honeydew_price 2 30 27 2 3 8 9 85

end darryl_honeydew_price_l1665_166548


namespace quadratic_equation_with_zero_root_l1665_166568

theorem quadratic_equation_with_zero_root (a : ℝ) : 
  (∃ x : ℝ, (a - 1) * x^2 + x + (a - 2) = 0) ∧ 
  ((a - 1) * 0^2 + 0 + (a - 2) = 0) → 
  a = 2 := by
sorry

end quadratic_equation_with_zero_root_l1665_166568


namespace a_most_stable_l1665_166596

/-- Represents a participant in the shooting test -/
inductive Participant
| A
| B
| C
| D

/-- Returns the variance of a participant's scores -/
def variance (p : Participant) : ℝ :=
  match p with
  | Participant.A => 0.54
  | Participant.B => 0.61
  | Participant.C => 0.7
  | Participant.D => 0.63

/-- Determines if a participant has the most stable performance -/
def has_most_stable_performance (p : Participant) : Prop :=
  ∀ q : Participant, variance p ≤ variance q

/-- Theorem: A has the most stable shooting performance -/
theorem a_most_stable : has_most_stable_performance Participant.A := by
  sorry

end a_most_stable_l1665_166596


namespace total_votes_cast_l1665_166504

theorem total_votes_cast (total_votes : ℕ) (votes_for : ℕ) (votes_against : ℕ) : 
  votes_for = votes_against + 70 →
  votes_against = (40 : ℕ) * total_votes / 100 →
  total_votes = votes_for + votes_against →
  total_votes = 350 := by
sorry

end total_votes_cast_l1665_166504


namespace floor_minus_x_is_zero_l1665_166561

theorem floor_minus_x_is_zero (x : ℝ) (h : ⌈x⌉ - ⌊x⌋ = 0) : ⌊x⌋ - x = 0 := by
  sorry

end floor_minus_x_is_zero_l1665_166561


namespace chord_length_concentric_circles_l1665_166571

/-- Given two concentric circles with radii R and r, where the area of the ring between them is 20π,
    the length of a chord of the larger circle that is tangent to the smaller circle is 4√5. -/
theorem chord_length_concentric_circles (R r : ℝ) (h : R > r) :
  (π * R^2 - π * r^2 = 20 * π) →
  ∃ (c : ℝ), c^2 = 80 ∧ c = 4 * Real.sqrt 5 := by
  sorry

end chord_length_concentric_circles_l1665_166571


namespace rationalize_denominator_l1665_166515

theorem rationalize_denominator : 
  Real.sqrt (5 / 18) = Real.sqrt 10 / 6 := by
  sorry

end rationalize_denominator_l1665_166515


namespace lcm_theorem_l1665_166572

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

def lcm_condition (ab cd : ℕ) : Prop :=
  is_two_digit ab ∧ is_two_digit cd ∧
  Nat.lcm ab cd = (7 * Nat.lcm (reverse_digits ab) (reverse_digits cd)) / 4

theorem lcm_theorem (ab cd : ℕ) (h : lcm_condition ab cd) :
  Nat.lcm ab cd = 252 := by
  sorry

end lcm_theorem_l1665_166572


namespace regular_polygon_with_150_degree_interior_angle_has_12_sides_l1665_166569

/-- A regular polygon with an interior angle of 150° has 12 sides -/
theorem regular_polygon_with_150_degree_interior_angle_has_12_sides :
  ∀ (n : ℕ) (interior_angle : ℝ),
    n ≥ 3 →
    interior_angle = 150 →
    interior_angle = (n - 2) * 180 / n →
    n = 12 := by
  sorry

end regular_polygon_with_150_degree_interior_angle_has_12_sides_l1665_166569


namespace rational_power_equality_l1665_166583

theorem rational_power_equality (x y : ℚ) (n : ℕ) (h_odd : Odd n) (h_pos : 0 < n)
  (h_eq : x^n - 2*x = y^n - 2*y) : x = y := by
  sorry

end rational_power_equality_l1665_166583


namespace gcd_of_225_and_135_l1665_166533

theorem gcd_of_225_and_135 : Nat.gcd 225 135 = 15 := by
  sorry

end gcd_of_225_and_135_l1665_166533


namespace complex_number_equality_l1665_166523

theorem complex_number_equality : ∀ (i : ℂ), i * i = -1 → (2 - i) * i = -1 + 2 * i := by
  sorry

end complex_number_equality_l1665_166523


namespace candidate_a_votes_l1665_166553

/-- Proves that given a ratio of 2:1 for votes between two candidates and a total of 21 votes,
    the candidate with the higher number of votes received 14 votes. -/
theorem candidate_a_votes (total_votes : ℕ) (ratio_a : ℕ) (ratio_b : ℕ) : 
  total_votes = 21 → ratio_a = 2 → ratio_b = 1 → 
  (ratio_a * total_votes) / (ratio_a + ratio_b) = 14 := by
sorry

end candidate_a_votes_l1665_166553


namespace quadratic_roots_problem_l1665_166554

theorem quadratic_roots_problem (a b n r s : ℝ) : 
  a^2 - n*a + 6 = 0 →
  b^2 - n*b + 6 = 0 →
  (a + 1/b)^2 - r*(a + 1/b) + s = 0 →
  (b + 1/a)^2 - r*(b + 1/a) + s = 0 →
  s = 49/6 := by
sorry

end quadratic_roots_problem_l1665_166554


namespace squirrels_and_nuts_l1665_166535

theorem squirrels_and_nuts :
  let num_squirrels : ℕ := 4
  let num_nuts : ℕ := 2
  num_squirrels - num_nuts = 2 :=
by sorry

end squirrels_and_nuts_l1665_166535


namespace union_of_A_and_B_l1665_166529

def A : Set ℤ := {0, -2}
def B : Set ℤ := {-4, 0}

theorem union_of_A_and_B :
  A ∪ B = {-4, -2, 0} := by sorry

end union_of_A_and_B_l1665_166529


namespace cost_price_per_meter_l1665_166555

/-- 
Given a trader who sells cloth with the following conditions:
- total_meters: The total number of meters of cloth sold
- selling_price: The total selling price for all meters of cloth
- profit_per_meter: The profit made per meter of cloth

This theorem proves that the cost price per meter of cloth is equal to
(selling_price - (total_meters * profit_per_meter)) / total_meters
-/
theorem cost_price_per_meter 
  (total_meters : ℕ) 
  (selling_price profit_per_meter : ℚ) 
  (h1 : total_meters = 85)
  (h2 : selling_price = 8925)
  (h3 : profit_per_meter = 5) :
  (selling_price - (total_meters : ℚ) * profit_per_meter) / total_meters = 100 :=
by sorry

end cost_price_per_meter_l1665_166555


namespace min_removed_length_345_square_l1665_166584

/-- Represents a right-angled triangle with integer side lengths -/
structure RightTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  is_right_angled : a^2 + b^2 = c^2

/-- Represents a square formed by four right-angled triangles -/
structure TriangleSquare where
  triangle : RightTriangle
  side_length : ℕ
  is_valid : side_length = triangle.a + triangle.b

/-- The minimum length of line segments to be removed to make the figure drawable in one stroke -/
def min_removed_length (square : TriangleSquare) : ℕ := sorry

/-- Theorem stating that the minimum length of removed line segments is 7 for a square formed by four 3-4-5 triangles -/
theorem min_removed_length_345_square :
  ∀ (square : TriangleSquare),
    square.triangle = { a := 3, b := 4, c := 5, is_right_angled := by norm_num }
    → min_removed_length square = 7 := by sorry

end min_removed_length_345_square_l1665_166584


namespace solution_x_l1665_166556

theorem solution_x (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 143) : x = 17 := by
  sorry

end solution_x_l1665_166556


namespace exponent_problem_l1665_166507

theorem exponent_problem (x y : ℝ) (m n : ℕ) (h : x ≠ 0) (h' : y ≠ 0) :
  x^m * y^n / ((1/4) * x^3 * y) = 4 * x^2 → m = 5 ∧ n = 1 := by
  sorry

end exponent_problem_l1665_166507


namespace hoseok_fruit_difference_l1665_166599

/-- The number of lemons eaten minus the number of pears eaten by Hoseok -/
def lemon_pear_difference (apples pears tangerines lemons watermelons : ℕ) : ℤ :=
  lemons - pears

theorem hoseok_fruit_difference :
  lemon_pear_difference 8 5 12 17 10 = 12 := by
  sorry

end hoseok_fruit_difference_l1665_166599


namespace tim_manicure_payment_l1665_166508

/-- The total amount paid for a manicure with tip, given the base cost and tip percentage. -/
def total_paid (base_cost : ℝ) (tip_percentage : ℝ) : ℝ :=
  base_cost * (1 + tip_percentage)

/-- Theorem stating that the total amount Tim paid for the manicure is $39. -/
theorem tim_manicure_payment : total_paid 30 0.3 = 39 := by
  sorry

end tim_manicure_payment_l1665_166508


namespace floor_sqrt_20_squared_l1665_166550

theorem floor_sqrt_20_squared : ⌊Real.sqrt 20⌋^2 = 16 := by sorry

end floor_sqrt_20_squared_l1665_166550


namespace largest_multiple_80_correct_l1665_166531

/-- Returns true if all digits of n are either 8 or 0 -/
def allDigits80 (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 8 ∨ d = 0

/-- The largest multiple of 20 with all digits 8 or 0 -/
def largestMultiple80 : ℕ := 8880

theorem largest_multiple_80_correct :
  largestMultiple80 % 20 = 0 ∧
  allDigits80 largestMultiple80 ∧
  ∀ n : ℕ, n > largestMultiple80 → ¬(n % 20 = 0 ∧ allDigits80 n) :=
by sorry

end largest_multiple_80_correct_l1665_166531


namespace initial_walking_time_l1665_166552

/-- Proves that given a person walking at 5 kilometers per hour, if they need 3 more hours to reach a total of 30 kilometers, then they have already walked for 3 hours. -/
theorem initial_walking_time (speed : ℝ) (additional_hours : ℝ) (total_distance : ℝ) 
  (h1 : speed = 5)
  (h2 : additional_hours = 3)
  (h3 : total_distance = 30) :
  (total_distance - additional_hours * speed) / speed = 3 := by
  sorry

end initial_walking_time_l1665_166552


namespace circle_angle_sum_l1665_166565

/-- Given a circle divided into 12 equal arcs, this theorem proves that the sum of
    half the central angle spanning 2 arcs and half the central angle spanning 4 arcs
    is equal to 90 degrees. -/
theorem circle_angle_sum (α β : Real) : 
  (∀ (n : Nat), n ≤ 12 → 360 / 12 * n = 30 * n) →
  α = (2 * 360 / 12) / 2 →
  β = (4 * 360 / 12) / 2 →
  α + β = 90 := by
sorry

end circle_angle_sum_l1665_166565


namespace blue_green_difference_l1665_166527

/-- Represents a hexagonal figure with blue and green tiles -/
structure HexagonalFigure where
  blue_tiles : ℕ
  green_tiles : ℕ

/-- Adds a border of tiles to a hexagonal figure -/
def add_border (figure : HexagonalFigure) : HexagonalFigure :=
  { blue_tiles := figure.blue_tiles + 18,
    green_tiles := figure.green_tiles + 18 }

/-- The initial hexagonal figure -/
def initial_figure : HexagonalFigure :=
  { blue_tiles := 15, green_tiles := 9 }

/-- The new figure after adding both borders -/
def new_figure : HexagonalFigure :=
  add_border (add_border initial_figure)

theorem blue_green_difference :
  new_figure.blue_tiles - new_figure.green_tiles = 6 := by
  sorry

end blue_green_difference_l1665_166527


namespace angle_measure_in_pentagon_and_triangle_l1665_166591

/-- Given a pentagon with angles A, B, C, E, and F, where angles D, E, and F form a triangle,
    this theorem proves that if m∠A = 80°, m∠B = 30°, and m∠C = 20°, then m∠D = 130°. -/
theorem angle_measure_in_pentagon_and_triangle 
  (A B C D E F : Real) 
  (pentagon : A + B + C + E + F = 540) 
  (triangle : D + E + F = 180) 
  (angle_A : A = 80) 
  (angle_B : B = 30) 
  (angle_C : C = 20) : 
  D = 130 := by
sorry

end angle_measure_in_pentagon_and_triangle_l1665_166591


namespace age_difference_l1665_166581

theorem age_difference (A B : ℕ) : B = 41 → A + 10 = 2 * (B - 10) → A - B = 11 := by
  sorry

end age_difference_l1665_166581


namespace high_school_students_l1665_166587

theorem high_school_students (music : ℕ) (art : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : music = 50)
  (h2 : art = 20)
  (h3 : both = 10)
  (h4 : neither = 440) :
  music + art - both + neither = 500 := by
  sorry

end high_school_students_l1665_166587


namespace isabel_spending_ratio_l1665_166502

/-- Given Isabel's initial amount, toy purchase, and final remaining amount,
    prove that the ratio of book cost to money after toy purchase is 1:2 -/
theorem isabel_spending_ratio (initial_amount : ℕ) (remaining_amount : ℕ)
    (h1 : initial_amount = 204)
    (h2 : remaining_amount = 51) :
  let toy_cost : ℕ := initial_amount / 2
  let after_toy : ℕ := initial_amount - toy_cost
  let book_cost : ℕ := after_toy - remaining_amount
  (book_cost : ℚ) / after_toy = 1 / 2 := by
sorry

end isabel_spending_ratio_l1665_166502


namespace even_red_faces_count_l1665_166513

/-- Represents a rectangular block with given dimensions -/
structure Block where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Counts the number of cubes with an even number of red faces in a painted block -/
def countEvenRedFaces (b : Block) : ℕ :=
  sorry

/-- The main theorem stating that a 6x4x2 block has 24 cubes with an even number of red faces -/
theorem even_red_faces_count (b : Block) (h1 : b.length = 6) (h2 : b.width = 4) (h3 : b.height = 2) :
  countEvenRedFaces b = 24 := by
  sorry

#check even_red_faces_count

end even_red_faces_count_l1665_166513


namespace arctan_sum_not_standard_angle_l1665_166594

theorem arctan_sum_not_standard_angle :
  let a : ℝ := 2/3
  let b : ℝ := (3 / (5/3)) - 1
  ¬(Real.arctan a + Real.arctan b = π/2 ∨
    Real.arctan a + Real.arctan b = π/3 ∨
    Real.arctan a + Real.arctan b = π/4 ∨
    Real.arctan a + Real.arctan b = π/5 ∨
    Real.arctan a + Real.arctan b = π/6) :=
by
  sorry

end arctan_sum_not_standard_angle_l1665_166594


namespace min_value_sum_reciprocals_min_value_sum_reciprocals_achievable_l1665_166566

theorem min_value_sum_reciprocals (p q r s t u : ℝ) 
  (hp : p > 0) (hq : q > 0) (hr : r > 0) (hs : s > 0) (ht : t > 0) (hu : u > 0)
  (hsum : p + q + r + s + t + u = 11) :
  1/p + 9/q + 25/r + 49/s + 81/t + 121/u ≥ 1296/11 := by
  sorry

theorem min_value_sum_reciprocals_achievable (ε : ℝ) (hε : ε > 0) :
  ∃ p q r s t u : ℝ, 
    p > 0 ∧ q > 0 ∧ r > 0 ∧ s > 0 ∧ t > 0 ∧ u > 0 ∧
    p + q + r + s + t + u = 11 ∧
    1/p + 9/q + 25/r + 49/s + 81/t + 121/u < 1296/11 + ε := by
  sorry

end min_value_sum_reciprocals_min_value_sum_reciprocals_achievable_l1665_166566


namespace vegetable_price_calculation_l1665_166544

/-- The price of vegetables and the final cost after discount -/
theorem vegetable_price_calculation :
  let cucumber_price : ℝ := 5
  let tomato_price : ℝ := cucumber_price * 0.8
  let bell_pepper_price : ℝ := cucumber_price * 1.5
  let total_cost : ℝ := 2 * tomato_price + 3 * cucumber_price + 4 * bell_pepper_price
  let discount_rate : ℝ := 0.1
  let final_price : ℝ := total_cost * (1 - discount_rate)
  final_price = 47.7 := by
sorry


end vegetable_price_calculation_l1665_166544


namespace unique_two_digit_number_with_reverse_difference_64_l1665_166592

theorem unique_two_digit_number_with_reverse_difference_64 :
  ∃! N : ℕ, 
    (N ≥ 10 ∧ N < 100) ∧ 
    (∃ a : ℕ, a < 10 ∧ N = 10 * a + 1) ∧
    ((10 * (N % 10) + N / 10) - N = 64) := by
  sorry

end unique_two_digit_number_with_reverse_difference_64_l1665_166592


namespace option_C_most_suitable_l1665_166539

/-- Represents a survey option -/
inductive SurveyOption
  | A  -- Understanding the sleep time of middle school students nationwide
  | B  -- Understanding the water quality of a river
  | C  -- Surveying the vision of all classmates
  | D  -- Surveying the number of fish in a pond

/-- Defines what makes a survey comprehensive -/
def isComprehensive (s : SurveyOption) : Prop :=
  match s with
  | SurveyOption.C => true
  | _ => false

/-- Theorem stating that option C is the most suitable for a comprehensive survey -/
theorem option_C_most_suitable :
  ∀ s : SurveyOption, isComprehensive s → s = SurveyOption.C :=
sorry

end option_C_most_suitable_l1665_166539


namespace sin_translation_l1665_166505

theorem sin_translation (x : ℝ) :
  let f : ℝ → ℝ := λ x => Real.sin (2 * x + π / 3)
  let translation : ℝ := π / 6
  let result : ℝ → ℝ := λ x => Real.sin (2 * x + 2 * π / 3)
  (λ x => f (x + translation)) = result := by
sorry

end sin_translation_l1665_166505


namespace min_sum_at_6_l1665_166545

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence
  S : ℕ → ℚ  -- The sum function

/-- The conditions of the problem -/
def problem_conditions (seq : ArithmeticSequence) : Prop :=
  seq.S 10 = -2 ∧ seq.S 20 = 16

/-- The main theorem -/
theorem min_sum_at_6 (seq : ArithmeticSequence) 
  (h : problem_conditions seq) :
  ∀ n : ℕ, n ≠ 0 → seq.S 6 ≤ seq.S n :=
sorry

end min_sum_at_6_l1665_166545


namespace four_letter_initials_count_l1665_166559

theorem four_letter_initials_count : 
  let letter_count : ℕ := 10
  let initial_length : ℕ := 4
  let order_matters : Bool := true
  let allow_repetition : Bool := true
  (letter_count ^ initial_length : ℕ) = 10000 := by
  sorry

end four_letter_initials_count_l1665_166559


namespace five_balls_four_boxes_l1665_166500

/-- The number of ways to place n distinguishable objects into k distinguishable containers -/
def placement_count (n k : ℕ) : ℕ := k^n

/-- Theorem: The number of ways to place 5 distinguishable balls into 4 distinguishable boxes is 1024 -/
theorem five_balls_four_boxes : placement_count 5 4 = 1024 := by
  sorry

end five_balls_four_boxes_l1665_166500


namespace badge_exchange_problem_l1665_166578

theorem badge_exchange_problem (vasya_initial tolya_initial : ℕ) : 
  vasya_initial = 50 ∧ tolya_initial = 45 →
  vasya_initial = tolya_initial + 5 ∧
  (vasya_initial - (vasya_initial * 24 / 100) + (tolya_initial * 20 / 100)) + 1 =
  (tolya_initial - (tolya_initial * 20 / 100) + (vasya_initial * 24 / 100)) :=
by sorry

end badge_exchange_problem_l1665_166578


namespace aqua_park_earnings_l1665_166534

/-- Calculate the total earnings of an aqua park given admission cost, tour cost, and group sizes. -/
theorem aqua_park_earnings
  (admission_cost : ℕ)
  (tour_cost : ℕ)
  (group1_size : ℕ)
  (group2_size : ℕ)
  (h1 : admission_cost = 12)
  (h2 : tour_cost = 6)
  (h3 : group1_size = 10)
  (h4 : group2_size = 5) :
  (group1_size * (admission_cost + tour_cost)) + (group2_size * admission_cost) = 240 := by
  sorry

#check aqua_park_earnings

end aqua_park_earnings_l1665_166534


namespace delta_properties_l1665_166521

def delta (m n : ℚ) : ℚ := (m + n) / (1 + m * n)

theorem delta_properties :
  (delta (-4) 4 = 0) ∧
  (delta (1/3) (1/4) = delta 3 4) ∧
  ∃ (m n : ℚ), delta (-m) n ≠ delta m (-n) := by
  sorry

end delta_properties_l1665_166521


namespace cube_congruence_l1665_166532

theorem cube_congruence (a b : ℕ) : a ≡ b [MOD 1000] → a^3 ≡ b^3 [MOD 1000] := by
  sorry

end cube_congruence_l1665_166532


namespace sum_of_squares_l1665_166530

theorem sum_of_squares (a b c : ℝ) 
  (h1 : (a + b + c) / 3 = 10)
  (h2 : (a * b * c)^(1/3 : ℝ) = 6)
  (h3 : 3 / (1/a + 1/b + 1/c) = 4) :
  a^2 + b^2 + c^2 = 576 := by sorry

end sum_of_squares_l1665_166530


namespace partial_fraction_decomposition_l1665_166597

theorem partial_fraction_decomposition (A B C : ℝ) :
  (∀ x : ℝ, x ≠ -5 ∧ x ≠ 3 →
    1 / (x^3 - x^2 - 21*x + 45) = A / (x + 5) + B / (x - 3) + C / ((x - 3)^2)) →
  A = 1/64 := by
sorry

end partial_fraction_decomposition_l1665_166597


namespace teacher_wang_travel_time_l1665_166542

theorem teacher_wang_travel_time (bicycle_speed : ℝ) (bicycle_time : ℝ) (walking_speed : ℝ) (max_walking_time : ℝ)
  (h1 : bicycle_speed = 15)
  (h2 : bicycle_time = 0.2)
  (h3 : walking_speed = 5)
  (h4 : max_walking_time = 0.7) :
  (bicycle_speed * bicycle_time) / walking_speed < max_walking_time :=
by sorry

end teacher_wang_travel_time_l1665_166542


namespace difference_of_squares_form_l1665_166546

theorem difference_of_squares_form (x y : ℝ) : 
  ∃ (a b : ℝ), (-x + y) * (x + y) = a^2 - b^2 := by sorry

end difference_of_squares_form_l1665_166546


namespace min_sum_bound_min_sum_achievable_l1665_166575

theorem min_sum_bound (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a / (3 * b) + b / (6 * c) + c / (9 * a) ≥ 3 / Real.rpow 162 (1/3) :=
by sorry

theorem min_sum_achievable :
  ∃ (a b c : ℝ), 0 < a ∧ 0 < b ∧ 0 < c ∧
    a / (3 * b) + b / (6 * c) + c / (9 * a) = 3 / Real.rpow 162 (1/3) :=
by sorry

end min_sum_bound_min_sum_achievable_l1665_166575


namespace three_even_out_of_five_probability_l1665_166501

/-- A fair 10-sided die -/
def TenSidedDie : Type := Fin 10

/-- The probability of rolling an even number on a 10-sided die -/
def probEven : ℚ := 1/2

/-- The number of dice rolled -/
def numDice : ℕ := 5

/-- The number of dice that should show an even number -/
def numEven : ℕ := 3

/-- The probability of exactly three out of five 10-sided dice showing an even number -/
def probThreeEvenOutOfFive : ℚ := 5/16

theorem three_even_out_of_five_probability :
  (Nat.choose numDice numEven : ℚ) * probEven^numEven * (1 - probEven)^(numDice - numEven) = probThreeEvenOutOfFive :=
sorry

end three_even_out_of_five_probability_l1665_166501


namespace coin_flip_probability_l1665_166522

theorem coin_flip_probability (n : ℕ) : n = 7 → (n.choose 2 : ℚ) / 2^n = 7 / 32 := by
  sorry

end coin_flip_probability_l1665_166522


namespace equation_solution_l1665_166541

theorem equation_solution (a : ℚ) : 
  (∀ x, a * x - 4 * (x - a) = 1) → (a * 2 - 4 * (2 - a) = 1) → a = 3/2 := by
  sorry

end equation_solution_l1665_166541


namespace function_behavior_l1665_166526

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

theorem function_behavior (f : ℝ → ℝ) :
  is_even_function f →
  has_period f 2 →
  is_decreasing_on f (-1) 0 →
  (is_increasing_on f 6 7 ∧ is_decreasing_on f 7 8) :=
by sorry

end function_behavior_l1665_166526


namespace smallest_c_value_l1665_166588

theorem smallest_c_value (c d : ℝ) (h_nonneg_c : c ≥ 0) (h_nonneg_d : d ≥ 0)
  (h_cos_eq : ∀ x : ℤ, Real.cos (c * ↑x - d) = Real.cos (35 * ↑x)) :
  c ≥ 35 ∧ ∀ c' ≥ 0, (∀ x : ℤ, Real.cos (c' * ↑x - d) = Real.cos (35 * ↑x)) → c' ≥ c :=
by sorry

end smallest_c_value_l1665_166588


namespace alice_paid_24_percent_l1665_166560

-- Define the suggested retail price
def suggested_retail_price : ℝ := 100

-- Define the marked price as 60% of the suggested retail price
def marked_price : ℝ := 0.6 * suggested_retail_price

-- Define Alice's purchase price as 40% of the marked price
def alice_price : ℝ := 0.4 * marked_price

-- Theorem to prove
theorem alice_paid_24_percent :
  alice_price / suggested_retail_price = 0.24 := by
  sorry

end alice_paid_24_percent_l1665_166560
