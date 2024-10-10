import Mathlib

namespace factors_of_81_l1770_177035

theorem factors_of_81 : Finset.card (Nat.divisors 81) = 5 := by
  sorry

end factors_of_81_l1770_177035


namespace tip_percentage_calculation_l1770_177063

theorem tip_percentage_calculation (total_bill : ℝ) (num_people : ℕ) (individual_payment : ℝ) :
  total_bill = 139 ∧ num_people = 5 ∧ individual_payment = 30.58 →
  (individual_payment * num_people - total_bill) / total_bill * 100 = 10 := by
  sorry

end tip_percentage_calculation_l1770_177063


namespace speedster_roadster_convertibles_l1770_177007

/-- Represents the inventory of an automobile company -/
structure Inventory where
  total : ℕ
  speedsters : ℕ
  roadsters : ℕ
  cruisers : ℕ
  speedster_convertibles : ℕ
  roadster_convertibles : ℕ
  cruiser_convertibles : ℕ

/-- Theorem stating the number of Speedster and Roadster convertibles -/
theorem speedster_roadster_convertibles (inv : Inventory) : 
  inv.speedster_convertibles + inv.roadster_convertibles = 52 :=
  by
  have h1 : inv.total = 100 := by sorry
  have h2 : inv.speedsters = inv.total * 2 / 5 := by sorry
  have h3 : inv.roadsters = inv.total * 3 / 10 := by sorry
  have h4 : inv.cruisers = inv.total - inv.speedsters - inv.roadsters := by sorry
  have h5 : inv.speedster_convertibles = inv.speedsters * 4 / 5 := by sorry
  have h6 : inv.roadster_convertibles = inv.roadsters * 2 / 3 := by sorry
  have h7 : inv.cruiser_convertibles = inv.cruisers * 1 / 4 := by sorry
  have h8 : inv.total - inv.speedsters = 60 := by sorry
  sorry

end speedster_roadster_convertibles_l1770_177007


namespace positive_X_value_l1770_177066

def hash (X Y : ℝ) : ℝ := X^2 + Y^2

theorem positive_X_value (X : ℝ) (h : hash X 7 = 290) : X = 17 := by
  sorry

end positive_X_value_l1770_177066


namespace special_triangle_side_length_l1770_177001

/-- An equilateral triangle with a special interior point -/
structure SpecialTriangle where
  -- The side length of the equilateral triangle
  s : ℝ
  -- The coordinates of the interior point P
  P : ℝ × ℝ
  -- Condition that the triangle is equilateral
  equilateral : s > 0
  -- Conditions for distances from P to vertices
  dist_AP : Real.sqrt ((P.1 - s/2)^2 + (P.2 - Real.sqrt 3 * s/2)^2) = Real.sqrt 2
  dist_BP : Real.sqrt ((P.1 - s)^2 + P.2^2) = 2
  dist_CP : Real.sqrt P.1^2 + P.2^2 = 1

/-- The side length of a special triangle is 5 -/
theorem special_triangle_side_length (t : SpecialTriangle) : t.s = 5 := by
  sorry


end special_triangle_side_length_l1770_177001


namespace rationalize_denominator_l1770_177028

theorem rationalize_denominator :
  ∃ (A B C D E : ℤ),
    (3 : ℝ) / (4 * Real.sqrt 7 + 3 * Real.sqrt 13) = 
      (A * Real.sqrt B + C * Real.sqrt D) / E ∧
    B < D ∧
    A = -12 ∧ B = 7 ∧ C = 9 ∧ D = 13 ∧ E = 5 :=
by sorry

end rationalize_denominator_l1770_177028


namespace systematic_sampling_probability_l1770_177029

/-- Represents a systematic sampling scenario -/
structure SystematicSampling where
  population : ℕ
  sampleSize : ℕ
  sampleSize_le_population : sampleSize ≤ population

/-- The probability of an individual being selected in a systematic sampling -/
def selectionProbability (s : SystematicSampling) : ℚ :=
  s.sampleSize / s.population

theorem systematic_sampling_probability 
  (s : SystematicSampling) 
  (h1 : s.population = 121) 
  (h2 : s.sampleSize = 12) : 
  selectionProbability s = 12 / 121 := by
  sorry

#check systematic_sampling_probability

end systematic_sampling_probability_l1770_177029


namespace quadratic_inequality_empty_solution_l1770_177014

theorem quadratic_inequality_empty_solution (a : ℝ) : 
  (∀ x : ℝ, x^2 - 4*x + a^2 > 0) → (a < -2 ∨ a > 2) := by
  sorry

end quadratic_inequality_empty_solution_l1770_177014


namespace interest_group_signups_l1770_177051

theorem interest_group_signups :
  let num_students : ℕ := 5
  let num_groups : ℕ := 3
  num_groups ^ num_students = 243 :=
by sorry

end interest_group_signups_l1770_177051


namespace tank_capacity_l1770_177041

theorem tank_capacity : ∃ (capacity : ℚ), 
  capacity > 0 ∧ 
  (1/3 : ℚ) * capacity + 180 = (2/3 : ℚ) * capacity ∧ 
  capacity = 540 := by
  sorry

end tank_capacity_l1770_177041


namespace sqrt_square_negative_two_l1770_177005

theorem sqrt_square_negative_two : Real.sqrt ((-2)^2) = 2 := by
  sorry

end sqrt_square_negative_two_l1770_177005


namespace a_must_be_positive_l1770_177091

theorem a_must_be_positive
  (a b c d : ℝ)
  (h1 : b ≠ 0)
  (h2 : d ≠ 0)
  (h3 : d > 0)
  (h4 : a / b > -(3 / (2 * d))) :
  a > 0 :=
by sorry

end a_must_be_positive_l1770_177091


namespace tetrahedron_has_six_edges_l1770_177002

/-- A tetrahedron is a three-dimensional geometric shape with four triangular faces. -/
structure Tetrahedron where
  vertices : Finset (Fin 4)
  faces : Finset (Finset (Fin 3))
  is_valid : faces.card = 4 ∧ ∀ f ∈ faces, f.card = 3

/-- The number of edges in a tetrahedron -/
def num_edges (t : Tetrahedron) : ℕ := sorry

/-- Theorem: A tetrahedron has exactly 6 edges -/
theorem tetrahedron_has_six_edges (t : Tetrahedron) : num_edges t = 6 := by sorry

end tetrahedron_has_six_edges_l1770_177002


namespace odometer_reading_before_trip_l1770_177053

theorem odometer_reading_before_trip 
  (odometer_at_lunch : ℝ) 
  (miles_traveled : ℝ) 
  (h1 : odometer_at_lunch = 372.0)
  (h2 : miles_traveled = 159.7) :
  odometer_at_lunch - miles_traveled = 212.3 := by
sorry

end odometer_reading_before_trip_l1770_177053


namespace min_sin_minus_cos_half_angle_l1770_177096

theorem min_sin_minus_cos_half_angle :
  let f : ℝ → ℝ := λ A ↦ Real.sin (A / 2) - Real.cos (A / 2)
  ∃ (min : ℝ) (A : ℝ), 
    (∀ x, f x ≥ min) ∧ 
    (f A = min) ∧ 
    (min = -Real.sqrt 2) ∧ 
    (A = 7 * Real.pi / 2) :=
by sorry

end min_sin_minus_cos_half_angle_l1770_177096


namespace inequality_solution_sets_l1770_177082

theorem inequality_solution_sets (a : ℝ) :
  (∀ x, ax^2 + 5*x - 2 > 0 ↔ 1/2 < x ∧ x < 2) →
  (∀ x, ax^2 - 5*x + a^2 - 1 > 0 ↔ -3 < x ∧ x < 1/2) :=
by sorry

end inequality_solution_sets_l1770_177082


namespace inverse_variation_cube_fourth_l1770_177077

/-- Given that x³ varies inversely with y⁴, and x = 2 when y = 4,
    prove that x³ = 1/2 when y = 8 -/
theorem inverse_variation_cube_fourth (k : ℝ) :
  (∀ x y : ℝ, x^3 * y^4 = k) →
  (2^3 * 4^4 = k) →
  ∃ x : ℝ, x^3 * 8^4 = k ∧ x^3 = (1/2 : ℝ) :=
by sorry

end inverse_variation_cube_fourth_l1770_177077


namespace xiao_hua_at_13_l1770_177068

/-- The floor Xiao Hua reaches when Xiao Li reaches a given floor -/
def xiao_hua_floor (xiao_li_floor : ℕ) : ℕ :=
  1 + ((xiao_li_floor - 1) * (3 - 1)) / (5 - 1)

/-- Theorem: When Xiao Li reaches the 25th floor, Xiao Hua will have reached the 13th floor -/
theorem xiao_hua_at_13 : xiao_hua_floor 25 = 13 := by
  sorry

end xiao_hua_at_13_l1770_177068


namespace pond_volume_calculation_l1770_177030

/-- The volume of a rectangular pond -/
def pond_volume (length width depth : ℝ) : ℝ := length * width * depth

/-- Theorem: The volume of a rectangular pond with dimensions 20 m x 10 m x 5 m is 1000 cubic meters -/
theorem pond_volume_calculation : pond_volume 20 10 5 = 1000 := by
  sorry

end pond_volume_calculation_l1770_177030


namespace earthworm_catches_centipede_l1770_177099

/-- The time (in minutes) it takes for an earthworm to catch up with a centipede given their speeds and initial distance -/
def catch_up_time (centipede_speed earthworm_speed initial_distance : ℚ) : ℚ :=
  initial_distance / (earthworm_speed - centipede_speed)

/-- Theorem stating that under the given conditions, the earthworm catches up with the centipede in 24 minutes -/
theorem earthworm_catches_centipede :
  let centipede_speed : ℚ := 5 / 3  -- 5 meters in 3 minutes
  let earthworm_speed : ℚ := 5 / 2  -- 5 meters in 2 minutes
  let initial_distance : ℚ := 20    -- 20 meters ahead
  catch_up_time centipede_speed earthworm_speed initial_distance = 24 := by
  sorry

#eval catch_up_time (5/3) (5/2) 20

end earthworm_catches_centipede_l1770_177099


namespace books_not_sold_l1770_177070

theorem books_not_sold (initial_stock : ℕ) (monday_sales tuesday_sales wednesday_sales thursday_sales friday_sales : ℕ) :
  initial_stock = 800 →
  monday_sales = 60 →
  tuesday_sales = 10 →
  wednesday_sales = 20 →
  thursday_sales = 44 →
  friday_sales = 66 →
  initial_stock - (monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales) = 600 := by
  sorry

end books_not_sold_l1770_177070


namespace profit_percentage_previous_year_l1770_177025

theorem profit_percentage_previous_year 
  (R : ℝ) -- Revenues in the previous year
  (P : ℝ) -- Profits in the previous year
  (h1 : 0.95 * R * 0.10 = 0.95 * P) -- Condition relating 2009 profits to previous year
  : P / R = 0.10 := by
  sorry

end profit_percentage_previous_year_l1770_177025


namespace ants_meet_at_66cm_l1770_177072

/-- Represents a point on the tile grid -/
structure GridPoint where
  x : ℕ
  y : ℕ

/-- Represents a movement on the grid -/
inductive GridMove
  | Right
  | Up
  | Left
  | Down

/-- The path of an ant on the grid -/
def AntPath := List GridMove

/-- Calculate the distance traveled given a path -/
def pathDistance (path : AntPath) (tileWidth tileLength : ℕ) : ℕ :=
  path.foldl (fun acc move =>
    acc + match move with
      | GridMove.Right => tileLength
      | GridMove.Up => tileWidth
      | GridMove.Left => tileLength
      | GridMove.Down => tileWidth) 0

/-- Check if two paths meet at the same point -/
def pathsMeet (path1 path2 : AntPath) (start1 start2 : GridPoint) : Prop :=
  sorry

theorem ants_meet_at_66cm (tileWidth tileLength : ℕ) (startM startN : GridPoint) 
    (pathM pathN : AntPath) : 
  tileWidth = 4 →
  tileLength = 6 →
  startM = ⟨0, 0⟩ →
  startN = ⟨14, 12⟩ →
  pathsMeet pathM pathN startM startN →
  pathDistance pathM tileWidth tileLength = 66 ∧
  pathDistance pathN tileWidth tileLength = 66 :=
by
  sorry

#check ants_meet_at_66cm

end ants_meet_at_66cm_l1770_177072


namespace max_cars_quotient_l1770_177034

/-- Represents the maximum number of cars that can pass a point on the highway in one hour -/
def M : ℕ :=
  -- Definition to be proved
  2000

/-- The length of each car in meters -/
def car_length : ℝ := 5

/-- Theorem stating that M divided by 10 equals 200 -/
theorem max_cars_quotient :
  M / 10 = 200 := by sorry

end max_cars_quotient_l1770_177034


namespace perpendicular_to_plane_implies_parallel_l1770_177095

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem perpendicular_to_plane_implies_parallel 
  (m n : Line) (α : Plane) 
  (h1 : perpendicular m α) 
  (h2 : perpendicular n α) 
  (h3 : m ≠ n) : 
  parallel m n :=
sorry

end perpendicular_to_plane_implies_parallel_l1770_177095


namespace headphones_to_case_ratio_l1770_177037

def phone_cost : ℚ := 1000
def contract_cost_per_month : ℚ := 200
def case_cost_percentage : ℚ := 20 / 100
def total_first_year_cost : ℚ := 3700

def case_cost : ℚ := phone_cost * case_cost_percentage
def contract_cost_year : ℚ := contract_cost_per_month * 12
def headphones_cost : ℚ := total_first_year_cost - (phone_cost + case_cost + contract_cost_year)

theorem headphones_to_case_ratio :
  headphones_cost / case_cost = 1 / 2 := by sorry

end headphones_to_case_ratio_l1770_177037


namespace new_bill_is_35_l1770_177031

/-- Calculates the new total bill after substitutions and delivery/tip --/
def calculate_new_bill (original_order : ℝ) 
                       (tomato_old tomato_new : ℝ) 
                       (lettuce_old lettuce_new : ℝ) 
                       (celery_old celery_new : ℝ) 
                       (delivery_tip : ℝ) : ℝ :=
  original_order + 
  (tomato_new - tomato_old) + 
  (lettuce_new - lettuce_old) + 
  (celery_new - celery_old) + 
  delivery_tip

/-- Theorem stating that the new bill is $35.00 --/
theorem new_bill_is_35 : 
  calculate_new_bill 25 0.99 2.20 1.00 1.75 1.96 2.00 8.00 = 35 :=
by
  sorry

end new_bill_is_35_l1770_177031


namespace quadrilateral_inequality_l1770_177097

-- Define the points
variable (A B C D E : EuclideanSpace ℝ (Fin 2))

-- Define the quadrilateral ABCD
def is_convex_quadrilateral (A B C D : EuclideanSpace ℝ (Fin 2)) : Prop :=
  sorry

-- Define the intersection of diagonals
def diagonals_intersect (A B C D E : EuclideanSpace ℝ (Fin 2)) : Prop :=
  sorry

-- Define the distance function
def distance (P Q : EuclideanSpace ℝ (Fin 2)) : ℝ :=
  sorry

-- State the theorem
theorem quadrilateral_inequality 
  (h_convex : is_convex_quadrilateral A B C D)
  (h_intersect : diagonals_intersect A B C D E)
  (h_AB : distance A B = 1)
  (h_BC : distance B C = 1)
  (h_CD : distance C D = 1)
  (h_DE : distance D E = 1) :
  distance A D < 2 :=
sorry

end quadrilateral_inequality_l1770_177097


namespace n_equals_t_plus_2_l1770_177060

theorem n_equals_t_plus_2 (t : ℝ) (h : t ≠ 3) :
  let n := (4*t^2 - 10*t - 2 - 3*(t^2 - t + 3) + t^2 + 5*t - 1) / ((t + 7) + (t - 13))
  n = t + 2 := by sorry

end n_equals_t_plus_2_l1770_177060


namespace meeting_day_is_wednesday_l1770_177078

-- Define the days of the week
inductive Day
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday
| Sunday

-- Define the brothers
inductive Brother
| Tralalala
| Trulala

def lies (b : Brother) (d : Day) : Prop :=
  match b with
  | Brother.Tralalala => d = Day.Monday ∨ d = Day.Tuesday ∨ d = Day.Wednesday
  | Brother.Trulala => d = Day.Thursday ∨ d = Day.Friday ∨ d = Day.Saturday

def next_day (d : Day) : Day :=
  match d with
  | Day.Monday => Day.Tuesday
  | Day.Tuesday => Day.Wednesday
  | Day.Wednesday => Day.Thursday
  | Day.Thursday => Day.Friday
  | Day.Friday => Day.Saturday
  | Day.Saturday => Day.Sunday
  | Day.Sunday => Day.Monday

theorem meeting_day_is_wednesday :
  ∃ (b1 b2 : Brother) (d : Day),
    b1 ≠ b2 ∧
    (lies b1 Day.Saturday ↔ lies b1 d) ∧
    (lies b2 (next_day d) ↔ ¬(lies b2 d)) ∧
    (lies b1 Day.Sunday ↔ lies b1 d) ∧
    d = Day.Wednesday :=
  sorry


end meeting_day_is_wednesday_l1770_177078


namespace min_ratio_cone_cylinder_volumes_l1770_177036

/-- The minimum ratio of the volume of a cone to the volume of a cylinder, 
    both circumscribed around the same sphere, is 4/3. -/
theorem min_ratio_cone_cylinder_volumes : ℝ := by
  -- Let r be the radius of the sphere
  -- Let h be the height of the cone
  -- Let a be the radius of the base of the cone
  -- The cylinder has height 2r and radius r
  -- The ratio of volumes is (π * a^2 * h / 3) / (π * r^2 * 2r)
  -- We need to prove that the minimum value of this ratio is 4/3
  sorry

end min_ratio_cone_cylinder_volumes_l1770_177036


namespace square_diff_eq_three_implies_product_eq_nine_l1770_177052

theorem square_diff_eq_three_implies_product_eq_nine (x y : ℝ) :
  x^2 - y^2 = 3 → (x + y)^2 * (x - y)^2 = 9 := by
  sorry

end square_diff_eq_three_implies_product_eq_nine_l1770_177052


namespace smallest_x_for_inequality_l1770_177065

theorem smallest_x_for_inequality :
  ∃ (x : ℝ), x = 49 ∧
  (∀ (a : ℝ), a ≥ 0 → a ≥ 14 * Real.sqrt a - x) ∧
  (∀ (y : ℝ), y < x → ∃ (a : ℝ), a ≥ 0 ∧ a < 14 * Real.sqrt a - y) :=
by sorry

end smallest_x_for_inequality_l1770_177065


namespace no_rain_probability_l1770_177024

theorem no_rain_probability (p : ℚ) (h : p = 2/3) :
  (1 - p)^5 = 1/243 := by
  sorry

end no_rain_probability_l1770_177024


namespace roots_sum_powers_l1770_177000

theorem roots_sum_powers (a b : ℝ) : 
  a^2 - 3*a + 2 = 0 → b^2 - 3*b + 2 = 0 → a^3 + a^4*b^2 + a^2*b^4 + b^3 = 29 := by
  sorry

end roots_sum_powers_l1770_177000


namespace binomial_expansion_sum_l1770_177054

theorem binomial_expansion_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x y : ℝ, (x + 2 * Real.sqrt y)^5 = a₀*x^5 + a₁*x^4*(Real.sqrt y) + a₂*x^3*y + a₃*x^2*y*(Real.sqrt y) + a₄*x*y^2 + a₅*y^(5/2)) →
  a₁ + a₃ + a₅ = 122 := by
sorry

end binomial_expansion_sum_l1770_177054


namespace ratio_to_ten_l1770_177055

theorem ratio_to_ten : ∃ x : ℚ, (15 : ℚ) / 1 = x / 10 ∧ x = 150 := by
  sorry

end ratio_to_ten_l1770_177055


namespace trajectory_equation_l1770_177027

/-- The trajectory of point M satisfying the given conditions -/
def trajectory_of_M (x y : ℝ) : Prop :=
  x ≠ 0 ∧ y > 0 ∧ y = (1/4) * x^2

/-- Point P -/
def P : ℝ × ℝ := (0, -3)

/-- Point A on the x-axis -/
def A (a : ℝ) : ℝ × ℝ := (a, 0)

/-- Point Q on the positive y-axis -/
def Q (b : ℝ) : ℝ × ℝ := (0, b)

/-- Condition: Q is on the positive half of y-axis -/
def Q_positive (b : ℝ) : Prop := b > 0

/-- Vector PA -/
def vec_PA (a : ℝ) : ℝ × ℝ := (a - P.1, 0 - P.2)

/-- Vector AM -/
def vec_AM (a x y : ℝ) : ℝ × ℝ := (x - a, y)

/-- Vector MQ -/
def vec_MQ (x y b : ℝ) : ℝ × ℝ := (0 - x, b - y)

/-- Dot product of PA and AM is zero -/
def PA_dot_AM_zero (a x y : ℝ) : Prop :=
  (vec_PA a).1 * (vec_AM a x y).1 + (vec_PA a).2 * (vec_AM a x y).2 = 0

/-- AM = -3/2 * MQ -/
def AM_eq_neg_three_half_MQ (a x y b : ℝ) : Prop :=
  vec_AM a x y = (-3/2 : ℝ) • vec_MQ x y b

/-- The main theorem: given the conditions, prove that M follows the trajectory equation -/
theorem trajectory_equation (x y a b : ℝ) : 
  Q_positive b →
  PA_dot_AM_zero a x y →
  AM_eq_neg_three_half_MQ a x y b →
  trajectory_of_M x y :=
sorry

end trajectory_equation_l1770_177027


namespace point_coordinates_on_directed_segment_l1770_177067

/-- Given points M and N, and point P on the directed line segment MN such that MP = 3PN,
    prove that the coordinates of point P are (11/4, -1/4). -/
theorem point_coordinates_on_directed_segment (M N P : ℝ × ℝ) :
  M = (2, 5) →
  N = (3, -2) →
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • M + t • N) →
  3 • (N - P) = P - M →
  P = (11/4, -1/4) := by
sorry

end point_coordinates_on_directed_segment_l1770_177067


namespace game_terminates_l1770_177015

/-- Represents the state of knowledge for each player -/
structure PlayerKnowledge where
  lower : Nat
  upper : Nat

/-- Represents the game state -/
structure GameState where
  player1 : PlayerKnowledge
  player2 : PlayerKnowledge
  turn : Nat

/-- Updates the game state based on a negative response -/
def updateGameState (state : GameState) : GameState :=
  sorry

/-- Checks if a player knows the other's number -/
def knowsNumber (knowledge : PlayerKnowledge) : Bool :=
  sorry

/-- Simulates the game for a given initial state -/
def playGame (initialState : GameState) : Nat :=
  sorry

/-- Theorem stating that the game will terminate -/
theorem game_terminates (n : Nat) :
  ∃ (k : Nat), ∀ (m : Nat),
    let initialState : GameState := {
      player1 := { lower := 1, upper := n + 1 },
      player2 := { lower := 1, upper := n + 1 },
      turn := 0
    }
    playGame initialState ≤ k :=
  sorry

end game_terminates_l1770_177015


namespace find_n_l1770_177050

theorem find_n (n : ℕ) (h1 : Nat.lcm n 12 = 42) (h2 : Nat.gcd n 12 = 6) : n = 21 := by
  sorry

end find_n_l1770_177050


namespace x_cubed_minus_2x_plus_1_l1770_177081

theorem x_cubed_minus_2x_plus_1 (x : ℝ) (h : x^2 - x - 1 = 0) : x^3 - 2*x + 1 = 2 := by
  sorry

end x_cubed_minus_2x_plus_1_l1770_177081


namespace smallest_n_for_irreducible_fractions_l1770_177019

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

theorem smallest_n_for_irreducible_fractions : 
  ∃ (n : ℕ), n = 28 ∧ 
  (∀ k : ℕ, 5 ≤ k → k ≤ 24 → is_coprime k (n + k + 1)) ∧
  (∀ m : ℕ, m < n → ∃ k : ℕ, 5 ≤ k ∧ k ≤ 24 ∧ ¬is_coprime k (m + k + 1)) :=
sorry

end smallest_n_for_irreducible_fractions_l1770_177019


namespace calculate_expression_l1770_177049

theorem calculate_expression : (-7)^7 / 7^4 + 2^8 - 10^1 = -97 := by
  sorry

end calculate_expression_l1770_177049


namespace inscribed_cube_volume_l1770_177040

theorem inscribed_cube_volume (large_cube_edge : ℝ) (sphere_diameter : ℝ) (small_cube_edge : ℝ) :
  large_cube_edge = 12 →
  sphere_diameter = large_cube_edge →
  sphere_diameter = small_cube_edge * Real.sqrt 3 →
  small_cube_edge^3 = 192 * Real.sqrt 3 :=
by sorry

end inscribed_cube_volume_l1770_177040


namespace valid_parameterization_l1770_177033

/-- A vector parameterization of a line --/
structure VectorParam where
  v : ℝ × ℝ  -- point vector
  d : ℝ × ℝ  -- direction vector

/-- The line y = 2x - 5 --/
def line (x : ℝ) : ℝ := 2 * x - 5

/-- Check if a point lies on the line --/
def on_line (p : ℝ × ℝ) : Prop :=
  p.2 = line p.1

/-- Check if a vector is a scalar multiple of (1, 2) --/
def is_valid_direction (v : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v = (k, 2 * k)

/-- A parameterization is valid if it satisfies both conditions --/
def is_valid_param (param : VectorParam) : Prop :=
  on_line param.v ∧ is_valid_direction param.d

theorem valid_parameterization (param : VectorParam) :
  is_valid_param param ↔ 
    (∀ (t : ℝ), on_line (param.v.1 + t * param.d.1, param.v.2 + t * param.d.2)) :=
sorry

end valid_parameterization_l1770_177033


namespace inequality_proof_l1770_177004

theorem inequality_proof (x y z : ℝ) (h : x + 2*y + 3*z + 8 = 0) :
  (x - 1)^2 + (y + 2)^2 + (z - 3)^2 ≥ 14 := by
  sorry

end inequality_proof_l1770_177004


namespace johnny_age_multiple_l1770_177038

theorem johnny_age_multiple (current_age : ℕ) (m : ℕ+) : current_age = 8 →
  (current_age + 2 : ℕ) = m * (current_age - 3) →
  m = 2 := by
  sorry

end johnny_age_multiple_l1770_177038


namespace hike_distance_l1770_177087

/-- Represents a 5-day hike with given conditions -/
structure FiveDayHike where
  day1 : ℝ
  day2 : ℝ
  day3 : ℝ
  day4 : ℝ
  day5 : ℝ
  first_two_days : day1 + day2 = 28
  second_fourth_avg : (day2 + day4) / 2 = 15
  last_three_days : day3 + day4 + day5 = 42
  first_third_days : day1 + day3 = 30

/-- The total distance of the hike is 70 miles -/
theorem hike_distance (h : FiveDayHike) : h.day1 + h.day2 + h.day3 + h.day4 + h.day5 = 70 := by
  sorry

end hike_distance_l1770_177087


namespace wendy_small_glasses_l1770_177061

/-- The number of small glasses polished by Wendy -/
def small_glasses : ℕ := 50

/-- The number of large glasses polished by Wendy -/
def large_glasses : ℕ := small_glasses + 10

/-- The total number of glasses polished by Wendy -/
def total_glasses : ℕ := 110

/-- Proof that Wendy polished 50 small glasses -/
theorem wendy_small_glasses :
  small_glasses = 50 ∧
  large_glasses = small_glasses + 10 ∧
  small_glasses + large_glasses = total_glasses :=
by sorry

end wendy_small_glasses_l1770_177061


namespace intersection_points_count_l1770_177080

-- Define the properties of the function f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def monotone_increasing_pos (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → x < y → f x < f y

def opposite_signs_at_1_2 (f : ℝ → ℝ) : Prop := f 1 * f 2 < 0

-- Define the number of intersections with the x-axis
def num_intersections (f : ℝ → ℝ) : ℕ :=
  -- This is a placeholder definition
  -- In practice, this would be defined more rigorously
  2

-- State the theorem
theorem intersection_points_count
  (f : ℝ → ℝ)
  (h_even : is_even f)
  (h_mono : monotone_increasing_pos f)
  (h_signs : opposite_signs_at_1_2 f) :
  num_intersections f = 2 :=
sorry

end intersection_points_count_l1770_177080


namespace area_of_ring_area_of_specific_ring_l1770_177074

/-- The area of a ring-shaped region formed by two concentric circles -/
theorem area_of_ring (r₁ r₂ : ℝ) (h : r₁ > r₂) :
  (π * r₁^2 - π * r₂^2) = π * (r₁^2 - r₂^2) :=
by sorry

/-- The area of a ring-shaped region formed by two concentric circles with radii 12 and 5 -/
theorem area_of_specific_ring :
  π * (12^2 - 5^2) = 119 * π :=
by sorry

end area_of_ring_area_of_specific_ring_l1770_177074


namespace javier_first_throw_l1770_177062

/-- Represents the distances of three javelin throws -/
structure JavelinThrows where
  first : ℝ
  second : ℝ
  third : ℝ

/-- Conditions for Javier's javelin throws -/
def javierThrows (t : JavelinThrows) : Prop :=
  t.first = 2 * t.second ∧
  t.first = 1/2 * t.third ∧
  t.first + t.second + t.third = 1050

/-- Theorem stating that Javier's first throw was 300 meters -/
theorem javier_first_throw :
  ∀ t : JavelinThrows, javierThrows t → t.first = 300 := by
  sorry

end javier_first_throw_l1770_177062


namespace prob_at_least_one_boy_one_girl_l1770_177085

-- Define the probability of having a boy or a girl
def p_boy_or_girl : ℚ := 1 / 2

-- Define the number of children in the family
def num_children : ℕ := 4

-- Theorem statement
theorem prob_at_least_one_boy_one_girl :
  1 - (p_boy_or_girl ^ num_children + p_boy_or_girl ^ num_children) = 7 / 8 :=
sorry

end prob_at_least_one_boy_one_girl_l1770_177085


namespace triangle_expression_positive_l1770_177048

theorem triangle_expression_positive (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  0 < 4 * b^2 * c^2 - (b^2 + c^2 - a^2)^2 := by
  sorry

end triangle_expression_positive_l1770_177048


namespace total_books_read_is_48cs_l1770_177089

/-- The number of books read by the entire student body in one year -/
def total_books_read (c s : ℕ) : ℕ :=
  c * s * 4 * 12

/-- Theorem: The total number of books read by the entire student body in one year is 48cs -/
theorem total_books_read_is_48cs (c s : ℕ) : total_books_read c s = 48 * c * s := by
  sorry

end total_books_read_is_48cs_l1770_177089


namespace distance_las_vegas_to_los_angeles_l1770_177045

/-- Calculates the distance from Las Vegas to Los Angeles given the total drive time,
    average speed, and distance from Salt Lake City to Las Vegas. -/
theorem distance_las_vegas_to_los_angeles
  (total_time : ℝ)
  (average_speed : ℝ)
  (distance_salt_lake_to_vegas : ℝ)
  (h1 : total_time = 11)
  (h2 : average_speed = 63)
  (h3 : distance_salt_lake_to_vegas = 420) :
  total_time * average_speed - distance_salt_lake_to_vegas = 273 :=
by
  sorry

end distance_las_vegas_to_los_angeles_l1770_177045


namespace bracelet_bead_ratio_l1770_177032

/-- Proves that the ratio of small beads to large beads in each bracelet is 1:1 --/
theorem bracelet_bead_ratio
  (total_beads : ℕ)
  (bracelets : ℕ)
  (large_beads_per_bracelet : ℕ)
  (h1 : total_beads = 528)
  (h2 : bracelets = 11)
  (h3 : large_beads_per_bracelet = 12)
  (h4 : total_beads % 2 = 0)  -- Equal amounts of small and large beads
  (h5 : (total_beads / 2) ≥ (bracelets * large_beads_per_bracelet)) :
  (total_beads / 2 - bracelets * large_beads_per_bracelet) / bracelets = large_beads_per_bracelet :=
by sorry

end bracelet_bead_ratio_l1770_177032


namespace weight_of_grapes_l1770_177021

/-- Given the weights of fruits ordered by Tommy, prove the weight of grapes. -/
theorem weight_of_grapes (total weight_apples weight_oranges weight_strawberries : ℕ) 
  (h_total : total = 10)
  (h_apples : weight_apples = 3)
  (h_oranges : weight_oranges = 1)
  (h_strawberries : weight_strawberries = 3) :
  total - (weight_apples + weight_oranges + weight_strawberries) = 3 := by
  sorry

end weight_of_grapes_l1770_177021


namespace volleyball_tournament_equation_l1770_177008

/-- Represents a volleyball tournament. -/
structure VolleyballTournament where
  /-- The number of teams in the tournament. -/
  num_teams : ℕ
  /-- The total number of matches played. -/
  total_matches : ℕ
  /-- Each pair of teams plays against each other once. -/
  each_pair_plays_once : True

/-- Theorem stating the correct equation for the volleyball tournament. -/
theorem volleyball_tournament_equation (t : VolleyballTournament) 
  (h : t.total_matches = 28) : 
  (t.num_teams * (t.num_teams - 1)) / 2 = t.total_matches := by
  sorry

end volleyball_tournament_equation_l1770_177008


namespace point_on_x_axis_l1770_177013

theorem point_on_x_axis (a : ℝ) : 
  (∃ P : ℝ × ℝ, P.1 = a - 3 ∧ P.2 = 2 * a + 1 ∧ P.2 = 0) → a = -1/2 :=
by sorry

end point_on_x_axis_l1770_177013


namespace units_digit_37_pow_37_l1770_177044

theorem units_digit_37_pow_37 : 37^37 ≡ 7 [ZMOD 10] := by
  sorry

end units_digit_37_pow_37_l1770_177044


namespace cone_surface_area_ratio_l1770_177057

/-- For a cone whose lateral surface unfolds into a sector with a central angle of 90°,
    the ratio of the lateral surface area to the base area is 4. -/
theorem cone_surface_area_ratio (r : ℝ) (h : r > 0) : 
  let R := 4 * r
  let base_area := π * r^2
  let lateral_area := (1/4) * π * R^2
  lateral_area / base_area = 4 := by
sorry

end cone_surface_area_ratio_l1770_177057


namespace range_of_a_l1770_177022

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x - 3| + |x - a| > 5) ↔ 
  (a > 8 ∨ a < -2) := by
sorry

end range_of_a_l1770_177022


namespace equator_scientific_notation_l1770_177092

/-- The circumference of the equator in meters -/
def equator_circumference : ℕ := 40210000

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ

/-- Convert a natural number to scientific notation -/
def to_scientific_notation (n : ℕ) : ScientificNotation :=
  sorry

theorem equator_scientific_notation :
  to_scientific_notation equator_circumference = ScientificNotation.mk 4.021 7 := by
  sorry

end equator_scientific_notation_l1770_177092


namespace tangent_trapezoid_EQ_length_l1770_177006

/-- Represents a trapezoid with a circle tangent to two sides --/
structure TangentTrapezoid where
  EF : ℝ
  FG : ℝ
  GH : ℝ
  HE : ℝ
  EQ : ℝ
  QF : ℝ
  EQ_QF_ratio : EQ / QF = 5 / 3

/-- The theorem stating the length of EQ in the given trapezoid --/
theorem tangent_trapezoid_EQ_length (t : TangentTrapezoid) 
  (h1 : t.EF = 150)
  (h2 : t.FG = 65)
  (h3 : t.GH = 35)
  (h4 : t.HE = 90)
  (h5 : t.EF = t.EQ + t.QF) :
  t.EQ = 375 / 4 := by
  sorry

end tangent_trapezoid_EQ_length_l1770_177006


namespace initial_pencils_theorem_l1770_177098

/-- The number of pencils initially in the drawer -/
def initial_pencils : ℕ := 34

/-- The number of pencils Dan took from the drawer -/
def pencils_taken : ℕ := 22

/-- The number of pencils remaining in the drawer -/
def pencils_remaining : ℕ := 12

/-- Theorem: The initial number of pencils equals the sum of pencils taken and pencils remaining -/
theorem initial_pencils_theorem : initial_pencils = pencils_taken + pencils_remaining := by
  sorry

end initial_pencils_theorem_l1770_177098


namespace billy_tickets_l1770_177064

theorem billy_tickets (tickets_won : ℕ) (tickets_left : ℕ) (difference : ℕ) : 
  tickets_left = 32 →
  difference = 16 →
  tickets_won - tickets_left = difference →
  tickets_won = 48 := by
sorry

end billy_tickets_l1770_177064


namespace simplify_expression_l1770_177042

theorem simplify_expression (x y : ℝ) : 5 * x - (x - 2 * y) = 4 * x + 2 * y := by
  sorry

end simplify_expression_l1770_177042


namespace largest_m_for_cubic_quintic_inequality_l1770_177059

theorem largest_m_for_cubic_quintic_inequality :
  ∃ (m : ℝ), m = 9 ∧
  (∀ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 1 →
    10 * (a^3 + b^3 + c^3) - m * (a^5 + b^5 + c^5) ≥ 1) ∧
  (∀ (m' : ℝ), m' > m →
    ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 1 ∧
      10 * (a^3 + b^3 + c^3) - m' * (a^5 + b^5 + c^5) < 1) :=
by sorry

end largest_m_for_cubic_quintic_inequality_l1770_177059


namespace initial_members_family_b_l1770_177020

/-- Represents the number of members in each family in Indira Nagar -/
structure FamilyMembers where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ
  f : ℕ

/-- The theorem stating the initial number of members in family b -/
theorem initial_members_family_b (fm : FamilyMembers) : 
  fm.a = 7 ∧ fm.c = 10 ∧ fm.d = 13 ∧ fm.e = 6 ∧ fm.f = 10 ∧
  (fm.a + fm.b + fm.c + fm.d + fm.e + fm.f - 6) / 6 = 8 →
  fm.b = 8 := by
  sorry

#check initial_members_family_b

end initial_members_family_b_l1770_177020


namespace cost_price_of_article_l1770_177010

/-- The cost price of an article satisfying certain selling price conditions -/
theorem cost_price_of_article : ∃ C : ℝ, 
  (C = 400) ∧ 
  (0.8 * C = C - 0.2 * C) ∧ 
  (1.05 * C = C + 0.05 * C) ∧ 
  (1.05 * C - 0.8 * C = 100) := by
  sorry

end cost_price_of_article_l1770_177010


namespace min_radius_value_l1770_177075

/-- A circle in the Cartesian plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point on the circle satisfying the given condition -/
structure PointOnCircle (c : Circle) where
  point : ℝ × ℝ
  on_circle : (point.1 - c.center.1)^2 + (point.2 - c.center.2)^2 = c.radius^2
  condition : point.2^2 ≥ 4 * point.1

/-- The theorem stating the minimum value of r -/
theorem min_radius_value (c : Circle) (p : PointOnCircle c) :
  c.center.1 = c.radius + 1 ∧ c.center.2 = 0 → c.radius ≥ 4 := by
  sorry

end min_radius_value_l1770_177075


namespace fraction_numerator_is_twelve_l1770_177011

theorem fraction_numerator_is_twelve :
  ∀ (numerator : ℚ),
    (∃ (denominator : ℚ),
      denominator = 2 * numerator + 4 ∧
      numerator / denominator = 3 / 7) →
    numerator = 12 := by
  sorry

end fraction_numerator_is_twelve_l1770_177011


namespace negation_of_all_students_prepared_l1770_177043

variable (α : Type)
variable (student : α → Prop)
variable (prepared : α → Prop)

theorem negation_of_all_students_prepared :
  (¬ ∀ x, student x → prepared x) ↔ (∃ x, student x ∧ ¬ prepared x) :=
by sorry

end negation_of_all_students_prepared_l1770_177043


namespace cube_surface_area_l1770_177003

/-- The surface area of a cube, given the distance between non-intersecting diagonals of adjacent faces -/
theorem cube_surface_area (d : ℝ) (h : d = 8) : 
  let a := d * 3 / Real.sqrt 3
  6 * a^2 = 1152 := by sorry

end cube_surface_area_l1770_177003


namespace bus_stop_time_l1770_177039

/-- Proves that a bus with given speeds stops for 10 minutes per hour -/
theorem bus_stop_time (speed_without_stops : ℝ) (speed_with_stops : ℝ) 
  (h1 : speed_without_stops = 54) 
  (h2 : speed_with_stops = 45) : ℝ :=
by
  sorry

#check bus_stop_time

end bus_stop_time_l1770_177039


namespace characterize_function_l1770_177094

-- Define the function type
def RealFunction := ℝ → ℝ

-- State the theorem
theorem characterize_function (f : RealFunction) :
  (∀ x y : ℝ, f (f x ^ 2 + f y) = x * f x + y) →
  (∀ x : ℝ, f x = x ∨ f x = -x) :=
by sorry

end characterize_function_l1770_177094


namespace min_visible_sum_is_90_l1770_177071

/-- Represents a cube with integers on each face -/
structure SmallCube where
  faces : Fin 6 → ℕ

/-- Represents the larger 3x3x3 cube -/
structure LargeCube where
  smallCubes : Fin 27 → SmallCube

/-- Calculates the sum of visible faces on the larger cube -/
def visibleSum (c : LargeCube) : ℕ := sorry

/-- The minimum possible sum of visible faces -/
def minVisibleSum : ℕ := 90

/-- Theorem stating that the minimum possible sum is 90 -/
theorem min_visible_sum_is_90 :
  ∀ c : LargeCube, visibleSum c ≥ minVisibleSum :=
sorry

end min_visible_sum_is_90_l1770_177071


namespace cross_in_square_l1770_177069

theorem cross_in_square (s : ℝ) : 
  (2 * (s/2)^2 + 2 * (s/4)^2 = 810) → s = 36 := by
  sorry

end cross_in_square_l1770_177069


namespace quadratic_real_roots_condition_l1770_177086

theorem quadratic_real_roots_condition (m : ℝ) :
  (∃ x : ℝ, x^2 + 2*x + m = 0) ↔ m ≤ 1 := by sorry

end quadratic_real_roots_condition_l1770_177086


namespace simplify_expression_l1770_177047

theorem simplify_expression (w x : ℝ) :
  3*w + 6*w + 9*w + 12*w + 15*w + 20*x + 24 = 45*w + 20*x + 24 := by
  sorry

end simplify_expression_l1770_177047


namespace smallest_number_with_five_remainders_l1770_177058

theorem smallest_number_with_five_remainders (n : ℕ) : 
  (∃ (a b c d e : ℕ), a < b ∧ b < c ∧ c < d ∧ d < e ∧ e ≤ n ∧
    a % 11 = 3 ∧ b % 11 = 3 ∧ c % 11 = 3 ∧ d % 11 = 3 ∧ e % 11 = 3 ∧
    ∀ (x : ℕ), x ≤ n → x % 11 = 3 → (x = a ∨ x = b ∨ x = c ∨ x = d ∨ x = e)) ↔
  n = 47 :=
by sorry

end smallest_number_with_five_remainders_l1770_177058


namespace toy_car_production_l1770_177090

theorem toy_car_production (yesterday : ℕ) (today : ℕ) (total : ℕ) : 
  today = 2 * yesterday → 
  total = yesterday + today → 
  total = 180 → 
  yesterday = 60 :=
by sorry

end toy_car_production_l1770_177090


namespace prob_divisible_by_4_5_or_7_l1770_177012

/-- The probability of selecting a number from 1 to 200 that is divisible by 4, 5, or 7 -/
theorem prob_divisible_by_4_5_or_7 : 
  let S := Finset.range 200
  let divisible_by_4_5_or_7 := fun n => n % 4 = 0 ∨ n % 5 = 0 ∨ n % 7 = 0
  (S.filter divisible_by_4_5_or_7).card / S.card = 97 / 200 := by
  sorry

end prob_divisible_by_4_5_or_7_l1770_177012


namespace barge_power_increase_l1770_177084

/-- Given a barge pushed by tugboats in water, this theorem proves that
    doubling the force results in a power increase by a factor of 2√2,
    when water resistance is proportional to the square of speed. -/
theorem barge_power_increase
  (F : ℝ) -- Initial force
  (v : ℝ) -- Initial velocity
  (k : ℝ) -- Constant of proportionality for water resistance
  (h1 : F = k * v^2) -- Initial force equals water resistance
  (h2 : 2 * F = k * v_new^2) -- New force equals new water resistance
  : (2 * F * v_new) / (F * v) = 2 * Real.sqrt 2 :=
by sorry

end barge_power_increase_l1770_177084


namespace square_difference_l1770_177073

theorem square_difference (a b : ℕ) (h1 : b - a = 3) (h2 : b = 8) : b^2 - a^2 = 39 := by
  sorry

end square_difference_l1770_177073


namespace cubeTowerSurfaceArea_8_l1770_177056

/-- Calculates the surface area of a cube tower -/
def cubeTowerSurfaceArea (n : Nat) : Nat :=
  let sideAreas : Nat → Nat := fun i => 6 * i^2
  let bottomAreas : Nat → Nat := fun i => i^2
  let adjustedAreas : List Nat := (List.range n).map (fun i =>
    if i = 0 then sideAreas (i + 1)
    else sideAreas (i + 1) - bottomAreas (i + 1))
  adjustedAreas.sum

/-- The surface area of a tower of 8 cubes with side lengths 1 to 8 is 1021 -/
theorem cubeTowerSurfaceArea_8 :
  cubeTowerSurfaceArea 8 = 1021 := by
  sorry

end cubeTowerSurfaceArea_8_l1770_177056


namespace festival_groups_l1770_177017

theorem festival_groups (n : ℕ) (h : n = 7) : 
  (Nat.choose n 4 = 35) ∧ (Nat.choose n 3 = 35) := by
  sorry

#check festival_groups

end festival_groups_l1770_177017


namespace certain_number_problem_l1770_177016

theorem certain_number_problem : 
  ∃ x : ℝ, 0.60 * x = 0.42 * 30 + 17.4 ∧ x = 50 := by
  sorry

end certain_number_problem_l1770_177016


namespace square_root_problem_l1770_177046

theorem square_root_problem (x y z a : ℝ) : 
  (∃ (x : ℝ), x > 0 ∧ (a - 3)^2 = x ∧ (2*a + 15)^2 = x) →
  y = (-3)^3 →
  z = Int.floor (Real.sqrt 13) →
  Real.sqrt (x + y - 2*z) = 4 ∨ Real.sqrt (x + y - 2*z) = -4 := by
  sorry

#check square_root_problem

end square_root_problem_l1770_177046


namespace constant_term_expansion_l1770_177009

theorem constant_term_expansion (x : ℝ) : 
  ∃ (c : ℝ), (x + 1/x + 2)^4 = c + (terms_with_x : ℝ) ∧ c = 70 :=
by sorry

end constant_term_expansion_l1770_177009


namespace tan_equality_implies_sixty_degrees_l1770_177088

theorem tan_equality_implies_sixty_degrees (n : ℤ) :
  -90 < n ∧ n < 90 →
  Real.tan (n * π / 180) = Real.tan (240 * π / 180) →
  n = 60 := by
sorry

end tan_equality_implies_sixty_degrees_l1770_177088


namespace perpendicular_lines_parallel_perpendicular_line_contained_line_l1770_177083

-- Define the types for lines and planes
def Line : Type := sorry
def Plane : Type := sorry

-- Define the relationships between lines and planes
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def parallel (l1 l2 : Line) : Prop := sorry
def contains (p : Plane) (l : Line) : Prop := sorry

-- State the theorems
theorem perpendicular_lines_parallel (m n : Line) (α : Plane) :
  perpendicular m α → perpendicular n α → parallel m n := by sorry

theorem perpendicular_line_contained_line (m n : Line) (α : Plane) :
  perpendicular m α → contains α n → perpendicular m n := by sorry

end perpendicular_lines_parallel_perpendicular_line_contained_line_l1770_177083


namespace hockey_league_teams_l1770_177076

/-- The number of teams in a hockey league -/
def num_teams : ℕ := 17

/-- The number of times each team faces every other team -/
def games_per_pair : ℕ := 10

/-- The total number of games played in the season -/
def total_games : ℕ := 1360

/-- Theorem stating that the number of teams is correct given the conditions -/
theorem hockey_league_teams :
  (num_teams * (num_teams - 1) * games_per_pair) / 2 = total_games :=
sorry

end hockey_league_teams_l1770_177076


namespace xiang_lake_one_millionth_closest_to_study_room_l1770_177079

/-- The combined area of Phase I and Phase II of Xiang Lake in square kilometers -/
def xiang_lake_area : ℝ := 10.6

/-- One million as a real number -/
def one_million : ℝ := 1000000

/-- Conversion factor from square kilometers to square meters -/
def km2_to_m2 : ℝ := 1000000

/-- Approximate area of a typical study room in square meters -/
def typical_study_room_area : ℝ := 10.6

/-- Theorem stating that one-millionth of Xiang Lake's area is closest to a typical study room's area -/
theorem xiang_lake_one_millionth_closest_to_study_room :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ 
  |xiang_lake_area * km2_to_m2 / one_million - typical_study_room_area| < ε :=
sorry

end xiang_lake_one_millionth_closest_to_study_room_l1770_177079


namespace jerry_birthday_money_weighted_mean_l1770_177093

-- Define the exchange rates
def euro_to_usd : ℝ := 1.20
def gbp_to_usd : ℝ := 1.38

-- Define the weighted percentages
def family_weight : ℝ := 0.40
def friends_weight : ℝ := 0.60

-- Define the money received from family members in USD
def aunt_gift : ℝ := 9
def uncle_gift : ℝ := 9 * euro_to_usd
def sister_gift : ℝ := 7

-- Define the money received from friends in USD
def friends_gifts : List ℝ := [22, 23, 18 * euro_to_usd, 15 * gbp_to_usd, 22]

-- Calculate total family gifts
def family_total : ℝ := aunt_gift + uncle_gift + sister_gift

-- Calculate total friends gifts
def friends_total : ℝ := friends_gifts.sum

-- Define the weighted mean calculation
def weighted_mean : ℝ := family_total * family_weight + friends_total * friends_weight

-- Theorem to prove
theorem jerry_birthday_money_weighted_mean :
  weighted_mean = 76.30 := by sorry

end jerry_birthday_money_weighted_mean_l1770_177093


namespace males_not_interested_count_l1770_177018

/-- Represents the survey results for a yoga class -/
structure YogaSurvey where
  total_not_interested : ℕ
  females_not_interested : ℕ

/-- Calculates the number of males not interested in the yoga class -/
def males_not_interested (survey : YogaSurvey) : ℕ :=
  survey.total_not_interested - survey.females_not_interested

/-- Theorem stating that the number of males not interested is 110 -/
theorem males_not_interested_count (survey : YogaSurvey) 
  (h1 : survey.total_not_interested = 200)
  (h2 : survey.females_not_interested = 90) : 
  males_not_interested survey = 110 := by
  sorry

#eval males_not_interested ⟨200, 90⟩

end males_not_interested_count_l1770_177018


namespace price_after_decrease_l1770_177026

/-- The original price of an article given its reduced price after a percentage decrease -/
def original_price (reduced_price : ℚ) (decrease_percentage : ℚ) : ℚ :=
  reduced_price / (1 - decrease_percentage)

/-- Theorem stating that if an article's price after a 56% decrease is Rs. 4400, 
    then its original price was Rs. 10000 -/
theorem price_after_decrease (reduced_price : ℚ) (h : reduced_price = 4400) :
  original_price reduced_price (56/100) = 10000 := by
  sorry

end price_after_decrease_l1770_177026


namespace clock_angle_division_theorem_l1770_177023

/-- The time when the second hand divides the angle between hour and minute hands -/
def clock_division_time (n : ℕ) (k : ℚ) : ℚ :=
  (43200 * (1 + k) * n) / (719 + 708 * k)

/-- Theorem stating the time when the second hand divides the angle between hour and minute hands -/
theorem clock_angle_division_theorem (n : ℕ) (k : ℚ) :
  let t := clock_division_time n k
  let second_pos := t
  let minute_pos := t / 60
  let hour_pos := t / 720
  (second_pos - hour_pos) / (minute_pos - second_pos) = k ∧
  t = (43200 * (1 + k) * n) / (719 + 708 * k) := by
  sorry


end clock_angle_division_theorem_l1770_177023
