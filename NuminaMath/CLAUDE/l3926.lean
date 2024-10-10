import Mathlib

namespace binomial_18_choose_6_l3926_392698

theorem binomial_18_choose_6 : Nat.choose 18 6 = 4765 := by
  sorry

end binomial_18_choose_6_l3926_392698


namespace jumping_contest_total_distance_l3926_392685

/-- Represents the distance jumped by an animal and the obstacle they cleared -/
structure JumpDistance where
  jump : ℕ
  obstacle : ℕ

/-- Calculates the total distance jumped including the obstacle -/
def totalDistance (jd : JumpDistance) : ℕ := jd.jump + jd.obstacle

theorem jumping_contest_total_distance 
  (grasshopper : JumpDistance)
  (frog : JumpDistance)
  (kangaroo : JumpDistance)
  (h1 : grasshopper.jump = 25 ∧ grasshopper.obstacle = 5)
  (h2 : frog.jump = grasshopper.jump + 15 ∧ frog.obstacle = 10)
  (h3 : kangaroo.jump = 2 * frog.jump ∧ kangaroo.obstacle = 15) :
  totalDistance grasshopper + totalDistance frog + totalDistance kangaroo = 175 := by
  sorry

#check jumping_contest_total_distance

end jumping_contest_total_distance_l3926_392685


namespace total_height_climbed_l3926_392658

/-- The number of staircases John climbs -/
def num_staircases : ℕ := 3

/-- The number of steps in the first staircase -/
def first_staircase : ℕ := 20

/-- The number of steps in the second staircase -/
def second_staircase : ℕ := 2 * first_staircase

/-- The number of steps in the third staircase -/
def third_staircase : ℕ := second_staircase - 10

/-- The height of each step in feet -/
def step_height : ℚ := 1/2

/-- The total number of steps climbed -/
def total_steps : ℕ := first_staircase + second_staircase + third_staircase

/-- The total height climbed in feet -/
def total_feet : ℚ := (total_steps : ℚ) * step_height

theorem total_height_climbed : total_feet = 45 := by
  sorry

end total_height_climbed_l3926_392658


namespace stones_can_be_combined_l3926_392688

/-- Definition of similar sizes -/
def similar_sizes (x y : ℕ) : Prop := x ≤ y ∧ y ≤ 2 * x

/-- A step in the combining process -/
inductive combine_step (stones : List ℕ) : List ℕ → Prop
  | combine (x y : ℕ) (rest : List ℕ) :
      x ∈ stones →
      y ∈ stones →
      similar_sizes x y →
      combine_step stones ((x + y) :: (stones.filter (λ z ↦ z ≠ x ∧ z ≠ y)))

/-- The transitive closure of combine_step -/
def can_combine := Relation.ReflTransGen combine_step

/-- The main theorem -/
theorem stones_can_be_combined (initial_stones : List ℕ) :
  ∃ (final_pile : ℕ), can_combine initial_stones [final_pile] :=
sorry

end stones_can_be_combined_l3926_392688


namespace car_speed_increase_l3926_392618

/-- Proves that the percentage increase in car Y's average speed compared to car Q's speed is 50% -/
theorem car_speed_increase (distance : ℝ) (time_Q time_Y : ℝ) 
  (h1 : distance = 80)
  (h2 : time_Q = 2)
  (h3 : time_Y = 1.3333333333333333)
  (h4 : distance / time_Y > distance / time_Q) :
  (distance / time_Y - distance / time_Q) / (distance / time_Q) * 100 = 50 := by
  sorry

end car_speed_increase_l3926_392618


namespace incorrect_inequality_transformation_l3926_392678

theorem incorrect_inequality_transformation (a b : ℝ) (h : a < b) :
  ¬(3 - a < 3 - b) := by
  sorry

end incorrect_inequality_transformation_l3926_392678


namespace smallest_value_of_complex_sum_l3926_392662

theorem smallest_value_of_complex_sum (a b c d : ℤ) (ω : ℂ) 
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_omega_power : ω^4 = 1)
  (h_omega_neq_one : ω ≠ 1) :
  ∃ (m : ℝ), m = Real.sqrt (9/2) ∧
    ∀ (x : ℂ), x = a + b*ω + c*ω^2 + d*ω^3 → Complex.abs x ≥ m :=
  sorry

end smallest_value_of_complex_sum_l3926_392662


namespace newspaper_delivery_totals_l3926_392682

/-- Represents the days of the week --/
inductive Day
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents the different routes --/
inductive Route
  | Route1
  | Route2
  | Route3
  | Route4
  | Route5

/-- Represents the different newspapers --/
inductive Newspaper
  | A
  | B
  | C

/-- Defines the delivery schedule for each newspaper --/
def delivery_schedule (n : Newspaper) (d : Day) (r : Route) : Nat :=
  match n, d, r with
  | Newspaper.A, Day.Sunday, Route.Route1 => 90
  | Newspaper.A, Day.Sunday, Route.Route2 => 30
  | Newspaper.A, _, Route.Route1 => 100
  | Newspaper.B, Day.Tuesday, Route.Route3 => 80
  | Newspaper.B, Day.Thursday, Route.Route3 => 80
  | Newspaper.B, Day.Saturday, Route.Route3 => 50
  | Newspaper.B, Day.Saturday, Route.Route4 => 20
  | Newspaper.B, Day.Sunday, Route.Route3 => 50
  | Newspaper.B, Day.Sunday, Route.Route4 => 20
  | Newspaper.C, Day.Monday, Route.Route5 => 70
  | Newspaper.C, Day.Wednesday, Route.Route5 => 70
  | Newspaper.C, Day.Friday, Route.Route5 => 70
  | Newspaper.C, Day.Sunday, Route.Route5 => 15
  | Newspaper.C, Day.Sunday, Route.Route4 => 25
  | _, _, _ => 0

/-- Calculates the total newspapers delivered for a given newspaper in a week --/
def total_newspapers (n : Newspaper) : Nat :=
  List.sum (List.map (fun d => List.sum (List.map (fun r => delivery_schedule n d r) [Route.Route1, Route.Route2, Route.Route3, Route.Route4, Route.Route5]))
    [Day.Monday, Day.Tuesday, Day.Wednesday, Day.Thursday, Day.Friday, Day.Saturday, Day.Sunday])

/-- Theorem stating the correct total number of newspapers delivered for each type in a week --/
theorem newspaper_delivery_totals :
  (total_newspapers Newspaper.A = 720) ∧
  (total_newspapers Newspaper.B = 300) ∧
  (total_newspapers Newspaper.C = 250) := by
  sorry

end newspaper_delivery_totals_l3926_392682


namespace product_of_numbers_with_given_sum_and_difference_l3926_392617

theorem product_of_numbers_with_given_sum_and_difference :
  ∀ x y : ℝ, x + y = 60 ∧ x - y = 10 → x * y = 875 := by
  sorry

end product_of_numbers_with_given_sum_and_difference_l3926_392617


namespace perfect_squares_mod_seven_l3926_392646

theorem perfect_squares_mod_seven :
  ∃! (S : Finset ℕ), (∀ n ∈ S, ∃ m : ℤ, (m ^ 2 : ℤ) % 7 = n) ∧
                     (∀ k : ℤ, ∃ n ∈ S, (k ^ 2 : ℤ) % 7 = n) ∧
                     S.card = 4 :=
by sorry

end perfect_squares_mod_seven_l3926_392646


namespace prob_select_copresidents_value_l3926_392627

/-- Represents a math club with a given number of students -/
structure MathClub where
  students : ℕ
  co_presidents : Fin 2
  vice_president : Fin 1

/-- The set of math clubs in the school district -/
def school_clubs : Finset MathClub := sorry

/-- The probability of selecting both co-presidents when randomly selecting 
    three members from a randomly selected club -/
def prob_select_copresidents (clubs : Finset MathClub) : ℚ := sorry

theorem prob_select_copresidents_value : 
  prob_select_copresidents school_clubs = 43 / 420 := by sorry

end prob_select_copresidents_value_l3926_392627


namespace greatest_number_in_set_l3926_392661

/-- Given a set of 45 consecutive multiples of 5 starting from 55, 
    the greatest number in the set is 275. -/
theorem greatest_number_in_set (s : Set ℕ) 
  (h1 : ∀ n ∈ s, ∃ k, n = 5 * k) 
  (h2 : ∀ n ∈ s, 55 ≤ n ∧ n ≤ 275)
  (h3 : ∀ n, 55 ≤ n ∧ n ≤ 275 ∧ 5 ∣ n → n ∈ s)
  (h4 : 55 ∈ s)
  (h5 : Fintype s)
  (h6 : Fintype.card s = 45) : 
  275 ∈ s ∧ ∀ n ∈ s, n ≤ 275 := by
  sorry

end greatest_number_in_set_l3926_392661


namespace integer_decimal_parts_sqrt10_l3926_392691

theorem integer_decimal_parts_sqrt10 (a b : ℝ) : 
  (a = ⌊6 - Real.sqrt 10⌋) → 
  (b = 6 - Real.sqrt 10 - a) → 
  (2 * a + Real.sqrt 10) * b = 6 := by
sorry

end integer_decimal_parts_sqrt10_l3926_392691


namespace backup_settings_count_l3926_392637

/-- Represents the weight of a single piece of silverware in ounces -/
def silverware_weight : ℕ := 4

/-- Represents the number of silverware pieces per setting -/
def silverware_per_setting : ℕ := 3

/-- Represents the weight of a single plate in ounces -/
def plate_weight : ℕ := 12

/-- Represents the number of plates per setting -/
def plates_per_setting : ℕ := 2

/-- Represents the number of tables -/
def num_tables : ℕ := 15

/-- Represents the number of settings per table -/
def settings_per_table : ℕ := 8

/-- Represents the total weight of all settings including backups in ounces -/
def total_weight : ℕ := 5040

/-- Calculates the number of backup settings needed -/
def backup_settings : ℕ := 
  let setting_weight := silverware_weight * silverware_per_setting + plate_weight * plates_per_setting
  let total_settings := num_tables * settings_per_table
  let regular_settings_weight := total_settings * setting_weight
  (total_weight - regular_settings_weight) / setting_weight

theorem backup_settings_count : backup_settings = 20 := by
  sorry

end backup_settings_count_l3926_392637


namespace art_collection_area_is_282_l3926_392695

/-- Calculates the total area of Davonte's art collection -/
def art_collection_area : ℕ :=
  let square_painting_area := 3 * (6 * 6)
  let small_painting_area := 4 * (2 * 3)
  let large_painting_area := 10 * 15
  square_painting_area + small_painting_area + large_painting_area

/-- Proves that the total area of Davonte's art collection is 282 square feet -/
theorem art_collection_area_is_282 : art_collection_area = 282 := by
  sorry

end art_collection_area_is_282_l3926_392695


namespace arithmetic_geometric_harmonic_means_l3926_392633

theorem arithmetic_geometric_harmonic_means (a b c : ℝ) :
  (a + b + c) / 3 = 9 →
  (a * b * c) ^ (1/3 : ℝ) = 6 →
  3 / (1/a + 1/b + 1/c) = 4 →
  a^2 + b^2 + c^2 = 405 := by
sorry

end arithmetic_geometric_harmonic_means_l3926_392633


namespace circles_externally_tangent_l3926_392670

/-- 
Given a line ax + by + 1 = 0 where the distance from the origin to this line is 1/2,
prove that the circles (x - a)² + y² = 1 and x² + (y - b)² = 1 are externally tangent.
-/
theorem circles_externally_tangent (a b : ℝ) 
  (h : (a^2 + b^2)⁻¹ = 1/4) : 
  let d := Real.sqrt (a^2 + b^2)
  d = 2 := by sorry

end circles_externally_tangent_l3926_392670


namespace root_range_theorem_l3926_392652

def f (m : ℝ) (x : ℝ) : ℝ := x^2 + (m-1)*x + m^2 - 2

theorem root_range_theorem (m : ℝ) : 
  (∃ x y : ℝ, x < -1 ∧ y > 1 ∧ 
   f m x = 0 ∧ f m y = 0 ∧ 
   ∀ z : ℝ, f m z = 0 → z = x ∨ z = y) ↔ 
  m > 0 ∧ m < 1 := by sorry

end root_range_theorem_l3926_392652


namespace green_space_equation_l3926_392645

/-- Represents a rectangular green space -/
structure GreenSpace where
  length : ℝ
  width : ℝ
  area : ℝ

/-- Theorem stating the properties of the green space and the resulting equation -/
theorem green_space_equation (g : GreenSpace) 
  (h1 : g.area = 1000)
  (h2 : g.length = g.width + 30)
  (h3 : g.area = g.length * g.width) :
  g.length * (g.length - 30) = 1000 := by
  sorry

#check green_space_equation

end green_space_equation_l3926_392645


namespace f_max_min_on_interval_l3926_392686

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 5

-- Define the interval [0, 3]
def interval : Set ℝ := { x | 0 ≤ x ∧ x ≤ 3 }

-- Theorem statement
theorem f_max_min_on_interval :
  (∃ x ∈ interval, ∀ y ∈ interval, f y ≤ f x) ∧
  (∃ x ∈ interval, ∀ y ∈ interval, f x ≤ f y) ∧
  (∀ x ∈ interval, f x ≤ 5) ∧
  (∀ x ∈ interval, -15 ≤ f x) ∧
  (∃ x ∈ interval, f x = 5) ∧
  (∃ x ∈ interval, f x = -15) :=
by sorry

end f_max_min_on_interval_l3926_392686


namespace intersection_A_complement_B_l3926_392649

-- Define the sets A and B
def A : Set ℝ := {x | x ≤ 3}
def B : Set ℝ := {x | x < 2}

-- Define the complement of B in ℝ
def complement_B : Set ℝ := {x | x ≥ 2}

-- State the theorem
theorem intersection_A_complement_B :
  A ∩ complement_B = Set.Icc 2 3 := by sorry

end intersection_A_complement_B_l3926_392649


namespace juan_running_time_l3926_392679

/-- Given that Juan ran at a speed of 10.0 miles per hour and covered a distance of 800 miles,
    prove that the time he ran equals 80 hours. -/
theorem juan_running_time (speed : ℝ) (distance : ℝ) (h1 : speed = 10.0) (h2 : distance = 800) :
  distance / speed = 80 :=
by sorry

end juan_running_time_l3926_392679


namespace sum_of_xyz_l3926_392615

theorem sum_of_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hxy : x * y = 30) (hxz : x * z = 60) (hyz : y * z = 90) :
  x + y + z = 11 * Real.sqrt 5 := by
sorry

end sum_of_xyz_l3926_392615


namespace equation_solution_l3926_392696

theorem equation_solution : ∃ x : ℝ, x ≠ 0 ∧ (2 / x + (3 / x) / (6 / x) = 1.5) ∧ x = 2 := by
  sorry

end equation_solution_l3926_392696


namespace quadratic_root_equivalence_l3926_392693

theorem quadratic_root_equivalence (a b c : ℝ) (ha : a ≠ 0) :
  (a + b + c = 0) ↔ (a * 1^2 + b * 1 + c = 0) :=
sorry

end quadratic_root_equivalence_l3926_392693


namespace clock_strike_time_l3926_392630

theorem clock_strike_time (strike_three : ℕ) (time_three : ℝ) (strike_six : ℕ) : 
  strike_three = 3 → time_three = 12 → strike_six = 6 → 
  ∃ (time_six : ℝ), time_six = 30 := by
  sorry

end clock_strike_time_l3926_392630


namespace water_bucket_problem_l3926_392660

theorem water_bucket_problem (initial_amount : ℝ) (added_amount : ℝ) :
  initial_amount = 3 →
  added_amount = 6.8 →
  initial_amount + added_amount = 9.8 :=
by
  sorry

end water_bucket_problem_l3926_392660


namespace new_cube_weight_l3926_392643

/-- Given a cube of weight 3 pounds and density D, prove that a new cube with sides twice as long
    and density 1.25D will weigh 30 pounds. -/
theorem new_cube_weight (D : ℝ) (D_pos : D > 0) : 
  let original_weight : ℝ := 3
  let original_volume : ℝ := original_weight / D
  let new_volume : ℝ := 8 * original_volume
  let new_density : ℝ := 1.25 * D
  new_density * new_volume = 30 := by
  sorry


end new_cube_weight_l3926_392643


namespace triangle_area_l3926_392673

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop := sorry

-- Define the height of the triangle
def Height (A B C H : ℝ × ℝ) (h : ℝ) : Prop := sorry

-- Define the angles of the triangle
def Angle (A B C : ℝ × ℝ) (α : ℝ) : Prop := sorry

-- Define the area of a triangle
def TriangleArea (A B C : ℝ × ℝ) : ℝ := sorry

theorem triangle_area (A B C H : ℝ × ℝ) (h α γ : ℝ) :
  Triangle A B C →
  Height A B C H h →
  Angle B A C α →
  Angle B C A γ →
  TriangleArea A B C = (h^2 * Real.sin α) / (2 * Real.sin γ * Real.sin (α + γ)) :=
by sorry

end triangle_area_l3926_392673


namespace polyhedron_with_specific_projections_l3926_392692

/-- A polyhedron is a three-dimensional geometric object with flat polygonal faces, straight edges and sharp corners or vertices. -/
def Polyhedron : Type := sorry

/-- A plane in three-dimensional space. -/
def Plane : Type := sorry

/-- A projection of a polyhedron onto a plane. -/
def projection (p : Polyhedron) (plane : Plane) : Set (ℝ × ℝ) := sorry

/-- A triangle is a polygon with three sides. -/
def isTriangle (s : Set (ℝ × ℝ)) : Prop := sorry

/-- A quadrilateral is a polygon with four sides. -/
def isQuadrilateral (s : Set (ℝ × ℝ)) : Prop := sorry

/-- A pentagon is a polygon with five sides. -/
def isPentagon (s : Set (ℝ × ℝ)) : Prop := sorry

/-- Two planes are perpendicular if they intersect at a right angle. -/
def arePerpendicular (p1 p2 : Plane) : Prop := sorry

theorem polyhedron_with_specific_projections :
  ∃ (p : Polyhedron) (p1 p2 p3 : Plane),
    arePerpendicular p1 p2 ∧
    arePerpendicular p2 p3 ∧
    arePerpendicular p3 p1 ∧
    isTriangle (projection p p1) ∧
    isQuadrilateral (projection p p2) ∧
    isPentagon (projection p p3) := by
  sorry

end polyhedron_with_specific_projections_l3926_392692


namespace rohans_salary_l3926_392687

/-- Rohan's monthly salary in Rupees -/
def monthly_salary : ℝ := 10000

/-- Percentage of salary spent on food -/
def food_percentage : ℝ := 40

/-- Percentage of salary spent on house rent -/
def rent_percentage : ℝ := 20

/-- Percentage of salary spent on entertainment -/
def entertainment_percentage : ℝ := 10

/-- Percentage of salary spent on conveyance -/
def conveyance_percentage : ℝ := 10

/-- Rohan's savings at the end of the month in Rupees -/
def savings : ℝ := 2000

theorem rohans_salary :
  monthly_salary * (1 - (food_percentage + rent_percentage + entertainment_percentage + conveyance_percentage) / 100) = savings := by
  sorry

#check rohans_salary

end rohans_salary_l3926_392687


namespace paint_cans_used_l3926_392690

/-- Given:
  - Paul originally had enough paint for 50 rooms.
  - He lost 5 cans of paint.
  - After losing the paint, he had enough for 40 rooms.
Prove that the number of cans of paint used for 40 rooms is 20. -/
theorem paint_cans_used (original_rooms : ℕ) (lost_cans : ℕ) (remaining_rooms : ℕ)
  (h1 : original_rooms = 50)
  (h2 : lost_cans = 5)
  (h3 : remaining_rooms = 40) :
  (remaining_rooms : ℚ) / ((original_rooms - remaining_rooms : ℕ) / lost_cans : ℚ) = 20 := by
  sorry

end paint_cans_used_l3926_392690


namespace special_line_properties_l3926_392620

/-- A line passing through (5, 2) with y-intercept twice its x-intercept -/
def special_line (x y : ℝ) : Prop :=
  2 * x - 5 * y + 60 = 0

theorem special_line_properties :
  (special_line 5 2) ∧ 
  (∃ (b : ℝ), special_line 0 b ∧ special_line (b/2) 0 ∧ b ≠ 0) :=
by sorry

end special_line_properties_l3926_392620


namespace car_overtake_distance_l3926_392622

theorem car_overtake_distance (speed_a speed_b time_to_overtake distance_ahead : ℝ) 
  (h1 : speed_a = 58)
  (h2 : speed_b = 50)
  (h3 : time_to_overtake = 4)
  (h4 : distance_ahead = 8) :
  speed_a * time_to_overtake - speed_b * time_to_overtake - distance_ahead = 40 := by
  sorry

end car_overtake_distance_l3926_392622


namespace defeat_points_is_zero_l3926_392600

/-- Represents the point system for a football competition -/
structure PointSystem where
  victory_points : ℕ
  draw_points : ℕ
  defeat_points : ℕ

/-- Represents the state of a team's performance -/
structure TeamPerformance where
  total_matches : ℕ
  matches_played : ℕ
  points : ℕ
  victories : ℕ
  draws : ℕ
  defeats : ℕ

/-- Theorem stating that the number of points for a defeat must be 0 -/
theorem defeat_points_is_zero 
  (ps : PointSystem) 
  (tp : TeamPerformance) 
  (h1 : ps.victory_points = 3)
  (h2 : ps.draw_points = 1)
  (h3 : tp.total_matches = 20)
  (h4 : tp.matches_played = 5)
  (h5 : tp.points = 8)
  (h6 : ∀ (future_victories : ℕ), 
        future_victories ≥ 9 → 
        tp.points + future_victories * ps.victory_points + 
        (tp.total_matches - tp.matches_played - future_victories) * ps.defeat_points ≥ 40) :
  ps.defeat_points = 0 := by
sorry

end defeat_points_is_zero_l3926_392600


namespace cost_of_3000_pencils_l3926_392605

def pencil_cost (quantity : ℕ) : ℚ :=
  let base_price := 36 / 120
  let discount_threshold := 2000
  let discount_factor := 0.9
  if quantity > discount_threshold
  then (quantity : ℚ) * base_price * discount_factor
  else (quantity : ℚ) * base_price

theorem cost_of_3000_pencils :
  pencil_cost 3000 = 810 := by sorry

end cost_of_3000_pencils_l3926_392605


namespace jana_travel_distance_l3926_392663

/-- Calculates the total distance traveled by Jana given her walking and cycling rates and times. -/
theorem jana_travel_distance (walking_rate : ℝ) (walking_time : ℝ) (cycling_rate : ℝ) (cycling_time : ℝ) :
  walking_rate = 1 / 30 →
  walking_time = 45 →
  cycling_rate = 2 / 15 →
  cycling_time = 30 →
  walking_rate * walking_time + cycling_rate * cycling_time = 5.5 :=
by
  sorry

end jana_travel_distance_l3926_392663


namespace coin_problem_l3926_392684

/-- Represents the number of different coin values that can be made -/
def different_values (x y : ℕ) : ℕ := 29 - (3 * x + 2 * y) / 2

/-- The coin problem -/
theorem coin_problem (total : ℕ) (values : ℕ) :
  total = 12 ∧ values = 21 →
  ∃ x y : ℕ, x + y = total ∧ different_values x y = values ∧ y = 7 := by
  sorry

end coin_problem_l3926_392684


namespace farm_animals_l3926_392666

theorem farm_animals (goats chickens ducks pigs : ℕ) : 
  goats = 66 →
  chickens = 2 * goats →
  ducks = (goats + chickens) / 2 →
  pigs = ducks / 3 →
  goats - pigs = 33 := by
  sorry

end farm_animals_l3926_392666


namespace quadratic_equation_relation_l3926_392647

theorem quadratic_equation_relation (x : ℝ) : 
  x^2 + 3*x + 5 = 7 → x^2 + 3*x - 2 = 0 := by
  sorry

end quadratic_equation_relation_l3926_392647


namespace complex_magnitude_equation_l3926_392657

theorem complex_magnitude_equation (t : ℝ) (h : t > 0) :
  Complex.abs (-6 + t * Complex.I) = 3 * Real.sqrt 10 → t = 3 * Real.sqrt 6 := by
  sorry

end complex_magnitude_equation_l3926_392657


namespace simplify_fraction_division_l3926_392626

theorem simplify_fraction_division (x : ℝ) 
  (h1 : x ≠ 3) (h2 : x ≠ 2) : 
  (x^2 - 4*x + 3) / (x^2 - 6*x + 9) / ((x^2 - 3*x + 2) / (x^2 - 4*x + 4)) = (x - 2) / (x - 3) :=
by sorry

end simplify_fraction_division_l3926_392626


namespace parabola_line_intersection_l3926_392654

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola of the form y^2 = 2px -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Given a parabola y^2 = 8x and a line passing through P(1, -1) intersecting 
    the parabola at points A and B, where P is the midpoint of AB, 
    prove that the equation of line AB is 4x + y - 3 = 0 -/
theorem parabola_line_intersection 
  (para : Parabola) 
  (P : Point) 
  (A B : Point) 
  (line : Line) : 
  para.p = 4 → 
  P.x = 1 → 
  P.y = -1 → 
  (A.x + B.x) / 2 = P.x → 
  (A.y + B.y) / 2 = P.y → 
  A.y^2 = 8 * A.x → 
  B.y^2 = 8 * B.x → 
  line.a * A.x + line.b * A.y + line.c = 0 → 
  line.a * B.x + line.b * B.y + line.c = 0 → 
  line.a = 4 ∧ line.b = 1 ∧ line.c = -3 := by 
sorry

end parabola_line_intersection_l3926_392654


namespace smallest_scalene_perimeter_l3926_392680

-- Define a scalene triangle with integer side lengths
def ScaleneTriangle (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  a + b > c ∧ b + c > a ∧ a + c > b

-- Theorem statement
theorem smallest_scalene_perimeter :
  ∀ a b c : ℕ, ScaleneTriangle a b c → a + b + c ≥ 9 :=
by sorry

end smallest_scalene_perimeter_l3926_392680


namespace license_plate_combinations_eq_960_l3926_392653

/-- Represents the set of possible characters for each position in the license plate --/
def LicensePlateChoices : Fin 5 → Finset Char :=
  fun i => match i with
    | 0 => {'3', '5', '6', '8', '9'}
    | 1 => {'B', 'C', 'D'}
    | _ => {'1', '3', '6', '9'}

/-- The number of possible license plate combinations --/
def LicensePlateCombinations : ℕ :=
  (LicensePlateChoices 0).card *
  (LicensePlateChoices 1).card *
  (LicensePlateChoices 2).card *
  (LicensePlateChoices 3).card *
  (LicensePlateChoices 4).card

/-- Theorem stating that the number of possible license plate combinations is 960 --/
theorem license_plate_combinations_eq_960 :
  LicensePlateCombinations = 960 := by
  sorry

end license_plate_combinations_eq_960_l3926_392653


namespace hyperbola_eccentricity_l3926_392689

/-- Given a hyperbola and a circle with specific properties, prove the eccentricity of the hyperbola -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let hyperbola := {(x, y) : ℝ × ℝ | x^2 / a^2 - y^2 / b^2 = 1}
  let circle := {(x, y) : ℝ × ℝ | (x - 1)^2 + y^2 = 1}
  let asymptote := {(x, y) : ℝ × ℝ | y = (b / a) * x}
  let chord_length := Real.sqrt 3
  (∃ (p q : ℝ × ℝ), p ∈ asymptote ∧ q ∈ asymptote ∧ p ∈ circle ∧ q ∈ circle ∧
    Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = chord_length) →
  let c := Real.sqrt (a^2 + b^2)
  c / a = 2 / 3 * Real.sqrt 3 :=
by sorry

end hyperbola_eccentricity_l3926_392689


namespace xy_equal_three_l3926_392604

theorem xy_equal_three (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x ≠ y)
  (h : x + 3 / x = y + 3 / y) : x * y = 3 := by
  sorry

end xy_equal_three_l3926_392604


namespace average_marks_math_chem_l3926_392650

theorem average_marks_math_chem (math physics chem : ℕ) : 
  math + physics = 60 →
  chem = physics + 20 →
  (math + chem) / 2 = 40 :=
by sorry

end average_marks_math_chem_l3926_392650


namespace factor_x_squared_minus_169_l3926_392628

theorem factor_x_squared_minus_169 (x : ℝ) : x^2 - 169 = (x - 13) * (x + 13) := by
  sorry

end factor_x_squared_minus_169_l3926_392628


namespace circle_perimeter_l3926_392603

theorem circle_perimeter (r : ℝ) (h : r = 4 / Real.pi) : 
  2 * Real.pi * r = 8 := by sorry

end circle_perimeter_l3926_392603


namespace rectangle_area_l3926_392641

theorem rectangle_area (r : ℝ) (ratio : ℝ) : r = 6 ∧ ratio = 3 →
  ∃ (length width : ℝ),
    width = 2 * r ∧
    length = ratio * width ∧
    length * width = 432 := by
  sorry

end rectangle_area_l3926_392641


namespace tournament_handshakes_eq_24_l3926_392697

/-- The number of handshakes in a tournament with 4 teams of 2 players each -/
def tournament_handshakes : ℕ :=
  let num_teams : ℕ := 4
  let players_per_team : ℕ := 2
  let total_players : ℕ := num_teams * players_per_team
  let handshakes_per_player : ℕ := total_players - players_per_team
  (total_players * handshakes_per_player) / 2

theorem tournament_handshakes_eq_24 : tournament_handshakes = 24 := by
  sorry

end tournament_handshakes_eq_24_l3926_392697


namespace mean_problem_l3926_392675

theorem mean_problem (x : ℝ) : 
  (12 + x + 42 + 78 + 104) / 5 = 62 → 
  (128 + 255 + 511 + 1023 + x) / 5 = 413 := by
sorry

end mean_problem_l3926_392675


namespace min_value_theorem_l3926_392636

theorem min_value_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a * b + a * c = 4) :
  (2 / a) + (2 / (b + c)) + (8 / (a + b + c)) ≥ 4 ∧
  ((2 / a) + (2 / (b + c)) + (8 / (a + b + c)) = 4 ↔ a = 2 ∧ b + c = 2) :=
by sorry

end min_value_theorem_l3926_392636


namespace sin_cos_sum_eighty_forty_l3926_392668

theorem sin_cos_sum_eighty_forty : 
  Real.sin (80 * π / 180) * Real.cos (40 * π / 180) + 
  Real.cos (80 * π / 180) * Real.sin (40 * π / 180) = 
  Real.sqrt 3 / 2 := by sorry

end sin_cos_sum_eighty_forty_l3926_392668


namespace system_solution_l3926_392664

theorem system_solution (x y : ℝ) : 
  (1 / x + 1 / y = 2.25 ∧ x^2 / y + y^2 / x = 32.0625) ↔ 
  ((x = 4 ∧ y = 1/2) ∨ 
   (x = 1/12 * (-19 + Real.sqrt (1691/3)) ∧ 
    y = 1/12 * (-19 - Real.sqrt (1691/3)))) :=
by sorry

end system_solution_l3926_392664


namespace no_valid_rectangle_l3926_392623

theorem no_valid_rectangle (a b x y : ℝ) : 
  a < b → 
  x < a → 
  y < a → 
  2 * (x + y) = (2/3) * (a + b) → 
  x * y = (1/3) * a * b → 
  False := by
sorry

end no_valid_rectangle_l3926_392623


namespace grapes_purchased_l3926_392624

theorem grapes_purchased (grape_price : ℕ) (mango_quantity : ℕ) (mango_price : ℕ) (total_paid : ℕ) :
  grape_price = 70 →
  mango_quantity = 9 →
  mango_price = 55 →
  total_paid = 705 →
  ∃ grape_quantity : ℕ, grape_quantity * grape_price + mango_quantity * mango_price = total_paid ∧ grape_quantity = 3 :=
by
  sorry

end grapes_purchased_l3926_392624


namespace jellybean_probability_l3926_392610

def total_jellybeans : ℕ := 15
def red_jellybeans : ℕ := 5
def blue_jellybeans : ℕ := 3
def white_jellybeans : ℕ := 5
def green_jellybeans : ℕ := 2
def picked_jellybeans : ℕ := 4

theorem jellybean_probability : 
  (Nat.choose red_jellybeans 3 * Nat.choose (total_jellybeans - red_jellybeans) 1) / 
  Nat.choose total_jellybeans picked_jellybeans = 20 / 273 :=
by sorry

end jellybean_probability_l3926_392610


namespace negation_of_existence_negation_of_quadratic_inequality_l3926_392611

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x : ℝ, P x) ↔ (∀ x : ℝ, ¬ P x) := by sorry

theorem negation_of_quadratic_inequality :
  (¬ ∃ x : ℝ, x^2 - 2*x + 1 ≤ 0) ↔ (∀ x : ℝ, x^2 - 2*x + 1 > 0) := by sorry

end negation_of_existence_negation_of_quadratic_inequality_l3926_392611


namespace product_of_quarters_l3926_392635

theorem product_of_quarters : (0.25 : ℝ) * 0.75 = 0.1875 := by
  sorry

end product_of_quarters_l3926_392635


namespace grocery_theorem_l3926_392648

def grocery_problem (initial_budget : ℚ) (bread_cost : ℚ) (candy_cost : ℚ) (final_remaining : ℚ) : Prop :=
  let remaining_after_bread_candy := initial_budget - (bread_cost + candy_cost)
  let spent_on_turkey := remaining_after_bread_candy - final_remaining
  spent_on_turkey / remaining_after_bread_candy = 1 / 3

theorem grocery_theorem :
  grocery_problem 32 3 2 18 := by
  sorry

end grocery_theorem_l3926_392648


namespace bubble_gum_count_l3926_392683

theorem bubble_gum_count (total_cost : ℕ) (cost_per_piece : ℕ) (h1 : total_cost = 2448) (h2 : cost_per_piece = 18) :
  total_cost / cost_per_piece = 136 := by
  sorry

end bubble_gum_count_l3926_392683


namespace fermatville_temperature_range_l3926_392656

/-- The temperature range in Fermatville on Monday -/
def temperature_range (min_temp max_temp : Int) : Int :=
  max_temp - min_temp

/-- Theorem: The temperature range in Fermatville on Monday was 25°C -/
theorem fermatville_temperature_range :
  let min_temp : Int := -11
  let max_temp : Int := 14
  temperature_range min_temp max_temp = 25 := by
  sorry

end fermatville_temperature_range_l3926_392656


namespace total_hike_length_l3926_392667

/-- Represents Ella's hike over three days -/
structure HikeData where
  day1 : ℝ
  day2 : ℝ
  day3 : ℝ

/-- Conditions of Ella's hike -/
def isValidHike (h : HikeData) : Prop :=
  h.day1 + h.day2 = 18 ∧
  (h.day1 + h.day3) / 2 = 12 ∧
  h.day2 + h.day3 = 24 ∧
  h.day2 + h.day3 = 20

/-- Theorem stating the total length of the trail -/
theorem total_hike_length (h : HikeData) (hValid : isValidHike h) :
  h.day1 + h.day2 + h.day3 = 31 := by
  sorry

end total_hike_length_l3926_392667


namespace partner_p_investment_time_l3926_392642

/-- Represents the investment and profit data for two partners -/
structure PartnershipData where
  investment_ratio_p : ℚ
  investment_ratio_q : ℚ
  profit_ratio_p : ℚ
  profit_ratio_q : ℚ
  investment_time_q : ℚ

/-- Calculates the investment time for partner p given the partnership data -/
def calculate_investment_time_p (data : PartnershipData) : ℚ :=
  (data.investment_ratio_q * data.profit_ratio_p * data.investment_time_q) /
  (data.investment_ratio_p * data.profit_ratio_q)

/-- Theorem stating that given the specific partnership data, partner p's investment time is 5 months -/
theorem partner_p_investment_time :
  let data : PartnershipData := {
    investment_ratio_p := 7,
    investment_ratio_q := 5,
    profit_ratio_p := 7,
    profit_ratio_q := 12,
    investment_time_q := 12
  }
  calculate_investment_time_p data = 5 := by sorry

end partner_p_investment_time_l3926_392642


namespace intersection_and_complement_intersection_l3926_392631

def I : Set ℕ := Set.univ

def A : Set ℕ := {x | ∃ n : ℕ, x = 3 * n ∧ n % 2 = 0}

def B : Set ℕ := {y | 24 % y = 0}

theorem intersection_and_complement_intersection :
  (A ∩ B = {6, 12, 24}) ∧
  ((I \ A) ∩ B = {1, 2, 3, 4, 8}) := by sorry

end intersection_and_complement_intersection_l3926_392631


namespace pentagon_area_l3926_392655

/-- The area of a pentagon with sides 18, 25, 30, 28, and 25 units is 1020 square units. -/
theorem pentagon_area : ℝ := by
  -- Define the pentagon
  let side1 : ℝ := 18
  let side2 : ℝ := 25
  let side3 : ℝ := 30
  let side4 : ℝ := 28
  let side5 : ℝ := 25

  -- Define the area of the pentagon
  let pentagon_area : ℝ := 1020

  -- The proof goes here
  sorry

end pentagon_area_l3926_392655


namespace last_car_probability_2012_l3926_392674

/-- Represents the parking procedure for a given number of spots. -/
def ParkingProcedure (n : ℕ) : Type :=
  Unit

/-- Calculates the probability of the last car parking in spot 1 given the parking procedure. -/
noncomputable def lastCarProbability (n : ℕ) (proc : ParkingProcedure n) : ℚ :=
  sorry

/-- The theorem stating the probability of the last car parking in spot 1 for 2012 spots. -/
theorem last_car_probability_2012 :
  ∃ (proc : ParkingProcedure 2012), lastCarProbability 2012 proc = 1 / 2062300 :=
by
  sorry

end last_car_probability_2012_l3926_392674


namespace james_work_hours_james_work_hours_proof_l3926_392608

/-- Calculates the number of hours James needs to work to pay for food waste and janitorial costs -/
theorem james_work_hours (james_wage : ℝ) (meat_cost meat_wasted : ℝ) 
  (fruit_veg_cost fruit_veg_wasted : ℝ) (bread_cost bread_wasted : ℝ)
  (janitor_wage janitor_hours : ℝ) : ℝ :=
  let total_cost := meat_cost * meat_wasted + fruit_veg_cost * fruit_veg_wasted + 
                    bread_cost * bread_wasted + janitor_wage * 1.5 * janitor_hours
  total_cost / james_wage

/-- Proves that James needs to work 50 hours given the specific conditions -/
theorem james_work_hours_proof : 
  james_work_hours 8 5 20 4 15 1.5 60 10 10 = 50 := by
  sorry

end james_work_hours_james_work_hours_proof_l3926_392608


namespace joes_height_l3926_392644

theorem joes_height (sara_height joe_height : ℕ) : 
  sara_height + joe_height = 120 →
  joe_height = 2 * sara_height + 6 →
  joe_height = 82 :=
by
  sorry

end joes_height_l3926_392644


namespace average_weight_of_twenty_boys_l3926_392629

theorem average_weight_of_twenty_boys 
  (num_group1 : ℕ) 
  (num_group2 : ℕ) 
  (avg_weight_group2 : ℝ) 
  (avg_weight_all : ℝ) :
  num_group1 = 20 →
  num_group2 = 8 →
  avg_weight_group2 = 45.15 →
  avg_weight_all = 48.792857142857144 →
  (num_group1 * 50.25 + num_group2 * avg_weight_group2) / (num_group1 + num_group2) = avg_weight_all :=
by sorry

end average_weight_of_twenty_boys_l3926_392629


namespace regular_hexagon_area_l3926_392601

/-- The area of a regular hexagon with vertices A(0,0) and C(6,2) is 20√3 -/
theorem regular_hexagon_area : 
  let A : ℝ × ℝ := (0, 0)
  let C : ℝ × ℝ := (6, 2)
  let AC : ℝ := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)
  let s : ℝ := AC / 2
  let hexagon_area : ℝ := 3 * Real.sqrt 3 * s^2 / 2
  hexagon_area = 20 * Real.sqrt 3 := by sorry

end regular_hexagon_area_l3926_392601


namespace triangle_problem_l3926_392613

theorem triangle_problem (a b c A B C : ℝ) 
  (h1 : Real.sqrt 3 * c * Real.sin A = a * Real.cos C)
  (h2 : c = 2 * a)
  (h3 : b = 2 * Real.sqrt 3)
  (h4 : 0 < a ∧ 0 < b ∧ 0 < c)
  (h5 : 0 < A ∧ A < π)
  (h6 : 0 < B ∧ B < π)
  (h7 : 0 < C ∧ C < π)
  (h8 : A + B + C = π) :
  C = π / 6 ∧ 
  (1/2 * a * b * Real.sin C = (Real.sqrt 15 - Real.sqrt 3) / 2) := by
  sorry

end triangle_problem_l3926_392613


namespace cos_sum_equality_l3926_392619

theorem cos_sum_equality (x : Real) (h : Real.sin (x + π / 3) = 1 / 3) :
  Real.cos x + Real.cos (π / 3 - x) = Real.sqrt 3 / 3 := by
  sorry

end cos_sum_equality_l3926_392619


namespace greatest_x_value_l3926_392651

theorem greatest_x_value (x : ℝ) : 
  (2 * x^2 + 7 * x + 3 = 5) → x ≤ (1/2 : ℝ) := by
  sorry

end greatest_x_value_l3926_392651


namespace inequality_proof_equality_condition_l3926_392659

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 1) :
  (x * y) / (x^5 + x * y + y^5) + (y * z) / (y^5 + y * z + z^5) + (z * x) / (z^5 + z * x + x^5) ≤ 1 :=
by sorry

theorem equality_condition (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 1) :
  (x * y) / (x^5 + x * y + y^5) + (y * z) / (y^5 + y * z + z^5) + (z * x) / (z^5 + z * x + x^5) = 1 ↔
  x = 1 ∧ y = 1 ∧ z = 1 :=
by sorry

end inequality_proof_equality_condition_l3926_392659


namespace inequality_proof_l3926_392625

theorem inequality_proof (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hne : ¬(x = y ∧ y = z)) : 
  (x + y) * (y + z) * (z + x) > 8 * x * y * z := by
  sorry

end inequality_proof_l3926_392625


namespace f_sum_symmetric_l3926_392669

def is_transformation (f g : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x : ℝ, f x = g (a * x + b) + c

theorem f_sum_symmetric (f : ℝ → ℝ) (hf : Differentiable ℝ f)
  (h1 : ∀ x : ℝ, f x + f (1 - x) = 10)
  (h2 : ∀ x : ℝ, f (1 + x) = 3 + f x)
  (h3 : ∃ g : ℝ → ℝ, is_transformation f g) :
  ∀ x : ℝ, f x + f (-x) = 7 :=
sorry

end f_sum_symmetric_l3926_392669


namespace candy_problem_l3926_392606

theorem candy_problem (given_away eaten remaining : ℕ) 
  (h1 : given_away = 18)
  (h2 : eaten = 7)
  (h3 : remaining = 16) :
  given_away + eaten + remaining = 41 := by
  sorry

end candy_problem_l3926_392606


namespace total_nuts_eq_3200_l3926_392676

/-- The number of busy squirrels -/
def busy_squirrels : ℕ := 2

/-- The number of nuts stockpiled by each busy squirrel per day -/
def busy_squirrel_nuts_per_day : ℕ := 30

/-- The number of sleepy squirrels -/
def sleepy_squirrels : ℕ := 1

/-- The number of nuts stockpiled by each sleepy squirrel per day -/
def sleepy_squirrel_nuts_per_day : ℕ := 20

/-- The number of days the squirrels have been stockpiling -/
def stockpiling_days : ℕ := 40

/-- The total number of nuts stockpiled by all squirrels -/
def total_nuts : ℕ := 
  (busy_squirrels * busy_squirrel_nuts_per_day + 
   sleepy_squirrels * sleepy_squirrel_nuts_per_day) * 
  stockpiling_days

theorem total_nuts_eq_3200 : total_nuts = 3200 :=
by sorry

end total_nuts_eq_3200_l3926_392676


namespace cubic_root_sum_l3926_392607

theorem cubic_root_sum (a b c : ℝ) : 
  a^3 - 15*a^2 + 13*a - 8 = 0 → 
  b^3 - 15*b^2 + 13*b - 8 = 0 → 
  c^3 - 15*c^2 + 13*c - 8 = 0 → 
  a / ((1/a) + b*c) + b / ((1/b) + c*a) + c / ((1/c) + a*b) = 199/9 := by
sorry

end cubic_root_sum_l3926_392607


namespace clerical_to_total_ratio_l3926_392639

def total_employees : ℕ := 3600

def clerical_ratio (c : ℕ) : Prop :=
  (c / 2 : ℚ) = 0.2 * (total_employees - c / 2 : ℚ)

theorem clerical_to_total_ratio :
  ∃ c : ℕ, clerical_ratio c ∧ c * 3 = total_employees :=
sorry

end clerical_to_total_ratio_l3926_392639


namespace circle_equation_radius_l3926_392612

theorem circle_equation_radius (c : ℝ) : 
  (∀ x y : ℝ, x^2 + 8*x + y^2 + 10*y + c = 0 ↔ (x + 4)^2 + (y + 5)^2 = 25) → 
  c = -16 := by
  sorry

end circle_equation_radius_l3926_392612


namespace exponential_simplification_l3926_392638

theorem exponential_simplification :
  (10 ^ 1.4) * (10 ^ 0.5) / ((10 ^ 0.4) * (10 ^ 0.1)) = 10 ^ 1.4 := by
  sorry

end exponential_simplification_l3926_392638


namespace expand_product_l3926_392640

theorem expand_product (x : ℝ) : (x + 3) * (x + 7) = x^2 + 10*x + 21 := by
  sorry

end expand_product_l3926_392640


namespace parakeet_to_kitten_ratio_l3926_392609

-- Define the number of each type of pet
def num_puppies : ℕ := 2
def num_kittens : ℕ := 2
def num_parakeets : ℕ := 3

-- Define the cost of a parakeet
def parakeet_cost : ℕ := 10

-- Define the relationship between puppy and parakeet costs
def puppy_cost : ℕ := 3 * parakeet_cost

-- Define the total cost of all pets
def total_cost : ℕ := 130

-- Define the cost of a kitten (to be proved)
def kitten_cost : ℕ := (total_cost - num_puppies * puppy_cost - num_parakeets * parakeet_cost) / num_kittens

-- Theorem to prove the ratio of parakeet cost to kitten cost
theorem parakeet_to_kitten_ratio :
  parakeet_cost * 2 = kitten_cost :=
by sorry

end parakeet_to_kitten_ratio_l3926_392609


namespace four_children_prob_l3926_392681

def prob_boy_or_girl : ℚ := 1/2

def prob_at_least_one_boy_and_girl (n : ℕ) : ℚ :=
  1 - (prob_boy_or_girl ^ n + prob_boy_or_girl ^ n)

theorem four_children_prob :
  prob_at_least_one_boy_and_girl 4 = 7/8 :=
by sorry

end four_children_prob_l3926_392681


namespace hyperbola_asymptote_implies_m_l3926_392671

/-- Given a hyperbola (x²/m² - y² = 1) with m > 0, if one of its asymptotes
    is the line x + √3y = 0, then m = √3 -/
theorem hyperbola_asymptote_implies_m (m : ℝ) :
  m > 0 →
  (∃ x y : ℝ, x^2 / m^2 - y^2 = 1) →
  (∃ x y : ℝ, x + Real.sqrt 3 * y = 0) →
  m = Real.sqrt 3 :=
by sorry

end hyperbola_asymptote_implies_m_l3926_392671


namespace spinner_probability_l3926_392621

theorem spinner_probability (p_A p_B p_C p_D p_E : ℚ) :
  p_A = 3/8 →
  p_B = 1/4 →
  p_C = p_D →
  p_C = p_E →
  p_A + p_B + p_C + p_D + p_E = 1 →
  p_C = 1/8 := by
sorry

end spinner_probability_l3926_392621


namespace sin_fourth_power_decomposition_l3926_392614

theorem sin_fourth_power_decomposition :
  ∃ (b₁ b₂ b₃ b₄ : ℝ),
    (∀ θ : ℝ, Real.sin θ ^ 4 = b₁ * Real.sin θ + b₂ * Real.sin (2 * θ) + b₃ * Real.sin (3 * θ) + b₄ * Real.sin (4 * θ)) →
    b₁^2 + b₂^2 + b₃^2 + b₄^2 = 17 / 64 :=
by sorry

end sin_fourth_power_decomposition_l3926_392614


namespace equation_proof_l3926_392616

theorem equation_proof : 289 + 2 * 17 * 5 + 25 = 484 := by
  sorry

end equation_proof_l3926_392616


namespace more_girls_than_boys_l3926_392672

theorem more_girls_than_boys (total_pupils : ℕ) (girls : ℕ) 
  (h1 : total_pupils = 926)
  (h2 : girls = 692)
  (h3 : girls > total_pupils - girls) :
  girls - (total_pupils - girls) = 458 := by
sorry

end more_girls_than_boys_l3926_392672


namespace polynomial_equality_l3926_392677

theorem polynomial_equality (a b A : ℝ) (h : A / (2 * a * b) = 1 - 4 * a^2) : 
  A = 2 * a * b - 8 * a^3 * b :=
by sorry

end polynomial_equality_l3926_392677


namespace smartphone_price_l3926_392699

theorem smartphone_price (x : ℝ) : (0.90 * x - 100) = (0.80 * x - 20) → x = 800 := by
  sorry

end smartphone_price_l3926_392699


namespace max_value_sqrt_sum_l3926_392602

theorem max_value_sqrt_sum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hsum : a + b + c = 1) :
  (∀ x y z : ℝ, x > 0 → y > 0 → z > 0 → x + y + z = 1 →
    Real.sqrt (4*x + 1) + Real.sqrt (4*y + 1) + Real.sqrt (4*z + 1) ≤ Real.sqrt (4*a + 1) + Real.sqrt (4*b + 1) + Real.sqrt (4*c + 1)) ∧
  Real.sqrt (4*a + 1) + Real.sqrt (4*b + 1) + Real.sqrt (4*c + 1) = Real.sqrt 21 :=
by sorry

end max_value_sqrt_sum_l3926_392602


namespace cards_per_deck_l3926_392634

theorem cards_per_deck 
  (num_decks : ℕ) 
  (num_layers : ℕ) 
  (cards_per_layer : ℕ) 
  (h1 : num_decks = 16) 
  (h2 : num_layers = 32) 
  (h3 : cards_per_layer = 26) : 
  (num_layers * cards_per_layer) / num_decks = 52 := by
  sorry

end cards_per_deck_l3926_392634


namespace third_day_sale_l3926_392694

/-- Proves that given an average sale of 625 for 5 days, and sales of 435, 927, 230, and 562
    for 4 of those days, the sale on the remaining day must be 971. -/
theorem third_day_sale (average : ℕ) (day1 day2 day4 day5 : ℕ) :
  average = 625 →
  day1 = 435 →
  day2 = 927 →
  day4 = 230 →
  day5 = 562 →
  ∃ day3 : ℕ, day3 = 971 ∧ (day1 + day2 + day3 + day4 + day5) / 5 = average :=
by sorry

end third_day_sale_l3926_392694


namespace parabola_single_intersection_l3926_392665

/-- A parabola with equation y = x^2 - x + k has only one intersection point with the x-axis if and only if k = 1/4 -/
theorem parabola_single_intersection (k : ℝ) : 
  (∃! x, x^2 - x + k = 0) ↔ k = 1/4 := by
  sorry

end parabola_single_intersection_l3926_392665


namespace sqrt_cube_equivalence_l3926_392632

theorem sqrt_cube_equivalence (x : ℝ) (h : x ≤ 0) :
  Real.sqrt (-2 * x^3) = -x * Real.sqrt (-2 * x) := by
  sorry

end sqrt_cube_equivalence_l3926_392632
