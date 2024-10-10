import Mathlib

namespace A_D_relationship_l1452_145229

-- Define propositions
variable (A B C D : Prop)

-- Define the relationships between propositions
variable (h1 : A → B)
variable (h2 : ¬(B → A))
variable (h3 : B ↔ C)
variable (h4 : C → D)
variable (h5 : ¬(D → C))

-- Theorem to prove
theorem A_D_relationship : (A → D) ∧ ¬(D → A) := by sorry

end A_D_relationship_l1452_145229


namespace laundry_detergent_cost_l1452_145262

def budget : ℕ := 60
def shower_gel_cost : ℕ := 4
def shower_gel_quantity : ℕ := 4
def toothpaste_cost : ℕ := 3
def remaining_budget : ℕ := 30

theorem laundry_detergent_cost :
  budget - remaining_budget - (shower_gel_cost * shower_gel_quantity + toothpaste_cost) = 11 := by
  sorry

end laundry_detergent_cost_l1452_145262


namespace carla_marbles_l1452_145211

/-- The number of marbles Carla has now, given her initial marbles and the number she bought -/
def total_marbles (initial : ℕ) (bought : ℕ) : ℕ := initial + bought

/-- Theorem stating that Carla now has 187 marbles -/
theorem carla_marbles : total_marbles 53 134 = 187 := by
  sorry

end carla_marbles_l1452_145211


namespace total_triangles_is_16_l1452_145239

/-- Represents a square with diagonals and an inner square formed by midpoints -/
structure SquareWithDiagonalsAndInnerSquare :=
  (s : ℝ) -- Side length of the larger square
  (has_diagonals : Bool) -- The larger square has diagonals
  (has_inner_square : Bool) -- There's an inner square formed by midpoints

/-- Counts the total number of triangles in the figure -/
def count_triangles (square : SquareWithDiagonalsAndInnerSquare) : ℕ :=
  sorry -- The actual counting logic would go here

/-- Theorem stating that the total number of triangles is 16 -/
theorem total_triangles_is_16 (square : SquareWithDiagonalsAndInnerSquare) :
  square.has_diagonals = true → square.has_inner_square = true → count_triangles square = 16 :=
by sorry

end total_triangles_is_16_l1452_145239


namespace sin_shift_equivalence_l1452_145283

theorem sin_shift_equivalence (x : ℝ) :
  2 * Real.sin (3 * x + π / 4) = 2 * Real.sin (3 * (x + π / 12)) :=
by sorry

end sin_shift_equivalence_l1452_145283


namespace percentage_problem_l1452_145249

theorem percentage_problem (number : ℝ) (excess : ℝ) (base_percentage : ℝ) (base_number : ℝ) (percentage : ℝ) : 
  number = 6400 →
  excess = 190 →
  base_percentage = 20 →
  base_number = 650 →
  percentage = 5 →
  percentage / 100 * number = base_percentage / 100 * base_number + excess :=
by sorry

end percentage_problem_l1452_145249


namespace sum_of_divisors_231_eq_384_l1452_145216

/-- The sum of the positive whole number divisors of 231 -/
def sum_of_divisors_231 : ℕ := sorry

/-- Theorem stating that the sum of the positive whole number divisors of 231 is 384 -/
theorem sum_of_divisors_231_eq_384 : sum_of_divisors_231 = 384 := by sorry

end sum_of_divisors_231_eq_384_l1452_145216


namespace a_eq_one_necessary_not_sufficient_l1452_145296

-- Define the lines l₁ and l₂
def l₁ (a : ℝ) (x y : ℝ) : Prop := a * x + 2 * y = 0
def l₂ (a : ℝ) (x y : ℝ) : Prop := x + (a + 1) * y + 4 = 0

-- Define what it means for two lines to be parallel
def parallel (a : ℝ) : Prop := ∃ k : ℝ, k ≠ 0 ∧ ∀ x y : ℝ, l₁ a x y ↔ l₂ a (k * x) (k * y)

-- State the theorem
theorem a_eq_one_necessary_not_sufficient :
  (∀ a : ℝ, parallel a → a = 1) ∧ ¬(∀ a : ℝ, a = 1 → parallel a) :=
sorry

end a_eq_one_necessary_not_sufficient_l1452_145296


namespace luna_kibble_remaining_l1452_145250

/-- The amount of kibble remaining in the bag after feeding Luna for a day -/
def remaining_kibble (initial_amount : ℕ) (mary_morning : ℕ) (mary_evening : ℕ) 
  (frank_afternoon : ℕ) : ℕ :=
  initial_amount - (mary_morning + mary_evening + frank_afternoon + 2 * frank_afternoon)

/-- Theorem stating the remaining amount of kibble in Luna's bag -/
theorem luna_kibble_remaining : 
  remaining_kibble 12 1 1 1 = 7 := by sorry

end luna_kibble_remaining_l1452_145250


namespace circle_diameter_from_area_l1452_145212

/-- Given a circle with area 16π, prove its diameter is 8 -/
theorem circle_diameter_from_area : 
  ∀ (r : ℝ), π * r^2 = 16 * π → 2 * r = 8 := by
  sorry

end circle_diameter_from_area_l1452_145212


namespace savings_percentage_is_10_percent_l1452_145297

def basic_salary : ℝ := 240
def sales : ℝ := 2500
def commission_rate : ℝ := 0.02
def savings : ℝ := 29

def commission : ℝ := sales * commission_rate
def total_earnings : ℝ := basic_salary + commission

theorem savings_percentage_is_10_percent :
  (savings / total_earnings) * 100 = 10 := by sorry

end savings_percentage_is_10_percent_l1452_145297


namespace range_of_a_l1452_145299

def prop_p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + a*x + 1 > 0

def prop_q (a : ℝ) : Prop := ∃ x : ℝ, x^2 - x + a = 0

theorem range_of_a (a : ℝ) :
  (prop_p a ∨ prop_q a) ∧ ¬(prop_p a ∧ prop_q a) →
  a ≤ -2 ∨ (1/4 < a ∧ a < 2) :=
sorry

end range_of_a_l1452_145299


namespace sin_pi_over_4n_lower_bound_l1452_145279

theorem sin_pi_over_4n_lower_bound (n : ℕ) (hn : n > 0) :
  Real.sin (π / (4 * n)) ≥ Real.sqrt 2 / (2 * n) := by
  sorry

end sin_pi_over_4n_lower_bound_l1452_145279


namespace bernoullis_inequality_l1452_145202

theorem bernoullis_inequality (n : ℕ) (a : ℝ) (h : a > -1) :
  (1 + a)^n ≥ n * a + 1 := by
  sorry

end bernoullis_inequality_l1452_145202


namespace remainder_problem_l1452_145294

theorem remainder_problem (n : ℤ) (h : n % 7 = 2) : (4 * n + 5) % 7 = 6 := by
  sorry

end remainder_problem_l1452_145294


namespace inequality_range_l1452_145289

theorem inequality_range (t : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc 0 2 → 
    (1/8) * (2*t - t^2) ≤ x^2 - 3*x + 2 ∧ 
    x^2 - 3*x + 2 ≤ 3 - t^2) ↔ 
  t ∈ Set.Icc (-1) (1 - Real.sqrt 3) := by
sorry

end inequality_range_l1452_145289


namespace system_solution_condition_l1452_145218

theorem system_solution_condition (a : ℕ+) (A B : ℝ) :
  (∃ x y z : ℕ+, 
    x^2 + y^2 + z^2 = (B * (a : ℝ))^2 ∧
    x^2 * (A * x^2 + B * y^2) + y^2 * (A * y^2 + B * z^2) + z^2 * (A * z^2 + B * x^2) = 
      (1/4) * (2*A + B) * (B * (a : ℝ))^4) ↔
  B = 2 * A := by
sorry

end system_solution_condition_l1452_145218


namespace square_difference_l1452_145207

theorem square_difference : (39 : ℕ)^2 = (40 : ℕ)^2 - 79 := by
  sorry

end square_difference_l1452_145207


namespace complex_square_eq_143_minus_48i_l1452_145232

theorem complex_square_eq_143_minus_48i :
  ∀ z : ℂ, z^2 = -143 - 48*I ↔ z = 2 - 12*I ∨ z = -2 + 12*I := by
  sorry

end complex_square_eq_143_minus_48i_l1452_145232


namespace cone_lateral_area_l1452_145276

theorem cone_lateral_area (circumference : Real) (slant_height : Real) :
  circumference = 4 * Real.pi →
  slant_height = 3 →
  π * (circumference / (2 * π)) * slant_height = 6 * π :=
by sorry

end cone_lateral_area_l1452_145276


namespace greatest_product_of_three_l1452_145284

def S : Finset Int := {-6, -4, -2, 0, 1, 3, 5, 7}

theorem greatest_product_of_three (a b c : Int) : 
  a ∈ S → b ∈ S → c ∈ S → 
  a ≠ b → b ≠ c → a ≠ c → 
  ∀ x y z : Int, x ∈ S → y ∈ S → z ∈ S → 
  x ≠ y → y ≠ z → x ≠ z → 
  a * b * c ≤ 168 ∧ (∃ p q r : Int, p ∈ S ∧ q ∈ S ∧ r ∈ S ∧ p ≠ q ∧ q ≠ r ∧ p ≠ r ∧ p * q * r = 168) :=
by sorry

end greatest_product_of_three_l1452_145284


namespace ab_power_2013_l1452_145255

theorem ab_power_2013 (a b : ℚ) (h : |a - 2| + (2*b + 1)^2 = 0) : (a*b)^2013 = -1 := by
  sorry

end ab_power_2013_l1452_145255


namespace number_exceeding_fraction_l1452_145234

theorem number_exceeding_fraction : 
  ∀ x : ℚ, x = (3 / 8) * x + 15 → x = 24 := by
  sorry

end number_exceeding_fraction_l1452_145234


namespace equiangular_polygon_with_specific_angle_ratio_is_decagon_l1452_145228

theorem equiangular_polygon_with_specific_angle_ratio_is_decagon :
  ∀ (n : ℕ) (exterior_angle interior_angle : ℝ),
    n ≥ 3 →
    exterior_angle > 0 →
    interior_angle > 0 →
    exterior_angle + interior_angle = 180 →
    exterior_angle = (1 / 4) * interior_angle →
    360 / exterior_angle = 10 :=
by sorry

end equiangular_polygon_with_specific_angle_ratio_is_decagon_l1452_145228


namespace total_cost_is_30_l1452_145258

-- Define the cost of silverware
def silverware_cost : ℝ := 20

-- Define the cost of dinner plates as 50% of silverware cost
def dinner_plates_cost : ℝ := silverware_cost * 0.5

-- Theorem: The total cost is $30
theorem total_cost_is_30 : silverware_cost + dinner_plates_cost = 30 := by
  sorry

end total_cost_is_30_l1452_145258


namespace length_of_segment_AB_l1452_145260

/-- Given two perpendicular lines and a point P, prove the length of AB --/
theorem length_of_segment_AB (a : ℝ) : 
  ∃ (A B : ℝ × ℝ),
    (2 * A.1 - A.2 = 0) ∧ 
    (B.1 + a * B.2 = 0) ∧
    ((0 : ℝ) = (A.1 + B.1) / 2) ∧
    ((10 / a) = (A.2 + B.2) / 2) ∧
    (2 * a = -1) →
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 10 := by
  sorry

end length_of_segment_AB_l1452_145260


namespace buckingham_palace_visitor_difference_l1452_145253

/-- The number of visitors to Buckingham Palace on the previous day -/
def previous_day_visitors : ℕ := 100

/-- The number of visitors to Buckingham Palace on that day -/
def that_day_visitors : ℕ := 666

/-- The difference in visitors between that day and the previous day -/
def visitor_difference : ℕ := that_day_visitors - previous_day_visitors

theorem buckingham_palace_visitor_difference :
  visitor_difference = 566 :=
by sorry

end buckingham_palace_visitor_difference_l1452_145253


namespace smaller_circle_radius_l1452_145251

/-- Given a configuration of circles where four congruent smaller circles
    are arranged inside a larger circle such that their diameters align
    with the diameter of the larger circle, this theorem states that
    the radius of each smaller circle is one-fourth of the radius of the larger circle. -/
theorem smaller_circle_radius (R : ℝ) (r : ℝ) 
    (h1 : R = 8) -- The radius of the larger circle is 8 meters
    (h2 : 4 * r = R) -- Four smaller circle diameters align with the larger circle diameter
    : r = 2 := by
  sorry

end smaller_circle_radius_l1452_145251


namespace ratio_problem_l1452_145268

theorem ratio_problem (x y : ℝ) (h : (3 * x - 2 * y) / (x + y) = 4 / 5) : 
  x / y = 14 / 11 := by
  sorry

end ratio_problem_l1452_145268


namespace min_height_is_six_l1452_145259

/-- Represents the dimensions of a rectangular box with square bases -/
structure BoxDimensions where
  base_side : ℝ
  height : ℝ

/-- The surface area of a rectangular box with square bases -/
def surface_area (d : BoxDimensions) : ℝ :=
  2 * d.base_side^2 + 4 * d.base_side * d.height

/-- The constraint that the height is 3 units greater than the base side -/
def height_constraint (d : BoxDimensions) : Prop :=
  d.height = d.base_side + 3

/-- The constraint that the surface area is at least 90 square units -/
def area_constraint (d : BoxDimensions) : Prop :=
  surface_area d ≥ 90

theorem min_height_is_six :
  ∃ (d : BoxDimensions),
    height_constraint d ∧
    area_constraint d ∧
    d.height = 6 ∧
    ∀ (d' : BoxDimensions),
      height_constraint d' → area_constraint d' → d'.height ≥ d.height :=
by sorry

end min_height_is_six_l1452_145259


namespace feeding_sequence_count_l1452_145264

/-- Represents the number of animal pairs in the safari park -/
def num_pairs : ℕ := 5

/-- Calculates the number of possible feeding sequences -/
def feeding_sequences : ℕ := 
  (num_pairs)  -- choices for first female
  * (num_pairs - 1)  -- choices for second male
  * (num_pairs - 1)  -- choices for second female
  * (num_pairs - 2)  -- choices for third male
  * (num_pairs - 2)  -- choices for third female
  * (num_pairs - 3)  -- choices for fourth male
  * (num_pairs - 3)  -- choices for fourth female
  * (num_pairs - 4)  -- choices for fifth male
  * (num_pairs - 4)  -- choices for fifth female

theorem feeding_sequence_count : feeding_sequences = 2880 := by
  sorry

end feeding_sequence_count_l1452_145264


namespace mask_package_duration_l1452_145230

/-- Calculates the number of days a package of masks will last for a family -/
def mask_duration (total_masks : ℕ) (family_size : ℕ) (days_per_mask : ℕ) : ℕ :=
  (total_masks / family_size) * days_per_mask

/-- Theorem: A package of 100 masks lasts 80 days for a family of 5, changing masks every 4 days -/
theorem mask_package_duration :
  mask_duration 100 5 4 = 80 := by
  sorry

end mask_package_duration_l1452_145230


namespace colored_polygons_equality_l1452_145204

/-- A regular polygon with n vertices -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ
  is_regular : sorry

/-- A coloring of the vertices of a regular polygon -/
def Coloring (n : ℕ) := Fin n → ℕ

/-- The set of vertices of a given color -/
def colorVertices (n : ℕ) (p : RegularPolygon n) (c : Coloring n) (color : ℕ) : Set (ℝ × ℝ) :=
  {v | ∃ i, c i = color ∧ p.vertices i = v}

/-- Predicate to check if a set of vertices forms a regular polygon -/
def isRegularPolygon (vertices : Set (ℝ × ℝ)) : Prop := sorry

theorem colored_polygons_equality (n : ℕ) (p : RegularPolygon n) (c : Coloring n) :
  (∀ color, isRegularPolygon (colorVertices n p c color)) →
  ∃ color1 color2, color1 ≠ color2 ∧ 
    colorVertices n p c color1 = colorVertices n p c color2 := by
  sorry

end colored_polygons_equality_l1452_145204


namespace geometric_sequences_with_specific_differences_l1452_145256

/-- Two geometric sequences with the same first term and specific differences between their terms -/
theorem geometric_sequences_with_specific_differences :
  ∃ (a p q : ℚ),
    (a ≠ 0) ∧
    (p ≠ 0) ∧
    (q ≠ 0) ∧
    (a * p - a * q = 5) ∧
    (a * p^2 - a * q^2 = -5/4) ∧
    (a * p^3 - a * q^3 = 35/16) :=
by sorry

end geometric_sequences_with_specific_differences_l1452_145256


namespace lopez_family_seating_arrangements_l1452_145214

/-- Represents a family member -/
inductive FamilyMember
| MrLopez
| MrsLopez
| Child1
| Child2
| Child3

/-- Represents a seat in the car -/
inductive Seat
| Driver
| FrontPassenger
| BackLeft
| BackMiddle
| BackRight

/-- A seating arrangement is a function from Seat to FamilyMember -/
def SeatingArrangement := Seat → FamilyMember

/-- Checks if a seating arrangement is valid -/
def isValidArrangement (arr : SeatingArrangement) : Prop :=
  (arr Seat.Driver = FamilyMember.MrLopez ∨ arr Seat.Driver = FamilyMember.MrsLopez) ∧
  (∀ s₁ s₂, s₁ ≠ s₂ → arr s₁ ≠ arr s₂)

/-- The number of valid seating arrangements -/
def numValidArrangements : ℕ := sorry

theorem lopez_family_seating_arrangements :
  numValidArrangements = 48 := by sorry

end lopez_family_seating_arrangements_l1452_145214


namespace remainder_problem_l1452_145252

theorem remainder_problem (x y z w : ℕ) 
  (hx : 4 ∣ x) (hy : 4 ∣ y) (hz : 4 ∣ z) (hw : 3 ∣ w) (hpos_x : x > 0) (hpos_y : y > 0) (hpos_z : z > 0) :
  (x^2 * (y*w + z*(x + y)^2) + 7) % 6 = 1 :=
sorry

end remainder_problem_l1452_145252


namespace q_over_p_is_five_thirds_l1452_145287

theorem q_over_p_is_five_thirds (P Q : ℤ) (h : ∀ (x : ℝ), x ≠ -6 ∧ x ≠ 0 ∧ x ≠ 6 →
  (P / (x + 6) + Q / (x^2 - 6*x) : ℝ) = (x^2 - 3*x + 12) / (x^3 + x^2 - 24*x)) :
  (Q : ℚ) / (P : ℚ) = 5 / 3 := by
sorry

end q_over_p_is_five_thirds_l1452_145287


namespace cakes_served_today_l1452_145271

theorem cakes_served_today (lunch_cakes dinner_cakes : ℕ) 
  (h1 : lunch_cakes = 6) 
  (h2 : dinner_cakes = 9) : 
  lunch_cakes + dinner_cakes = 15 := by
  sorry

end cakes_served_today_l1452_145271


namespace wishing_pond_problem_l1452_145292

/-- The number of coins each person throws into the pond -/
structure CoinCounts where
  cindy_dimes : ℕ
  eric_quarters : ℕ
  garrick_nickels : ℕ
  ivy_pennies : ℕ

/-- The value of each coin type in cents -/
def coin_values : CoinCounts → ℕ
  | ⟨cd, eq, gn, ip⟩ => cd * 10 + eq * 25 + gn * 5 + ip * 1

/-- The problem statement -/
theorem wishing_pond_problem (coins : CoinCounts) : 
  coins.eric_quarters = 3 →
  coins.garrick_nickels = 8 →
  coins.ivy_pennies = 60 →
  coin_values coins = 200 →
  coins.cindy_dimes = 2 := by
  sorry

#eval coin_values ⟨2, 3, 8, 60⟩

end wishing_pond_problem_l1452_145292


namespace fourth_root_simplification_l1452_145272

theorem fourth_root_simplification :
  ∃ (c d : ℕ+), (c : ℝ) * ((d : ℝ)^(1/4 : ℝ)) = (3^5 * 5^3 : ℝ)^(1/4 : ℝ) ∧ c + d = 378 := by
  sorry

end fourth_root_simplification_l1452_145272


namespace tamias_dinner_problem_l1452_145225

/-- The number of smaller pieces each large slice is cut into, given the total number of bell peppers,
    the number of large slices per bell pepper, and the total number of slices and pieces. -/
def smaller_pieces_per_slice (total_peppers : ℕ) (large_slices_per_pepper : ℕ) (total_slices_and_pieces : ℕ) : ℕ :=
  let total_large_slices := total_peppers * large_slices_per_pepper
  let large_slices_to_cut := total_large_slices / 2
  let smaller_pieces_needed := total_slices_and_pieces - total_large_slices
  smaller_pieces_needed / large_slices_to_cut

theorem tamias_dinner_problem :
  smaller_pieces_per_slice 5 20 200 = 2 := by
  sorry

end tamias_dinner_problem_l1452_145225


namespace euros_to_rubles_conversion_l1452_145237

/-- Exchange rate from euros to US dollars -/
def euro_to_usd_rate : ℚ := 12 / 10

/-- Exchange rate from US dollars to rubles -/
def usd_to_ruble_rate : ℚ := 60

/-- Cost of the travel package in euros -/
def travel_package_cost : ℚ := 600

/-- Theorem stating the equivalence of 600 euros to 43200 rubles given the exchange rates -/
theorem euros_to_rubles_conversion :
  (travel_package_cost * euro_to_usd_rate * usd_to_ruble_rate : ℚ) = 43200 := by
  sorry


end euros_to_rubles_conversion_l1452_145237


namespace circle_k_value_l1452_145263

def larger_circle_radius : ℝ := 15
def smaller_circle_radius : ℝ := 10
def point_P : ℝ × ℝ := (9, 12)
def point_S (k : ℝ) : ℝ × ℝ := (0, k)
def QR : ℝ := 5

theorem circle_k_value :
  ∀ k : ℝ,
  (point_P.1^2 + point_P.2^2 = larger_circle_radius^2) →
  ((point_S k).1^2 + (point_S k).2^2 = smaller_circle_radius^2) →
  (larger_circle_radius - smaller_circle_radius = QR) →
  (k = 10 ∨ k = -10) :=
by sorry

end circle_k_value_l1452_145263


namespace eleven_step_paths_through_F_l1452_145270

/-- A point on the 6x6 grid -/
structure Point where
  x : Nat
  y : Nat
  h_x : x ≤ 5
  h_y : y ≤ 5

/-- The number of paths between two points on the grid -/
def num_paths (start finish : Point) : Nat :=
  Nat.choose (finish.x - start.x + finish.y - start.y) (finish.x - start.x)

theorem eleven_step_paths_through_F : 
  let E : Point := ⟨0, 5, by norm_num, by norm_num⟩
  let F : Point := ⟨3, 3, by norm_num, by norm_num⟩
  let G : Point := ⟨5, 0, by norm_num, by norm_num⟩
  (num_paths E F) * (num_paths F G) = 100 := by
  sorry

end eleven_step_paths_through_F_l1452_145270


namespace expression_simplification_l1452_145220

theorem expression_simplification (a : ℝ) (h : a = Real.sqrt 3 + 1) :
  (a + 1) / (a^2 - 2*a + 1) / (1 + 2 / (a - 1)) = Real.sqrt 3 / 3 := by
  sorry

end expression_simplification_l1452_145220


namespace union_equals_reals_l1452_145248

-- Define set A
def A : Set ℝ := {x | x^2 - x - 2 ≥ 0}

-- Define set B
def B : Set ℝ := {x | x > -1}

-- Theorem statement
theorem union_equals_reals : A ∪ B = Set.univ := by sorry

end union_equals_reals_l1452_145248


namespace total_participants_l1452_145293

/-- Represents the exam scores and statistics -/
structure ExamStatistics where
  low_scorers : ℕ  -- Number of people scoring no more than 30
  low_avg : ℝ      -- Average score of low scorers
  high_scorers : ℕ -- Number of people scoring no less than 80
  high_avg : ℝ     -- Average score of high scorers
  above_30_avg : ℝ -- Average score of those scoring more than 30
  below_80_avg : ℝ -- Average score of those scoring less than 80

/-- Theorem stating the total number of participants in the exam -/
theorem total_participants (stats : ExamStatistics) 
  (h1 : stats.low_scorers = 153)
  (h2 : stats.low_avg = 24)
  (h3 : stats.high_scorers = 59)
  (h4 : stats.high_avg = 92)
  (h5 : stats.above_30_avg = 62)
  (h6 : stats.below_80_avg = 54) :
  stats.low_scorers + stats.high_scorers + 
  ((stats.low_scorers * (stats.below_80_avg - stats.low_avg) + 
    stats.high_scorers * (stats.high_avg - stats.above_30_avg)) / 
   (stats.above_30_avg - stats.below_80_avg)) = 1007 := by
  sorry


end total_participants_l1452_145293


namespace function_properties_imply_b_range_l1452_145206

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

def zero_points_count (f : ℝ → ℝ) (a b : ℝ) (n : ℕ) : Prop :=
  ∃ (zeros : Finset ℝ), zeros.card = n ∧ (∀ x ∈ zeros, a ≤ x ∧ x ≤ b ∧ f x = 0)

theorem function_properties_imply_b_range (f : ℝ → ℝ) (b : ℝ) :
  is_odd_function f →
  has_period f 4 →
  (∀ x ∈ Set.Ioo 0 2, f x = Real.log (x^2 - x + b)) →
  zero_points_count f (-2) 2 5 →
  (1/4 < b ∧ b ≤ 1) ∨ b = 5/4 :=
by sorry

end function_properties_imply_b_range_l1452_145206


namespace function_derivative_problem_l1452_145236

/-- Given a function f(x) = x(x+k)(x+2k)(x-3k) where f'(0) = 6, prove that k = -1 -/
theorem function_derivative_problem (k : ℝ) : 
  (∃ f : ℝ → ℝ, (∀ x, f x = x * (x + k) * (x + 2*k) * (x - 3*k)) ∧ 
   (deriv f) 0 = 6) → 
  k = -1 := by
  sorry

end function_derivative_problem_l1452_145236


namespace tall_students_not_well_defined_other_options_well_defined_l1452_145224

-- Define a type for potential sets
inductive PotentialSet
  | NaturalNumbers1to20
  | AllRectangles
  | NaturalNumbersLessThan10
  | TallStudents

-- Define a predicate for well-defined sets
def isWellDefinedSet (s : PotentialSet) : Prop :=
  match s with
  | PotentialSet.NaturalNumbers1to20 => true
  | PotentialSet.AllRectangles => true
  | PotentialSet.NaturalNumbersLessThan10 => true
  | PotentialSet.TallStudents => false

-- Theorem stating that "Tall students" is not a well-defined set
theorem tall_students_not_well_defined :
  ¬(isWellDefinedSet PotentialSet.TallStudents) :=
by sorry

-- Theorem stating that other options are well-defined sets
theorem other_options_well_defined :
  (isWellDefinedSet PotentialSet.NaturalNumbers1to20) ∧
  (isWellDefinedSet PotentialSet.AllRectangles) ∧
  (isWellDefinedSet PotentialSet.NaturalNumbersLessThan10) :=
by sorry

end tall_students_not_well_defined_other_options_well_defined_l1452_145224


namespace arithmetic_calculation_l1452_145238

theorem arithmetic_calculation : 5 * 7 + 9 * 4 - 36 / 3 = 59 := by
  sorry

end arithmetic_calculation_l1452_145238


namespace simplify_expression_l1452_145210

theorem simplify_expression (a : ℝ) (h : 1 < a ∧ a < 2) :
  Real.sqrt ((a - 3)^2) + |1 - a| = 2 := by
sorry

end simplify_expression_l1452_145210


namespace correlation_coefficient_inequality_l1452_145298

def X : List ℝ := [10, 11.3, 11.8, 12.5, 13]
def Y : List ℝ := [1, 2, 3, 4, 5]
def U : List ℝ := [10, 11.3, 11.8, 12.5, 13]
def V : List ℝ := [5, 4, 3, 2, 1]

def linear_correlation_coefficient (x y : List ℝ) : ℝ :=
  sorry

def r₁ : ℝ := linear_correlation_coefficient X Y
def r₂ : ℝ := linear_correlation_coefficient U V

theorem correlation_coefficient_inequality : r₂ < 0 ∧ 0 < r₁ := by
  sorry

end correlation_coefficient_inequality_l1452_145298


namespace difference_of_squares_l1452_145282

theorem difference_of_squares : 525^2 - 475^2 = 50000 := by sorry

end difference_of_squares_l1452_145282


namespace range_of_t_l1452_145291

theorem range_of_t (a b t : ℝ) (ha : a > 0) (hb : b > 0) (hab : 2 * a + b = 1) 
  (ht : ∀ a b, a > 0 → b > 0 → 2 * a + b = 1 → 2 * Real.sqrt (a * b) - 4 * a^2 - b^2 ≤ t - 1/2) : 
  t ≥ Real.sqrt 2 / 2 := by
  sorry

end range_of_t_l1452_145291


namespace margo_distance_l1452_145281

/-- Represents the total distance Margo traveled in miles -/
def total_distance : ℝ := 2.5

/-- Represents the time Margo took to walk to her friend's house in minutes -/
def walk_time : ℝ := 15

/-- Represents the time Margo took to jog back home in minutes -/
def jog_time : ℝ := 10

/-- Represents Margo's average speed for the entire trip in miles per hour -/
def average_speed : ℝ := 6

/-- Theorem stating that the total distance Margo traveled is 2.5 miles -/
theorem margo_distance :
  total_distance = average_speed * (walk_time + jog_time) / 60 :=
by sorry

end margo_distance_l1452_145281


namespace inscribed_square_area_l1452_145265

/-- The area of a square inscribed in a circle, which is itself inscribed in an equilateral triangle -/
theorem inscribed_square_area (s : ℝ) (h : s = 6) : 
  let r := s / (2 * Real.sqrt 3)
  let d := 2 * r
  let side := d / Real.sqrt 2
  side ^ 2 = 6 := by sorry

end inscribed_square_area_l1452_145265


namespace find_k_l1452_145241

-- Define the polynomials A and B
def A (x k : ℝ) : ℝ := 2 * x^2 + k * x - 6 * x

def B (x k : ℝ) : ℝ := -x^2 + k * x - 1

-- Define the condition for A + 2B to be independent of x
def independent_of_x (k : ℝ) : Prop :=
  ∀ x : ℝ, ∃ c : ℝ, A x k + 2 * B x k = c

-- Theorem statement
theorem find_k : ∃ k : ℝ, independent_of_x k ∧ k = 2 :=
sorry

end find_k_l1452_145241


namespace binomial_expansion_problem_l1452_145247

theorem binomial_expansion_problem (n : ℕ) (x : ℝ) :
  (∃ (a b : ℕ), a ≠ b ∧ a > 2 ∧ b > 2 ∧ (Nat.choose n a = Nat.choose n b)) →
  (n = 6 ∧ 
   ∃ (k : ℕ), k = 3 ∧
   ((-1)^k * 2^(n-k) * Nat.choose n k : ℤ) = -160) := by
  sorry

end binomial_expansion_problem_l1452_145247


namespace equation_transformation_l1452_145203

variable (x y : ℝ)

theorem equation_transformation (h : y = x + 1/x) :
  x^4 - x^3 - 6*x^2 - x + 1 = 0 ↔ x^2 * (y^2 - y - 6) = 0 :=
by sorry

end equation_transformation_l1452_145203


namespace sum_of_a_and_b_is_71_l1452_145213

/-- Represents the product of a sequence following the pattern (n+1)/n from 5/3 to a/b --/
def sequence_product (a b : ℕ) : ℚ :=
  a / 3

theorem sum_of_a_and_b_is_71 (a b : ℕ) (h : sequence_product a b = 12) : a + b = 71 := by
  sorry

end sum_of_a_and_b_is_71_l1452_145213


namespace isosceles_trapezoid_larger_base_l1452_145261

/-- An isosceles trapezoid with given measurements -/
structure IsoscelesTrapezoid where
  leg : ℝ
  smallerBase : ℝ
  diagonal : ℝ
  largerBase : ℝ

/-- The isosceles trapezoid satisfies the given conditions -/
def satisfiesConditions (t : IsoscelesTrapezoid) : Prop :=
  t.leg = 10 ∧ t.smallerBase = 6 ∧ t.diagonal = 14

/-- Theorem: The larger base of the isosceles trapezoid is 16 -/
theorem isosceles_trapezoid_larger_base
  (t : IsoscelesTrapezoid)
  (h : satisfiesConditions t) :
  t.largerBase = 16 := by
  sorry

end isosceles_trapezoid_larger_base_l1452_145261


namespace ian_painted_48_faces_l1452_145227

/-- The number of faces on a single cuboid -/
def faces_per_cuboid : ℕ := 6

/-- The number of cuboids Ian painted -/
def num_cuboids : ℕ := 8

/-- The total number of faces painted by Ian -/
def total_faces_painted : ℕ := faces_per_cuboid * num_cuboids

/-- Theorem stating that the total number of faces painted by Ian is 48 -/
theorem ian_painted_48_faces : total_faces_painted = 48 := by
  sorry

end ian_painted_48_faces_l1452_145227


namespace min_value_w_l1452_145240

theorem min_value_w (x y : ℝ) : 3 * x^2 + 3 * y^2 + 12 * x - 6 * y + 30 ≥ 15 := by
  sorry

end min_value_w_l1452_145240


namespace scientific_notation_of_0_00000065_l1452_145254

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_of_0_00000065 :
  toScientificNotation 0.00000065 = ScientificNotation.mk 6.5 (-7) sorry := by
  sorry

end scientific_notation_of_0_00000065_l1452_145254


namespace complex_modulus_sqrt_two_l1452_145269

theorem complex_modulus_sqrt_two (x y : ℝ) :
  (Complex.I + 1) * x = Complex.I * y + 1 →
  Complex.abs (x + Complex.I * y) = Real.sqrt 2 := by
sorry

end complex_modulus_sqrt_two_l1452_145269


namespace water_tank_problem_l1452_145217

/-- Represents the time (in minutes) it takes for pipe A to fill the tank -/
def fill_time : ℚ := 15

/-- Represents the time (in minutes) it takes for pipe B to empty the tank -/
def empty_time : ℚ := 6

/-- Represents the time (in minutes) it takes to empty or fill the tank completely with both pipes open -/
def both_pipes_time : ℚ := 2

/-- Represents the fraction of the tank that is currently full -/
def current_fill : ℚ := 4/5

theorem water_tank_problem :
  (1 / fill_time - 1 / empty_time) * both_pipes_time = 1 - current_fill :=
by sorry

end water_tank_problem_l1452_145217


namespace mean_squared_sum_l1452_145219

theorem mean_squared_sum (a b c : ℝ) 
  (h_arithmetic : (a + b + c) / 3 = 7)
  (h_geometric : (a * b * c) ^ (1/3 : ℝ) = 6)
  (h_harmonic : 3 / (1/a + 1/b + 1/c) = 5) :
  a^2 + b^2 + c^2 = 181.8 := by
  sorry

end mean_squared_sum_l1452_145219


namespace rectangle_ratio_is_two_l1452_145274

/-- Represents a rectangle with side lengths x and y -/
structure Rectangle where
  x : ℝ
  y : ℝ

/-- Represents a square configuration with an inner square and four surrounding rectangles -/
structure SquareConfig where
  inner_side : ℝ
  rect : Rectangle
  h_outer_area : (inner_side + 2 * rect.y)^2 = 9 * inner_side^2
  h_rect_placement : inner_side + rect.x = inner_side + 2 * rect.y

theorem rectangle_ratio_is_two (config : SquareConfig) :
  config.rect.x / config.rect.y = 2 := by
  sorry

end rectangle_ratio_is_two_l1452_145274


namespace lenas_muffins_l1452_145226

/-- Represents the cost of a single item -/
structure ItemCost where
  cake : ℚ
  muffin : ℚ
  bagel : ℚ

/-- Represents a purchase of items -/
structure Purchase where
  cakes : ℕ
  muffins : ℕ
  bagels : ℕ

/-- Calculates the total cost of a purchase -/
def totalCost (cost : ItemCost) (purchase : Purchase) : ℚ :=
  cost.cake * purchase.cakes + cost.muffin * purchase.muffins + cost.bagel * purchase.bagels

/-- The main theorem to prove -/
theorem lenas_muffins (cost : ItemCost) : 
  let petya := Purchase.mk 1 2 3
  let anya := Purchase.mk 3 0 1
  let kolya := Purchase.mk 0 6 0
  let lena := Purchase.mk 2 0 2
  totalCost cost petya = totalCost cost anya ∧ 
  totalCost cost anya = totalCost cost kolya →
  ∃ n : ℕ, totalCost cost lena = totalCost cost (Purchase.mk 0 n 0) ∧ n = 5 := by
  sorry


end lenas_muffins_l1452_145226


namespace product_121_54_l1452_145266

theorem product_121_54 : 121 * 54 = 6534 := by
  sorry

end product_121_54_l1452_145266


namespace ellipse_coincide_hyperbola_focus_l1452_145233

def ellipse_equation (a b x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

def hyperbola_equation (x y : ℝ) : Prop := x^2 - y^2 / 3 = 1

def eccentricity (e : ℝ) : Prop := e = 1 / 2

theorem ellipse_coincide_hyperbola_focus (a b : ℝ) :
  eccentricity (1 / 2) →
  (∃ x y, ellipse_equation a b x y) →
  (∃ x y, hyperbola_equation x y) →
  (∀ x y, ellipse_equation a b x y ↔ ellipse_equation 4 (12 : ℝ).sqrt x y) :=
sorry

end ellipse_coincide_hyperbola_focus_l1452_145233


namespace cube_diagonals_count_l1452_145223

structure Cube where
  vertices : Nat
  edges : Nat

def face_diagonals (c : Cube) : Nat := 12

def space_diagonals (c : Cube) : Nat := 4

def total_diagonals (c : Cube) : Nat := face_diagonals c + space_diagonals c

theorem cube_diagonals_count (c : Cube) (h1 : c.vertices = 8) (h2 : c.edges = 12) :
  total_diagonals c = 16 ∧ face_diagonals c = 12 ∧ space_diagonals c = 4 := by
  sorry

end cube_diagonals_count_l1452_145223


namespace smallest_sum_of_four_consecutive_primes_divisible_by_five_l1452_145275

/-- A function that returns true if a number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that checks if four consecutive numbers are all prime -/
def fourConsecutivePrimes (a b c d : ℕ) : Prop :=
  isPrime a ∧ isPrime b ∧ isPrime c ∧ isPrime d ∧
  b = a + 1 ∧ c = b + 1 ∧ d = c + 1

/-- The main theorem -/
theorem smallest_sum_of_four_consecutive_primes_divisible_by_five :
  ∃ (a b c d : ℕ),
    fourConsecutivePrimes a b c d ∧
    (a + b + c + d) % 5 = 0 ∧
    a + b + c + d = 60 ∧
    ∀ (w x y z : ℕ),
      fourConsecutivePrimes w x y z →
      (w + x + y + z) % 5 = 0 →
      w + x + y + z ≥ 60 :=
sorry

end smallest_sum_of_four_consecutive_primes_divisible_by_five_l1452_145275


namespace binomial_expansion_coefficient_l1452_145209

theorem binomial_expansion_coefficient (x : ℝ) (a : Fin 9 → ℝ) :
  (x - 1)^8 = a 0 + a 1 * (1 + x) + a 2 * (1 + x)^2 + a 3 * (1 + x)^3 + 
              a 4 * (1 + x)^4 + a 5 * (1 + x)^5 + a 6 * (1 + x)^6 + 
              a 7 * (1 + x)^7 + a 8 * (1 + x)^8 →
  a 5 = -448 := by
sorry

end binomial_expansion_coefficient_l1452_145209


namespace right_triangle_area_l1452_145205

/-- The area of a right triangle with hypotenuse 10 and sum of other sides 14 is 24 -/
theorem right_triangle_area (a b c : ℝ) (h1 : a + b = 14) (h2 : c = 10) (h3 : a^2 + b^2 = c^2) :
  (1/2) * a * b = 24 := by
sorry

end right_triangle_area_l1452_145205


namespace complex_equation_unique_solution_l1452_145273

theorem complex_equation_unique_solution :
  ∃! (c : ℝ), Complex.abs (1 - 2 * Complex.I - (c - 3 * Complex.I)) = 1 :=
by sorry

end complex_equation_unique_solution_l1452_145273


namespace sequence_sum_problem_l1452_145200

/-- Sum of an arithmetic sequence -/
def arithmeticSum (a₁ : ℕ) (aₙ : ℕ) : ℕ := 
  let n := aₙ - a₁ + 1
  n * (a₁ + aₙ) / 2

theorem sequence_sum_problem : 
  (arithmeticSum 2001 2093) - (arithmeticSum 221 313) + (arithmeticSum 401 493) = 207141 := by
  sorry

end sequence_sum_problem_l1452_145200


namespace XZ_length_l1452_145208

-- Define the circle and points
def Circle : Type := Unit
def Point : Type := ℝ × ℝ

-- Define the radius of the circle
def radius : ℝ := 7

-- Define the points on the circle
def X : Point := sorry
def Y : Point := sorry
def Z : Point := sorry
def W : Point := sorry

-- Define the distance function
def distance (p q : Point) : ℝ := sorry

-- State the conditions
axiom on_circle_X : distance X (0, 0) = radius
axiom on_circle_Y : distance Y (0, 0) = radius
axiom XY_distance : distance X Y = 8
axiom Z_midpoint_arc : sorry  -- Z is the midpoint of the minor arc XY
axiom W_midpoint_XZ : distance X W = distance W Z
axiom YW_distance : distance Y W = 6

-- State the theorem to be proved
theorem XZ_length : distance X Z = 8 := by sorry

end XZ_length_l1452_145208


namespace negation_equivalence_l1452_145278

theorem negation_equivalence : 
  (¬ ∃ x : ℝ, x^2 > 1) ↔ (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1) :=
by sorry

end negation_equivalence_l1452_145278


namespace find_other_divisor_l1452_145215

theorem find_other_divisor : ∃ (x : ℕ), x > 1 ∧ 261 % 7 = 2 ∧ 261 % x = 2 ∧ ∀ (y : ℕ), y > 1 → 261 % y = 2 → y = 7 ∨ y = x := by
  sorry

end find_other_divisor_l1452_145215


namespace ellipse_foci_coordinates_l1452_145277

/-- The coordinates of the foci of an ellipse with equation x^2/10 + y^2 = 1 are (3,0) and (-3,0) -/
theorem ellipse_foci_coordinates :
  let ellipse := {(x, y) : ℝ × ℝ | x^2/10 + y^2 = 1}
  ∃ (f₁ f₂ : ℝ × ℝ), f₁ ∈ ellipse ∧ f₂ ∈ ellipse ∧ f₁ = (3, 0) ∧ f₂ = (-3, 0) ∧
    ∀ (f : ℝ × ℝ), f ∈ ellipse → f = f₁ ∨ f = f₂ :=
by sorry

end ellipse_foci_coordinates_l1452_145277


namespace monitor_pixels_l1452_145295

/-- Calculates the total number of pixels on a monitor given its dimensions and DPI. -/
theorem monitor_pixels 
  (width : ℕ) 
  (height : ℕ) 
  (dpi : ℕ) 
  (h1 : width = 21) 
  (h2 : height = 12) 
  (h3 : dpi = 100) : 
  width * dpi * (height * dpi) = 2520000 := by
  sorry

#check monitor_pixels

end monitor_pixels_l1452_145295


namespace opposite_edge_angles_not_all_acute_or_obtuse_l1452_145257

/-- Represents a convex polyhedral angle -/
structure ConvexPolyhedralAngle where
  /-- All dihedral angles are 60° -/
  dihedral_angles_60 : Bool

/-- Represents the angles between opposite edges of a polyhedral angle -/
inductive OppositeEdgeAngles
  | Acute : OppositeEdgeAngles
  | Obtuse : OppositeEdgeAngles
  | Mixed : OppositeEdgeAngles

/-- 
Given a convex polyhedral angle with all dihedral angles equal to 60°, 
it's impossible for the angles between opposite edges to be simultaneously acute or simultaneously obtuse.
-/
theorem opposite_edge_angles_not_all_acute_or_obtuse (angle : ConvexPolyhedralAngle) 
  (h : angle.dihedral_angles_60 = true) : 
  ∃ (opp_angles : OppositeEdgeAngles), opp_angles = OppositeEdgeAngles.Mixed :=
sorry

end opposite_edge_angles_not_all_acute_or_obtuse_l1452_145257


namespace vector_difference_magnitude_l1452_145290

theorem vector_difference_magnitude : 
  let a : ℝ × ℝ := (Real.cos (π / 6), Real.sin (π / 6))
  let b : ℝ × ℝ := (Real.cos (5 * π / 6), Real.sin (5 * π / 6))
  ((a.1 - b.1)^2 + (a.2 - b.2)^2).sqrt = Real.sqrt 3 := by
  sorry

end vector_difference_magnitude_l1452_145290


namespace triangle_angle_problem_l1452_145280

theorem triangle_angle_problem (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 → -- angles are positive
  b = 2 * a → -- second angle is double the first
  c = a - 40 → -- third angle is 40 less than the first
  a + b + c = 180 → -- sum of angles in a triangle
  a = 55 := by sorry

end triangle_angle_problem_l1452_145280


namespace sum_five_consecutive_odds_mod_12_l1452_145267

theorem sum_five_consecutive_odds_mod_12 (n : ℕ) : 
  (((2*n + 1) + (2*n + 3) + (2*n + 5) + (2*n + 7) + (2*n + 9)) % 12) = 9 := by
  sorry

end sum_five_consecutive_odds_mod_12_l1452_145267


namespace purely_imaginary_complex_number_l1452_145245

theorem purely_imaginary_complex_number (a : ℝ) : 
  (((a^2 - 3*a + 2) : ℂ) + (a - 1)*I = (0 : ℂ) + ((a - 1)*I)) → a = 2 :=
by
  sorry

end purely_imaginary_complex_number_l1452_145245


namespace josh_initial_money_l1452_145221

/-- Josh's initial amount of money given his expenses and remaining balance -/
def initial_amount (spent1 spent2 remaining : ℚ) : ℚ :=
  spent1 + spent2 + remaining

theorem josh_initial_money :
  initial_amount 1.75 1.25 6 = 9 := by
  sorry

end josh_initial_money_l1452_145221


namespace f_4_equals_1559_l1452_145288

-- Define the polynomial f(x) = x^5 + 3x^4 - 5x^3 + 7x^2 - 9x + 11
def f (x : ℝ) : ℝ := x^5 + 3*x^4 - 5*x^3 + 7*x^2 - 9*x + 11

-- Define Horner's method for this specific polynomial
def horner (x : ℝ) : ℝ := ((((x + 3) * x - 5) * x + 7) * x - 9) * x + 11

-- Theorem stating that f(4) = 1559 using Horner's method
theorem f_4_equals_1559 : f 4 = 1559 ∧ horner 4 = 1559 := by
  sorry

end f_4_equals_1559_l1452_145288


namespace arithmetic_sequence_properties_l1452_145231

/-- An arithmetic sequence with given first and last terms -/
structure ArithmeticSequence where
  a₁ : ℚ  -- First term
  a₃₀ : ℚ  -- 30th term
  is_arithmetic : a₃₀ = a₁ + 29 * ((a₃₀ - a₁) / 29)  -- Condition for arithmetic sequence

/-- Properties of the arithmetic sequence -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) 
    (h₁ : seq.a₁ = 5)
    (h₂ : seq.a₃₀ = 100) : 
  let d := (seq.a₃₀ - seq.a₁) / 29
  let a₈ := seq.a₁ + 7 * d
  let a₁₅ := seq.a₁ + 14 * d
  let S₁₅ := 15 / 2 * (seq.a₁ + a₁₅)
  (a₈ = 25 + 1 / 29) ∧ (S₁₅ = 393 + 2 / 29) := by
  sorry


end arithmetic_sequence_properties_l1452_145231


namespace shortest_distance_parabola_line_l1452_145222

/-- The shortest distance between a point on the parabola y = x^2 - 4x + 7
    and a point on the line y = 2x - 5 is 3√5/5 -/
theorem shortest_distance_parabola_line : 
  let parabola := fun x : ℝ => x^2 - 4*x + 7
  let line := fun x : ℝ => 2*x - 5
  ∃ (min_dist : ℝ), 
    (∀ (p q : ℝ × ℝ), 
      (p.2 = parabola p.1) → 
      (q.2 = line q.1) → 
      dist p q ≥ min_dist) ∧
    (∃ (p q : ℝ × ℝ), 
      (p.2 = parabola p.1) ∧ 
      (q.2 = line q.1) ∧ 
      dist p q = min_dist) ∧
    min_dist = 3 * Real.sqrt 5 / 5 := by
  sorry

end shortest_distance_parabola_line_l1452_145222


namespace josh_paid_six_dollars_l1452_145201

/-- The amount Josh paid for string cheese -/
def string_cheese_cost (packs : ℕ) (cheeses_per_pack : ℕ) (cents_per_cheese : ℕ) : ℚ :=
  (packs * cheeses_per_pack * cents_per_cheese : ℚ) / 100

/-- Theorem stating that Josh paid 6 dollars for the string cheese -/
theorem josh_paid_six_dollars :
  string_cheese_cost 3 20 10 = 6 := by
  sorry

end josh_paid_six_dollars_l1452_145201


namespace milena_age_l1452_145286

theorem milena_age :
  ∀ (milena_age grandmother_age grandfather_age : ℕ),
    grandmother_age = 9 * milena_age →
    grandfather_age = grandmother_age + 2 →
    grandfather_age - milena_age = 58 →
    milena_age = 7 := by
  sorry

end milena_age_l1452_145286


namespace sequence_problem_l1452_145285

/-- Given a sequence {a_n} and an arithmetic sequence {b_n}, prove that a_6 = 33 -/
theorem sequence_problem (a b : ℕ → ℕ) : 
  a 1 = 3 →  -- First term of {a_n} is 3
  b 1 = 2 →  -- b_1 = 2
  b 3 = 6 →  -- b_3 = 6
  (∀ n : ℕ, n > 0 → b n = a (n + 1) - a n) →  -- b_n = a_{n+1} - a_n for n ∈ ℕ*
  (∀ n : ℕ, n > 0 → ∃ d : ℕ, b (n + 1) = b n + d) →  -- {b_n} is an arithmetic sequence
  a 6 = 33 := by
sorry

end sequence_problem_l1452_145285


namespace lars_baking_capacity_l1452_145242

/-- Represents the baking capacity of Lars' bakeshop -/
structure Bakeshop where
  baguettes_per_two_hours : ℕ
  baking_hours_per_day : ℕ
  total_breads_per_day : ℕ

/-- Calculates the number of loaves of bread that can be baked per hour -/
def loaves_per_hour (shop : Bakeshop) : ℚ :=
  let baguettes_per_day := shop.baguettes_per_two_hours * (shop.baking_hours_per_day / 2)
  let loaves_per_day := shop.total_breads_per_day - baguettes_per_day
  loaves_per_day / shop.baking_hours_per_day

/-- Theorem stating that Lars can bake 10 loaves of bread per hour -/
theorem lars_baking_capacity :
  let lars_shop : Bakeshop := {
    baguettes_per_two_hours := 30,
    baking_hours_per_day := 6,
    total_breads_per_day := 150
  }
  loaves_per_hour lars_shop = 10 := by
  sorry

end lars_baking_capacity_l1452_145242


namespace quadratic_completion_of_square_l1452_145235

/-- Given a quadratic expression 3x^2 + 9x + 20, when expressed in the form a(x - h)^2 + k,
    the value of h is -3/2. -/
theorem quadratic_completion_of_square (x : ℝ) :
  ∃ (a k : ℝ), 3*x^2 + 9*x + 20 = a*(x + 3/2)^2 + k :=
by sorry

end quadratic_completion_of_square_l1452_145235


namespace four_digit_number_with_specific_remainders_l1452_145246

theorem four_digit_number_with_specific_remainders :
  ∃! n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 131 = 112 ∧ n % 132 = 98 :=
by
  sorry

end four_digit_number_with_specific_remainders_l1452_145246


namespace percent_y_of_x_l1452_145243

theorem percent_y_of_x (x y : ℝ) (h : 0.5 * (x - y) = 0.3 * (x + y)) : y = 0.25 * x := by
  sorry

end percent_y_of_x_l1452_145243


namespace second_divisor_problem_l1452_145244

theorem second_divisor_problem (x : ℕ) : 
  (210 % 13 = 3) → (210 % x = 7) → (x = 203) :=
by sorry

end second_divisor_problem_l1452_145244
