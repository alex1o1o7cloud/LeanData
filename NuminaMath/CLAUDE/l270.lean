import Mathlib

namespace commission_rate_proof_l270_27084

/-- The commission rate for an agent who earned a commission of 12.50 on sales of 250. -/
theorem commission_rate_proof (commission : ℝ) (sales : ℝ) 
  (h1 : commission = 12.50) (h2 : sales = 250) :
  (commission / sales) * 100 = 5 := by
sorry

end commission_rate_proof_l270_27084


namespace speed_ratio_after_meeting_l270_27067

/-- Represents a car with a speed -/
structure Car where
  speed : ℝ

/-- Represents the scenario of two cars meeting -/
structure CarMeeting where
  carA : Car
  carB : Car
  totalDistance : ℝ
  timeToMeet : ℝ
  timeAAfterMeet : ℝ
  timeBAfterMeet : ℝ

/-- The theorem stating the ratio of speeds given the conditions -/
theorem speed_ratio_after_meeting (m : CarMeeting) 
  (h1 : m.timeAAfterMeet = 4)
  (h2 : m.timeBAfterMeet = 1)
  (h3 : m.totalDistance = m.carA.speed * m.timeToMeet + m.carB.speed * m.timeToMeet)
  (h4 : m.carA.speed * m.timeAAfterMeet = m.totalDistance - m.carA.speed * m.timeToMeet)
  (h5 : m.carB.speed * m.timeBAfterMeet = m.totalDistance - m.carB.speed * m.timeToMeet) :
  m.carA.speed / m.carB.speed = 1 / 2 := by
  sorry

end speed_ratio_after_meeting_l270_27067


namespace circle_M_fixed_point_l270_27011

-- Define the circle M
def circle_M (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - Real.sqrt 3)^2 = 4

-- Define the curve on which the center of M lies
def center_curve (x y : ℝ) : Prop :=
  y = Real.sqrt 3 / x

-- Define the line l
def line_l (x y : ℝ) : Prop :=
  y = -Real.sqrt 3 / 3 * x + 4

-- Define the line y = √3
def line_sqrt3 (x y : ℝ) : Prop :=
  y = Real.sqrt 3

-- Define the line x = 5
def line_x5 (x : ℝ) : Prop :=
  x = 5

-- Theorem statement
theorem circle_M_fixed_point :
  ∀ (O C D E F G H P : ℝ × ℝ),
    (O = (0, 0)) →
    (circle_M O.1 O.2) →
    (∃ (cx cy : ℝ), center_curve cx cy ∧ circle_M cx cy) →
    (line_l C.1 C.2) ∧ (line_l D.1 D.2) →
    (circle_M C.1 C.2) ∧ (circle_M D.1 D.2) →
    (Real.sqrt ((C.1 - O.1)^2 + (C.2 - O.2)^2) = Real.sqrt ((D.1 - O.1)^2 + (D.2 - O.2)^2)) →
    (line_sqrt3 E.1 E.2) ∧ (line_sqrt3 F.1 F.2) →
    (circle_M E.1 E.2) ∧ (circle_M F.1 F.2) →
    (line_x5 P.1) →
    (∃ (k b : ℝ), G.2 = k * G.1 + b ∧ H.2 = k * H.1 + b) →
    (circle_M G.1 G.2) ∧ (circle_M H.1 H.2) →
    (∃ (m : ℝ), G.2 - E.2 = m * (G.1 - E.1) ∧ G.2 - P.2 = m * (G.1 - P.1)) →
    (∃ (n : ℝ), H.2 - F.2 = n * (H.1 - F.1) ∧ H.2 - P.2 = n * (H.1 - P.1)) →
    (((E.1 < G.1 ∧ G.1 < F.1) ∧ (F.1 < H.1 ∨ H.1 < E.1)) ∨
     ((E.1 < H.1 ∧ H.1 < F.1) ∧ (F.1 < G.1 ∨ G.1 < E.1))) →
    ∃ (k b : ℝ), G.2 = k * G.1 + b ∧ H.2 = k * H.1 + b ∧ 2 = k * 2 + b ∧ Real.sqrt 3 = k * 2 + b :=
by sorry

end circle_M_fixed_point_l270_27011


namespace unique_steakmaker_pair_l270_27053

/-- A pair of positive integers (m,n) is 'steakmaker' if 1 + 2^m = n^2 -/
def is_steakmaker (m n : ℕ+) : Prop := 1 + 2^(m.val) = n.val^2

theorem unique_steakmaker_pair :
  ∃! (m n : ℕ+), is_steakmaker m n ∧ m.val * n.val = 9 :=
sorry

#check unique_steakmaker_pair

end unique_steakmaker_pair_l270_27053


namespace unfair_coin_probability_l270_27009

def num_flips : ℕ := 10
def num_heads : ℕ := 3
def prob_heads : ℚ := 1/3
def prob_tails : ℚ := 2/3

theorem unfair_coin_probability : 
  (Nat.choose num_flips num_heads : ℚ) * prob_heads ^ num_heads * prob_tails ^ (num_flips - num_heads) = 15360/59049 := by
  sorry

end unfair_coin_probability_l270_27009


namespace complement_intersection_theorem_l270_27003

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def M : Set Nat := {1, 4, 5}
def N : Set Nat := {2, 3}

theorem complement_intersection_theorem :
  (U \ N) ∩ M = {1, 4, 5} := by sorry

end complement_intersection_theorem_l270_27003


namespace math_club_smallest_size_l270_27082

theorem math_club_smallest_size :
  ∀ (total boys girls : ℕ),
    total = boys + girls →
    girls ≥ 2 →
    boys > (91 : ℝ) / 100 * total →
    total ≥ 23 ∧ ∃ (t b g : ℕ), t = 23 ∧ b + g = t ∧ g ≥ 2 ∧ b > (91 : ℝ) / 100 * t :=
by
  sorry

end math_club_smallest_size_l270_27082


namespace smallest_integer_y_smallest_solution_l270_27071

theorem smallest_integer_y (y : ℤ) : (7 - 3 * y ≥ 22) ↔ (y ≤ -5) := by sorry

theorem smallest_solution : ∃ (y : ℤ), (7 - 3 * y ≥ 22) ∧ ∀ (z : ℤ), (7 - 3 * z ≥ 22) → (y ≤ z) := by sorry

end smallest_integer_y_smallest_solution_l270_27071


namespace sheridan_fish_count_l270_27085

/-- Calculate the number of fish Mrs. Sheridan has left -/
def fish_remaining (initial : ℕ) (received : ℕ) (given_away : ℕ) (sold : ℕ) : ℕ :=
  initial + received - given_away - sold

/-- Theorem stating that Mrs. Sheridan has 46 fish left -/
theorem sheridan_fish_count : fish_remaining 22 47 15 8 = 46 := by
  sorry

end sheridan_fish_count_l270_27085


namespace initial_men_count_l270_27058

/-- Represents the amount of food consumed by one person in one day -/
def FoodPerPersonPerDay : ℝ := 1

/-- Calculates the total amount of food -/
def TotalFood (initialMen : ℕ) : ℝ := initialMen * 22 * FoodPerPersonPerDay

/-- Calculates the amount of food consumed in the first two days -/
def FoodConsumedInTwoDays (initialMen : ℕ) : ℝ := initialMen * 2 * FoodPerPersonPerDay

/-- Calculates the remaining food after two days -/
def RemainingFood (initialMen : ℕ) : ℝ := TotalFood initialMen - FoodConsumedInTwoDays initialMen

theorem initial_men_count (initialMen : ℕ) : 
  TotalFood initialMen = initialMen * 22 * FoodPerPersonPerDay ∧
  RemainingFood initialMen = (initialMen + 760) * 10 * FoodPerPersonPerDay →
  initialMen = 760 := by
  sorry

end initial_men_count_l270_27058


namespace rectangle_length_l270_27040

theorem rectangle_length (s : ℝ) (l : ℝ) : 
  s > 0 → l > 0 →
  s^2 = 5 * (l * 10) →
  4 * s = 200 →
  l = 50 := by
sorry

end rectangle_length_l270_27040


namespace inverse_proportional_problem_l270_27068

/-- Given that x and y are inversely proportional, x + y = 36, and x - y = 12,
    prove that when x = 8, y = 36. -/
theorem inverse_proportional_problem (x y : ℝ) (k : ℝ) 
  (h_inverse : x * y = k)
  (h_sum : x + y = 36)
  (h_diff : x - y = 12)
  (h_x : x = 8) : 
  y = 36 := by
sorry

end inverse_proportional_problem_l270_27068


namespace equation_proof_l270_27025

theorem equation_proof : 16 * 0.2 * 5 * 0.5 / 2 = 4 := by
  sorry

end equation_proof_l270_27025


namespace least_value_quadratic_l270_27032

theorem least_value_quadratic (y : ℝ) : 
  (2 * y^2 + 7 * y + 3 = 6) → 
  y ≥ (-7 - Real.sqrt 73) / 4 ∧ 
  ∃ (y_min : ℝ), 2 * y_min^2 + 7 * y_min + 3 = 6 ∧ y_min = (-7 - Real.sqrt 73) / 4 :=
by
  sorry

end least_value_quadratic_l270_27032


namespace triangle_cosine_sum_l270_27063

theorem triangle_cosine_sum (A B C : Real) (a b c : Real) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  (b + c = 12) →  -- Given condition
  (C = 2 * π / 3) →  -- 120° in radians
  (Real.sin B = 5 * Real.sqrt 3 / 14) →
  (Real.cos A + Real.cos B = 12 / 7) := by
  sorry

end triangle_cosine_sum_l270_27063


namespace complex_modulus_sum_l270_27023

theorem complex_modulus_sum : Complex.abs (3 - 5*Complex.I) + Complex.abs (3 + 5*Complex.I) = 2 * Real.sqrt 34 := by
  sorry

end complex_modulus_sum_l270_27023


namespace vector_BC_coordinates_l270_27083

/-- Given points A and B, and vector AC, prove that vector BC has specific coordinates -/
theorem vector_BC_coordinates (A B C : ℝ × ℝ) : 
  A = (0, 1) → B = (3, 2) → C - A = (-4, -3) → C - B = (-7, -4) := by
  sorry

end vector_BC_coordinates_l270_27083


namespace find_other_number_l270_27094

theorem find_other_number (A B : ℕ+) (h1 : A = 24) (h2 : Nat.gcd A B = 15) (h3 : Nat.lcm A B = 312) :
  B = 195 := by
  sorry

end find_other_number_l270_27094


namespace prime_multiple_all_ones_l270_27087

theorem prime_multiple_all_ones (p : ℕ) (hp : Prime p) (hp_not_two : p ≠ 2) (hp_not_five : p ≠ 5) :
  ∃ k : ℕ, ∃ n : ℕ, p * k = 10^n - 1 :=
sorry

end prime_multiple_all_ones_l270_27087


namespace symmetric_point_of_A_l270_27077

/-- A point in 3D Cartesian space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The origin in 3D Cartesian space -/
def origin : Point3D := ⟨0, 0, 0⟩

/-- Point A with coordinates (1, 1, 2) -/
def pointA : Point3D := ⟨1, 1, 2⟩

/-- A point is symmetric to another point with respect to the origin if the origin is the midpoint of the line segment connecting the two points -/
def isSymmetricWrtOrigin (p q : Point3D) : Prop :=
  origin.x = (p.x + q.x) / 2 ∧
  origin.y = (p.y + q.y) / 2 ∧
  origin.z = (p.z + q.z) / 2

/-- The theorem stating that the point symmetric to A(1, 1, 2) with respect to the origin has coordinates (-1, -1, -2) -/
theorem symmetric_point_of_A :
  ∃ (B : Point3D), isSymmetricWrtOrigin pointA B ∧ B = ⟨-1, -1, -2⟩ :=
sorry

end symmetric_point_of_A_l270_27077


namespace insurance_premium_calculation_l270_27016

/-- Calculates the new insurance premium after accidents and tickets. -/
theorem insurance_premium_calculation
  (initial_premium : ℝ)
  (accident_increase_percent : ℝ)
  (ticket_increase : ℝ)
  (num_accidents : ℕ)
  (num_tickets : ℕ)
  (h1 : initial_premium = 50)
  (h2 : accident_increase_percent = 0.1)
  (h3 : ticket_increase = 5)
  (h4 : num_accidents = 1)
  (h5 : num_tickets = 3) :
  initial_premium * (1 + num_accidents * accident_increase_percent) + num_tickets * ticket_increase = 70 :=
by sorry


end insurance_premium_calculation_l270_27016


namespace prob_at_least_one_woman_l270_27060

/-- The probability of selecting at least one woman when randomly choosing 3 people from a group of 5 men and 5 women is 5/6. -/
theorem prob_at_least_one_woman (n_men n_women n_select : ℕ) (h_men : n_men = 5) (h_women : n_women = 5) (h_select : n_select = 3) :
  let total := n_men + n_women
  let prob_no_women := (n_men.choose n_select : ℚ) / (total.choose n_select : ℚ)
  1 - prob_no_women = 5 / 6 := by
  sorry

end prob_at_least_one_woman_l270_27060


namespace dave_ticket_problem_l270_27027

theorem dave_ticket_problem (total_used : ℕ) (difference : ℕ) 
  (h1 : total_used = 12) (h2 : difference = 5) : 
  ∃ (clothes_tickets : ℕ), 
    clothes_tickets + (clothes_tickets + difference) = total_used ∧ 
    clothes_tickets = 7 := by
  sorry

end dave_ticket_problem_l270_27027


namespace similar_terms_and_system_solution_l270_27002

theorem similar_terms_and_system_solution :
  ∀ (m n : ℤ) (a b : ℝ) (x y : ℝ),
    (m - 1 = n - 2*m ∧ m + n = 3*m + n - 4) →
    (m*x + (n-2)*y = 24 ∧ 2*m*x + n*y = 46) →
    (x = 9 ∧ y = 2) := by
  sorry

end similar_terms_and_system_solution_l270_27002


namespace ellipse_axis_endpoint_distance_l270_27054

/-- Given an ellipse with equation 16(x+2)^2 + 4y^2 = 64, 
    the distance between an endpoint of its major axis 
    and an endpoint of its minor axis is 2√5. -/
theorem ellipse_axis_endpoint_distance :
  ∀ (x y : ℝ), 16 * (x + 2)^2 + 4 * y^2 = 64 →
  ∃ (C D : ℝ × ℝ),
    (C.1 + 2)^2 / 4 + C.2^2 / 16 = 1 ∧
    (D.1 + 2)^2 / 4 + D.2^2 / 16 = 1 ∧
    (C.2 = 0 ∨ C.2 = 0) ∧
    (D.1 = -2 ∨ D.1 = -2) ∧
    Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = 2 * Real.sqrt 5 :=
by sorry

end ellipse_axis_endpoint_distance_l270_27054


namespace symposium_partition_exists_l270_27031

/-- Represents a symposium with delegates and their acquaintances. -/
structure Symposium where
  delegates : Finset Nat
  acquainted : Nat → Nat → Prop
  acquainted_symmetric : ∀ a b, acquainted a b ↔ acquainted b a
  acquainted_irreflexive : ∀ a, ¬acquainted a a
  has_acquaintance : ∀ a ∈ delegates, ∃ b ∈ delegates, a ≠ b ∧ acquainted a b
  not_all_acquainted : ∀ a ∈ delegates, ∃ b ∈ delegates, a ≠ b ∧ ¬acquainted a b

/-- Represents a partition of delegates into two groups. -/
structure Partition (s : Symposium) where
  group1 : Finset Nat
  group2 : Finset Nat
  covers : group1 ∪ group2 = s.delegates
  disjoint : group1 ∩ group2 = ∅
  nonempty : group1.Nonempty ∧ group2.Nonempty

/-- The main theorem stating that a valid partition exists for any symposium. -/
theorem symposium_partition_exists (s : Symposium) :
  ∃ p : Partition s, ∀ a ∈ s.delegates,
    (a ∈ p.group1 → ∃ b ∈ p.group1, a ≠ b ∧ s.acquainted a b) ∧
    (a ∈ p.group2 → ∃ b ∈ p.group2, a ≠ b ∧ s.acquainted a b) :=
  sorry

end symposium_partition_exists_l270_27031


namespace max_sum_squared_sum_l270_27039

theorem max_sum_squared_sum (a b c : ℝ) (h : a + b + c = a^2 + b^2 + c^2) :
  a + b + c ≤ 3 ∧ ∃ x y z : ℝ, x + y + z = x^2 + y^2 + z^2 ∧ x + y + z = 3 :=
sorry

end max_sum_squared_sum_l270_27039


namespace calculation_proofs_l270_27012

theorem calculation_proofs :
  (2 * Real.sqrt 12 * (Real.sqrt 3 / 4) / Real.sqrt 2 = (3 * Real.sqrt 2) / 2) ∧
  ((Real.sqrt 3 - Real.sqrt 2)^2 + (Real.sqrt 8 - Real.sqrt 3) * (2 * Real.sqrt 2 + Real.sqrt 3) = 10 - 2 * Real.sqrt 6) := by
  sorry

end calculation_proofs_l270_27012


namespace unique_solution_l270_27005

-- Define the equation
def equation (x : ℝ) : Prop :=
  x ≠ 0 ∧ x ≠ 4 ∧ (3 * x^2 - 12 * x) / (x^2 - 4 * x) = x - 2

-- Theorem statement
theorem unique_solution : ∃! x : ℝ, equation x := by
  sorry

end unique_solution_l270_27005


namespace inequality_solution_set_l270_27097

theorem inequality_solution_set (m n : ℝ) : 
  (∀ x, x^2 - m*x - 6*n < 0 ↔ -3 < x ∧ x < 6) → m + n = 6 := by
  sorry

end inequality_solution_set_l270_27097


namespace arc_length_sector_l270_27055

/-- The arc length of a sector with radius π cm and central angle 2π/3 radians is 2π²/3 cm. -/
theorem arc_length_sector (r : Real) (θ : Real) (l : Real) :
  r = π → θ = 2 * π / 3 → l = θ * r → l = 2 * π^2 / 3 := by
  sorry

#check arc_length_sector

end arc_length_sector_l270_27055


namespace perpendicular_vectors_m_value_l270_27033

/-- Given two vectors OA and OB in 2D space, if they are perpendicular,
    then the second component of OB is 3/2. -/
theorem perpendicular_vectors_m_value (OA OB : ℝ × ℝ) :
  OA = (-1, 2) → OB.1 = 3 → OA.1 * OB.1 + OA.2 * OB.2 = 0 → OB.2 = 3/2 := by
  sorry

end perpendicular_vectors_m_value_l270_27033


namespace thabo_hardcover_books_l270_27010

/-- Represents the number of books Thabo owns in each category -/
structure BookCollection where
  hardcover_nonfiction : ℕ
  paperback_nonfiction : ℕ
  paperback_fiction : ℕ

/-- Thabo's book collection satisfies the given conditions -/
def is_valid_collection (books : BookCollection) : Prop :=
  books.hardcover_nonfiction + books.paperback_nonfiction + books.paperback_fiction = 200 ∧
  books.paperback_nonfiction = books.hardcover_nonfiction + 20 ∧
  books.paperback_fiction = 2 * books.paperback_nonfiction

theorem thabo_hardcover_books :
  ∀ (books : BookCollection), is_valid_collection books → books.hardcover_nonfiction = 35 := by
  sorry

end thabo_hardcover_books_l270_27010


namespace min_y_max_x_l270_27099

theorem min_y_max_x (x y : ℝ) (h : x^2 + y^2 = 18*x + 40*y) : 
  (∀ y' : ℝ, x^2 + y'^2 = 18*x + 40*y' → y ≤ y') ∧ 
  (∀ x' : ℝ, x'^2 + y^2 = 18*x' + 40*y → x' ≤ x) → 
  y = 20 - Real.sqrt 481 ∧ x = 9 + Real.sqrt 481 :=
by sorry

end min_y_max_x_l270_27099


namespace prime_factor_difference_l270_27052

theorem prime_factor_difference (n : Nat) (h : n = 173459) :
  ∃ (p₁ p₂ p₃ p₄ : Nat),
    Nat.Prime p₁ ∧ Nat.Prime p₂ ∧ Nat.Prime p₃ ∧ Nat.Prime p₄ ∧
    n = p₁ * p₂ * p₃ * p₄ ∧
    p₁ ≤ p₂ ∧ p₂ ≤ p₃ ∧ p₃ ≤ p₄ ∧
    p₄ - p₂ = 144 :=
by sorry

end prime_factor_difference_l270_27052


namespace money_difference_specific_money_difference_l270_27034

/-- The difference between Dave's and Derek's remaining money after expenses -/
theorem money_difference (derek_initial : ℕ) (derek_lunch1 : ℕ) (derek_lunch_dad : ℕ) (derek_lunch2 : ℕ)
                         (dave_initial : ℕ) (dave_lunch_mom : ℕ) : ℕ :=
  let derek_spent := derek_lunch1 + derek_lunch_dad + derek_lunch2
  let derek_remaining := derek_initial - derek_spent
  let dave_remaining := dave_initial - dave_lunch_mom
  dave_remaining - derek_remaining

/-- Proof of the specific problem -/
theorem specific_money_difference :
  money_difference 40 14 11 5 50 7 = 33 := by
  sorry

end money_difference_specific_money_difference_l270_27034


namespace count_pairs_eq_12_l270_27093

def count_pairs : ℕ :=
  Finset.sum (Finset.range 3) (fun x =>
    Finset.card (Finset.filter (fun y => x + 1 + y < 7) (Finset.range 6)))

theorem count_pairs_eq_12 : count_pairs = 12 := by
  sorry

end count_pairs_eq_12_l270_27093


namespace billboard_perimeter_l270_27004

/-- A rectangular billboard with given area and width has a specific perimeter -/
theorem billboard_perimeter (area : ℝ) (width : ℝ) (h1 : area = 117) (h2 : width = 9) :
  2 * (area / width) + 2 * width = 44 := by
  sorry

#check billboard_perimeter

end billboard_perimeter_l270_27004


namespace m_range_l270_27019

-- Define the conditions
def P (x : ℝ) : Prop := x^2 - 3*x + 2 > 0
def q (x m : ℝ) : Prop := x < m

-- Define the theorem
theorem m_range (m : ℝ) : 
  (∀ x, ¬(P x) → q x m) ∧ (∃ x, q x m ∧ P x) → m > 2 := by
  sorry

end m_range_l270_27019


namespace not_enough_unique_names_l270_27041

/-- Represents the number of possible occurrences of each letter (a, o, u) in a standardized word -/
def letter_choices : ℕ := 7

/-- Represents the total number of tribe members -/
def tribe_members : ℕ := 400

/-- Represents the number of unique standardized words in the Mumbo-Jumbo language -/
def unique_words : ℕ := letter_choices ^ 3

theorem not_enough_unique_names : unique_words < tribe_members := by
  sorry

end not_enough_unique_names_l270_27041


namespace tangent_line_at_two_condition_equivalent_to_range_l270_27020

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x + (2 * a - 1) / x + 1 - 3 * a

-- Define the tangent line equation
def tangent_line (x y : ℝ) : Prop := 3 * x - 4 * y - 4 = 0

-- Theorem for part I
theorem tangent_line_at_two (a : ℝ) (h : a = 1) :
  ∃ y, f a 2 = y ∧ tangent_line 2 y :=
sorry

-- Theorem for part II
theorem condition_equivalent_to_range (a : ℝ) (h : a > 0) :
  (∀ x : ℝ, x ≥ 1 → f a x ≥ (1 - a) * Real.log x) ↔ a ≥ 1/3 :=
sorry

end

end tangent_line_at_two_condition_equivalent_to_range_l270_27020


namespace dice_probability_l270_27046

/-- The number of sides on each die -/
def num_sides : ℕ := 12

/-- The number of dice rolled -/
def num_dice : ℕ := 5

/-- The number of one-digit outcomes on each die -/
def one_digit_outcomes : ℕ := 9

/-- The number of two-digit outcomes on each die -/
def two_digit_outcomes : ℕ := num_sides - one_digit_outcomes

/-- The probability of rolling a one-digit number on a single die -/
def prob_one_digit : ℚ := one_digit_outcomes / num_sides

/-- The probability of rolling a two-digit number on a single die -/
def prob_two_digit : ℚ := two_digit_outcomes / num_sides

/-- The number of dice showing one-digit numbers -/
def num_one_digit : ℕ := 3

/-- The number of dice showing two-digit numbers -/
def num_two_digit : ℕ := num_dice - num_one_digit

theorem dice_probability :
  (Nat.choose num_dice num_one_digit : ℚ) *
  (prob_one_digit ^ num_one_digit) *
  (prob_two_digit ^ num_two_digit) =
  135 / 512 := by sorry

end dice_probability_l270_27046


namespace real_roots_condition_specific_roots_condition_l270_27021

variable (m : ℝ)
variable (x₁ x₂ : ℝ)

-- Define the quadratic equation
def quadratic (x : ℝ) := x^2 - 6*x + (4*m + 1)

-- Theorem 1: For real roots, m ≤ 2
theorem real_roots_condition : (∃ x : ℝ, quadratic m x = 0) → m ≤ 2 := by sorry

-- Theorem 2: If x₁ and x₂ are roots and x₁² + x₂² = 26, then m = 1
theorem specific_roots_condition : 
  quadratic m x₁ = 0 → quadratic m x₂ = 0 → x₁^2 + x₂^2 = 26 → m = 1 := by sorry

end real_roots_condition_specific_roots_condition_l270_27021


namespace sotka_not_divisible_by_nine_l270_27028

/-- Represents a mapping of letters to digits -/
def LetterMapping := Char → Nat

/-- Checks if a number represented by a string is divisible by a given number -/
def isDivisible (s : String) (n : Nat) (mapping : LetterMapping) : Prop :=
  (s.toList.map mapping).sum % n = 0

/-- Ensures that each letter maps to a unique digit between 0 and 9 -/
def isValidMapping (mapping : LetterMapping) : Prop :=
  ∀ c₁ c₂, c₁ ≠ c₂ → mapping c₁ ≠ mapping c₂ ∧ mapping c₁ < 10 ∧ mapping c₂ < 10

theorem sotka_not_divisible_by_nine :
  ∀ mapping : LetterMapping,
    isValidMapping mapping →
    isDivisible "ДЕВЯНОСТО" 90 mapping →
    isDivisible "ДЕВЯТКА" 9 mapping →
    mapping 'О' = 0 →
    ¬ isDivisible "СОТКА" 9 mapping :=
by
  sorry

end sotka_not_divisible_by_nine_l270_27028


namespace reduced_price_is_six_l270_27062

/-- Represents the price of apples and the quantity that can be purchased -/
structure ApplePricing where
  originalPrice : ℚ
  quantityBefore : ℚ
  quantityAfter : ℚ

/-- Calculates the reduced price per dozen apples -/
def reducedPricePerDozen (ap : ApplePricing) : ℚ :=
  6

/-- Theorem stating the reduced price per dozen apples is 6 rupees -/
theorem reduced_price_is_six (ap : ApplePricing) 
  (h1 : ap.quantityAfter = ap.quantityBefore + 50)
  (h2 : ap.quantityBefore * ap.originalPrice = 50)
  (h3 : ap.quantityAfter * (ap.originalPrice / 2) = 50) : 
  reducedPricePerDozen ap = 6 := by
  sorry

#check reduced_price_is_six

end reduced_price_is_six_l270_27062


namespace sum_of_n_terms_l270_27050

/-- Given an arithmetic sequence {a_n} with sum of first n terms S_n -/
def arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = n * a 1 + (n * (n - 1) / 2) * (a 2 - a 1)

/-- S_1, S_3, and S_2 form an arithmetic sequence -/
def S_arithmetic (S : ℕ → ℝ) : Prop :=
  S 3 - S 2 = S 2 - S 1

/-- a_1 - a_3 = 3 -/
def a_difference (a : ℕ → ℝ) : Prop :=
  a 1 - a 3 = 3

/-- Theorem: Sum of first n terms of the sequence -/
theorem sum_of_n_terms
  (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : arithmetic_sequence a S)
  (h2 : S_arithmetic S)
  (h3 : a_difference a) :
  ∀ n, S n = (8/3) * (1 - (-1/2)^n) :=
sorry

end sum_of_n_terms_l270_27050


namespace total_bags_delivered_l270_27056

-- Define the problem parameters
def bags_per_trip_light : ℕ := 15
def bags_per_trip_heavy : ℕ := 20
def total_days : ℕ := 7
def trips_per_day_light : ℕ := 25
def trips_per_day_heavy : ℕ := 18
def days_with_light_bags : ℕ := 3
def days_with_heavy_bags : ℕ := 4

-- Define the theorem
theorem total_bags_delivered : 
  (days_with_light_bags * trips_per_day_light * bags_per_trip_light) +
  (days_with_heavy_bags * trips_per_day_heavy * bags_per_trip_heavy) = 2565 :=
by sorry

end total_bags_delivered_l270_27056


namespace peaches_per_basket_l270_27038

/-- The number of red peaches in each basket -/
def red_peaches : ℕ := 7

/-- The number of green peaches in each basket -/
def green_peaches : ℕ := 3

/-- The total number of peaches in each basket -/
def total_peaches : ℕ := red_peaches + green_peaches

theorem peaches_per_basket : total_peaches = 10 := by
  sorry

end peaches_per_basket_l270_27038


namespace tree_spacing_l270_27029

/-- Given a yard of length 225 meters with 26 trees planted at equal distances,
    including one tree at each end, the distance between two consecutive trees is 9 meters. -/
theorem tree_spacing (yard_length : ℝ) (num_trees : ℕ) (tree_spacing : ℝ) : 
  yard_length = 225 →
  num_trees = 26 →
  tree_spacing * (num_trees - 1) = yard_length →
  tree_spacing = 9 := by sorry

end tree_spacing_l270_27029


namespace cricket_team_size_l270_27007

/-- Represents the number of players on a cricket team -/
def total_players : ℕ := 61

/-- Represents the number of throwers on the team -/
def throwers : ℕ := 37

/-- Represents the number of right-handed players on the team -/
def right_handed : ℕ := 53

/-- Theorem stating that the total number of players is 61 -/
theorem cricket_team_size :
  total_players = throwers + (right_handed - throwers) * 3 / 2 :=
by sorry

end cricket_team_size_l270_27007


namespace probability_one_of_each_type_l270_27090

def total_silverware : ℕ := 30
def forks : ℕ := 10
def spoons : ℕ := 10
def knives : ℕ := 10

theorem probability_one_of_each_type (total_silverware forks spoons knives : ℕ) :
  total_silverware = forks + spoons + knives →
  (Nat.choose total_silverware 3 : ℚ) ≠ 0 →
  (forks * spoons * knives : ℚ) / Nat.choose total_silverware 3 = 500 / 203 := by
  sorry

end probability_one_of_each_type_l270_27090


namespace only_math_scores_need_census_l270_27098

-- Define the survey types
inductive SurveyType
  | Sampling
  | Census

-- Define the survey options
inductive SurveyOption
  | WeeklyAllowance
  | MathTestScores
  | TVWatchTime
  | ExtracurricularReading

-- Function to determine the appropriate survey type for each option
def appropriateSurveyType (option : SurveyOption) : SurveyType :=
  match option with
  | SurveyOption.MathTestScores => SurveyType.Census
  | _ => SurveyType.Sampling

-- Theorem stating that only the MathTestScores option requires a census
theorem only_math_scores_need_census :
  ∀ (option : SurveyOption),
    appropriateSurveyType option = SurveyType.Census ↔ option = SurveyOption.MathTestScores :=
by sorry


end only_math_scores_need_census_l270_27098


namespace running_time_difference_l270_27015

/-- The difference in running time between two runners -/
theorem running_time_difference
  (d : ℝ) -- Total distance
  (lawrence_distance : ℝ) -- Lawrence's running distance
  (lawrence_speed : ℝ) -- Lawrence's speed in minutes per kilometer
  (george_distance : ℝ) -- George's running distance
  (george_speed : ℝ) -- George's speed in minutes per kilometer
  (h1 : lawrence_distance = d / 2)
  (h2 : george_distance = d / 2)
  (h3 : lawrence_speed = 8)
  (h4 : george_speed = 12) :
  george_distance * george_speed - lawrence_distance * lawrence_speed = 2 * d :=
by sorry

end running_time_difference_l270_27015


namespace system_solution_l270_27035

/- Define the system of equations -/
def equation1 (x y : ℚ) : Prop := 4 * x - 7 * y = -14
def equation2 (x y : ℚ) : Prop := 5 * x + 3 * y = -7

/- Define the solution -/
def solution_x : ℚ := -91/47
def solution_y : ℚ := -42/47

/- Theorem statement -/
theorem system_solution :
  equation1 solution_x solution_y ∧ equation2 solution_x solution_y :=
by sorry

end system_solution_l270_27035


namespace partnership_profit_theorem_l270_27026

/-- Represents the profit distribution in a partnership business -/
structure PartnershipProfit where
  investment_A : ℕ
  investment_B : ℕ
  investment_C : ℕ
  profit_share_C : ℕ

/-- Calculates the total profit of the partnership -/
def total_profit (p : PartnershipProfit) : ℕ :=
  (p.profit_share_C * (p.investment_A + p.investment_B + p.investment_C)) / p.investment_C

/-- Theorem stating that given the investments and C's profit share, the total profit is 80000 -/
theorem partnership_profit_theorem (p : PartnershipProfit) 
  (h1 : p.investment_A = 27000)
  (h2 : p.investment_B = 72000)
  (h3 : p.investment_C = 81000)
  (h4 : p.profit_share_C = 36000) :
  total_profit p = 80000 := by
  sorry

#eval total_profit { investment_A := 27000, investment_B := 72000, investment_C := 81000, profit_share_C := 36000 }

end partnership_profit_theorem_l270_27026


namespace lcm_24_36_45_l270_27030

theorem lcm_24_36_45 : Nat.lcm 24 (Nat.lcm 36 45) = 360 := by
  sorry

end lcm_24_36_45_l270_27030


namespace binary_110011_equals_51_l270_27049

/-- Converts a list of binary digits to its decimal representation -/
def binary_to_decimal (bits : List Bool) : Nat :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of the number we want to convert -/
def binary_number : List Bool := [true, true, false, false, true, true]

/-- Theorem stating that the binary number 110011 is equal to the decimal number 51 -/
theorem binary_110011_equals_51 :
  binary_to_decimal (binary_number.reverse) = 51 := by
  sorry

end binary_110011_equals_51_l270_27049


namespace quadratic_inequality_solution_set_l270_27006

theorem quadratic_inequality_solution_set 
  (a b : ℝ) 
  (h : Set.Ioo (-3 : ℝ) 2 = {x : ℝ | a * x^2 - 5*x + b > 0}) :
  {x : ℝ | b * x^2 - 5*x + a > 0} = Set.Iic (-1/3 : ℝ) ∪ Set.Ici (1/2 : ℝ) :=
by sorry

end quadratic_inequality_solution_set_l270_27006


namespace a_minus_b_value_l270_27000

theorem a_minus_b_value (a b : ℝ) (h1 : |a| = 8) (h2 : |b| = 6) (h3 : a * b < 0) :
  a - b = 14 ∨ a - b = -14 := by
  sorry

end a_minus_b_value_l270_27000


namespace josh_initial_marbles_l270_27042

/-- The number of marbles Josh found -/
def marbles_found : ℕ := 7

/-- The current total number of marbles Josh has -/
def current_total : ℕ := 28

/-- The initial number of marbles in Josh's collection -/
def initial_marbles : ℕ := current_total - marbles_found

theorem josh_initial_marbles :
  initial_marbles = 21 := by sorry

end josh_initial_marbles_l270_27042


namespace sum_and_transformations_l270_27079

theorem sum_and_transformations (x y z M : ℚ) : 
  x + y + z = 72 ∧ 
  x - 9 = M ∧ 
  y + 9 = M ∧ 
  9 * z = M → 
  M = 34 := by
sorry

end sum_and_transformations_l270_27079


namespace farmer_land_area_l270_27008

theorem farmer_land_area (A : ℝ) 
  (h1 : 0.9 * A * 0.1 = 360) 
  (h2 : 0.9 * A * 0.6 + 0.9 * A * 0.3 + 360 = 0.9 * A) : 
  A = 4000 := by
  sorry

end farmer_land_area_l270_27008


namespace product_plus_one_is_perfect_square_l270_27069

theorem product_plus_one_is_perfect_square (n m : ℤ) : 
  m - n = 2 → ∃ k : ℤ, n * m + 1 = k^2 := by sorry

end product_plus_one_is_perfect_square_l270_27069


namespace complex_subtraction_l270_27065

theorem complex_subtraction (c d : ℂ) (h1 : c = 5 - 3*I) (h2 : d = 2 - I) : 
  c - 3*d = -1 := by
  sorry

end complex_subtraction_l270_27065


namespace bernoulli_inequality_l270_27080

theorem bernoulli_inequality (n : ℕ+) (x : ℝ) (h : x > -1) :
  (1 + x)^(n : ℝ) ≥ 1 + n * x := by
  sorry

end bernoulli_inequality_l270_27080


namespace inequality_proof_l270_27017

theorem inequality_proof (x y z : ℝ) : x^2 + y^2 + z^2 ≥ Real.sqrt 2 * (x*y + y*z) := by
  sorry

end inequality_proof_l270_27017


namespace lateral_surface_area_is_four_l270_27070

/-- A regular quadrilateral pyramid inscribed in a unit sphere -/
structure RegularQuadPyramid where
  /-- The radius of the sphere in which the pyramid is inscribed -/
  radius : ℝ
  /-- The dihedral angle at the apex of the pyramid in radians -/
  dihedral_angle : ℝ
  /-- Assertion that the radius is 1 -/
  radius_is_one : radius = 1
  /-- Assertion that the dihedral angle is π/4 (45 degrees) -/
  angle_is_45 : dihedral_angle = Real.pi / 4

/-- The lateral surface area of a regular quadrilateral pyramid -/
def lateral_surface_area (p : RegularQuadPyramid) : ℝ :=
  sorry

/-- Theorem: The lateral surface area of the specified pyramid is 4 -/
theorem lateral_surface_area_is_four (p : RegularQuadPyramid) :
  lateral_surface_area p = 4 := by
  sorry

end lateral_surface_area_is_four_l270_27070


namespace class_size_l270_27074

/-- The number of girls in Tom's class -/
def girls : ℕ := 22

/-- The difference between the number of girls and boys in Tom's class -/
def difference : ℕ := 3

/-- The total number of students in Tom's class -/
def total_students : ℕ := girls + (girls - difference)

theorem class_size : total_students = 41 := by
  sorry

end class_size_l270_27074


namespace phase_shift_cosine_l270_27036

theorem phase_shift_cosine (x : Real) :
  let f : Real → Real := fun x ↦ 5 * Real.cos (x - π/3 + π/6)
  (∃ (k : Real), ∀ x, f x = 5 * Real.cos (x - k)) ∧
  (∀ k : Real, (∀ x, f x = 5 * Real.cos (x - k)) → k = π/6) :=
by sorry

end phase_shift_cosine_l270_27036


namespace fraction_of_72_l270_27091

theorem fraction_of_72 : (1 / 3 : ℚ) * (1 / 4 : ℚ) * (1 / 6 : ℚ) * 72 = 1 := by
  sorry

end fraction_of_72_l270_27091


namespace find_divisor_l270_27095

theorem find_divisor (divisor : ℕ) : 
  (127 / divisor = 9) ∧ (127 % divisor = 1) → divisor = 14 := by
sorry

end find_divisor_l270_27095


namespace digit_sum_unbounded_l270_27037

/-- Sum of digits of a positive integer -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Sequence of sum of digits of a^n -/
def digitSumSequence (a : ℕ) (n : ℕ) : ℕ := sumOfDigits (a^n)

theorem digit_sum_unbounded (a : ℕ) (h1 : Even a) (h2 : ¬(5 ∣ a)) :
  ∀ M : ℕ, ∃ N : ℕ, ∀ n ≥ N, digitSumSequence a n > M :=
sorry

end digit_sum_unbounded_l270_27037


namespace least_number_with_remainder_l270_27013

theorem least_number_with_remainder (n : ℕ) : n = 184 ↔
  n > 0 ∧
  n % 5 = 4 ∧
  n % 9 = 4 ∧
  n % 12 = 4 ∧
  n % 18 = 4 ∧
  ∀ m : ℕ, m > 0 →
    m % 5 = 4 →
    m % 9 = 4 →
    m % 12 = 4 →
    m % 18 = 4 →
    n ≤ m :=
by sorry

end least_number_with_remainder_l270_27013


namespace correct_mark_l270_27088

theorem correct_mark (wrong_mark : ℕ) (class_size : ℕ) (average_increase : ℚ) 
  (h1 : wrong_mark = 79)
  (h2 : class_size = 68)
  (h3 : average_increase = 1/2) : 
  ∃ (correct_mark : ℕ), 
    (wrong_mark : ℚ) - correct_mark = average_increase * class_size ∧ 
    correct_mark = 45 := by
  sorry

end correct_mark_l270_27088


namespace solve_system_l270_27066

theorem solve_system (a b : ℝ) 
  (eq1 : a * (a - 4) = 5)
  (eq2 : b * (b - 4) = 5)
  (neq : a ≠ b)
  (sum : a + b = 4) :
  a = -1 := by sorry

end solve_system_l270_27066


namespace sum_of_parts_l270_27059

/-- Given a number 24 divided into two parts, where the first part is 13.0,
    prove that the sum of 7 times the first part and 5 times the second part is 146. -/
theorem sum_of_parts (first_part second_part : ℝ) : 
  first_part + second_part = 24 →
  first_part = 13 →
  7 * first_part + 5 * second_part = 146 := by
sorry

end sum_of_parts_l270_27059


namespace square_area_increase_l270_27048

theorem square_area_increase (s : ℝ) (h : s > 0) : 
  let new_side := 1.25 * s
  let original_area := s^2
  let new_area := new_side^2
  (new_area - original_area) / original_area = 0.5625 := by
sorry

end square_area_increase_l270_27048


namespace constant_term_expansion_l270_27064

def binomialCoefficient (n k : ℕ) : ℕ := sorry

def constantTermInExpansion (n : ℕ) : ℤ :=
  (binomialCoefficient n (n - 2)) * ((-2) ^ 2)

theorem constant_term_expansion :
  constantTermInExpansion 6 = 60 := by sorry

end constant_term_expansion_l270_27064


namespace tan_alpha_equals_negative_one_l270_27072

theorem tan_alpha_equals_negative_one (α : Real) 
  (h1 : |Real.sin α| = |Real.cos α|) 
  (h2 : α > Real.pi / 2 ∧ α < Real.pi) : 
  Real.tan α = -1 := by
  sorry

end tan_alpha_equals_negative_one_l270_27072


namespace earliest_100_degrees_l270_27018

-- Define the temperature function
def temperature (t : ℝ) : ℝ := -t^2 + 15*t + 40

-- State the theorem
theorem earliest_100_degrees :
  ∃ t : ℝ, t ≥ 0 ∧ temperature t = 100 ∧ ∀ s, s ≥ 0 ∧ temperature s = 100 → s ≥ t :=
by
  -- The proof goes here
  sorry

end earliest_100_degrees_l270_27018


namespace odd_function_range_l270_27078

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def range (f : ℝ → ℝ) : Set ℝ := {y | ∃ x, f x = y}

theorem odd_function_range (f : ℝ → ℝ) (h_odd : is_odd f) (h_pos : ∀ x > 0, f x = 2) :
  range f = {-2, 0, 2} := by
  sorry

end odd_function_range_l270_27078


namespace ellipse_area_lower_bound_l270_27022

/-- Given a right-angled triangle with area t, where the endpoints of its hypotenuse
    lie at the foci of an ellipse and the third vertex lies on the ellipse,
    the area of the ellipse is at least √2πt. -/
theorem ellipse_area_lower_bound (t : ℝ) (a b c : ℝ) (h1 : 0 < t) (h2 : 0 < b) (h3 : b < a)
    (h4 : a^2 = b^2 + c^2) (h5 : t = b^2) : π * a * b ≥ Real.sqrt 2 * π * t :=
by sorry

end ellipse_area_lower_bound_l270_27022


namespace find_divisor_l270_27014

theorem find_divisor : ∃ (d : ℕ), d > 1 ∧ 
  (3198 + 2) % d = 0 ∧ 
  3198 % d ≠ 0 ∧ 
  ∀ (k : ℕ), k > 1 → (3198 + 2) % k = 0 → 3198 % k ≠ 0 → d ≤ k :=
by sorry

end find_divisor_l270_27014


namespace max_blocks_fit_l270_27044

/-- Represents the dimensions of a rectangular solid -/
structure Dimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a rectangular solid given its dimensions -/
def volume (d : Dimensions) : ℕ :=
  d.length * d.width * d.height

/-- The dimensions of the box -/
def box : Dimensions :=
  { length := 4, width := 3, height := 3 }

/-- The dimensions of a block -/
def block : Dimensions :=
  { length := 2, width := 3, height := 1 }

/-- The maximum number of blocks that can fit in the box -/
def max_blocks : ℕ :=
  volume box / volume block

theorem max_blocks_fit :
  max_blocks = 6 ∧
  (∀ n : ℕ, n > max_blocks → ¬ (n * volume block ≤ volume box)) :=
by sorry

end max_blocks_fit_l270_27044


namespace other_number_proof_l270_27001

theorem other_number_proof (a b : ℕ+) 
  (hcf : Nat.gcd a b = 14)
  (lcm : Nat.lcm a b = 396)
  (ha : a = 36) :
  b = 66 := by
  sorry

end other_number_proof_l270_27001


namespace geometric_sequence_sum_l270_27043

/-- Given a geometric sequence {a_n} with first term a and common ratio q,
    if a_2 * a_5 = 2 * a_3 and (a_4 + 2 * a_7) / 2 = 5/4,
    then the sum of the first 5 terms (S_5) is equal to 31. -/
theorem geometric_sequence_sum (a q : ℝ) : 
  (a * q * (a * q^4) = 2 * (a * q^2)) →
  ((a * q^3 + 2 * (a * q^6)) / 2 = 5/4) →
  (a * (1 - q^5)) / (1 - q) = 31 :=
by sorry

end geometric_sequence_sum_l270_27043


namespace cos_pi_sixth_plus_alpha_l270_27061

theorem cos_pi_sixth_plus_alpha (α : ℝ) (h : Real.sin (π / 3 - α) = 1 / 4) :
  Real.cos (π / 6 + α) = 1 / 4 := by
  sorry

end cos_pi_sixth_plus_alpha_l270_27061


namespace num_factors_48_mult_6_eq_4_l270_27089

/-- The number of positive factors of 48 that are also multiples of 6 -/
def num_factors_48_mult_6 : ℕ :=
  (Finset.filter (λ x => x ∣ 48 ∧ 6 ∣ x) (Finset.range 49)).card

/-- Theorem stating that the number of positive factors of 48 that are also multiples of 6 is 4 -/
theorem num_factors_48_mult_6_eq_4 : num_factors_48_mult_6 = 4 := by
  sorry

end num_factors_48_mult_6_eq_4_l270_27089


namespace rectangular_field_dimension_l270_27073

theorem rectangular_field_dimension (m : ℝ) : ∃! m : ℝ, (3*m + 5)*(m - 1) = 104 ∧ m > 1 := by
  sorry

end rectangular_field_dimension_l270_27073


namespace lila_seventh_l270_27024

/-- Represents the finishing position of a racer -/
def Position : Type := Fin 12

/-- Represents a racer in the race -/
structure Racer :=
  (name : String)
  (position : Position)

/-- The race with given conditions -/
structure Race :=
  (racers : List Racer)
  (jessica_behind_esther : ∃ (j e : Racer), j.name = "Jessica" ∧ e.name = "Esther" ∧ j.position.val = e.position.val + 7)
  (ivan_behind_noel : ∃ (i n : Racer), i.name = "Ivan" ∧ n.name = "Noel" ∧ i.position.val = n.position.val + 2)
  (lila_behind_esther : ∃ (l e : Racer), l.name = "Lila" ∧ e.name = "Esther" ∧ l.position.val = e.position.val + 4)
  (noel_behind_omar : ∃ (n o : Racer), n.name = "Noel" ∧ o.name = "Omar" ∧ n.position.val = o.position.val + 4)
  (omar_behind_esther : ∃ (o e : Racer), o.name = "Omar" ∧ e.name = "Esther" ∧ o.position.val = e.position.val + 3)
  (ivan_fourth : ∃ (i : Racer), i.name = "Ivan" ∧ i.position.val = 4)

/-- Theorem stating that Lila finished in 7th place -/
theorem lila_seventh (race : Race) : ∃ (l : Racer), l.name = "Lila" ∧ l.position.val = 7 := by
  sorry

end lila_seventh_l270_27024


namespace lily_to_rose_ratio_l270_27076

def number_of_roses : ℕ := 20
def cost_of_rose : ℕ := 5
def total_spent : ℕ := 250

theorem lily_to_rose_ratio :
  let cost_of_lily : ℕ := 2 * cost_of_rose
  let total_spent_on_roses : ℕ := number_of_roses * cost_of_rose
  let total_spent_on_lilies : ℕ := total_spent - total_spent_on_roses
  let number_of_lilies : ℕ := total_spent_on_lilies / cost_of_lily
  (number_of_lilies : ℚ) / (number_of_roses : ℚ) = 3 / 4 :=
by
  sorry

end lily_to_rose_ratio_l270_27076


namespace probability_two_red_balls_probability_two_red_balls_is_5_22_l270_27051

/-- The probability of picking two red balls from a bag containing 6 red balls, 4 blue balls, and 2 green balls when 2 balls are picked at random -/
theorem probability_two_red_balls : ℚ :=
  let total_balls : ℕ := 6 + 4 + 2
  let red_balls : ℕ := 6
  let prob_first_red : ℚ := red_balls / total_balls
  let prob_second_red : ℚ := (red_balls - 1) / (total_balls - 1)
  prob_first_red * prob_second_red

/-- Proof that the probability of picking two red balls is 5/22 -/
theorem probability_two_red_balls_is_5_22 : 
  probability_two_red_balls = 5 / 22 := by
  sorry

end probability_two_red_balls_probability_two_red_balls_is_5_22_l270_27051


namespace borrowed_amount_l270_27096

/-- Represents the financial transaction described in the problem -/
structure Transaction where
  amount : ℝ  -- The amount borrowed/lent
  borrowRate : ℝ  -- Borrowing interest rate (as a decimal)
  lendRate : ℝ  -- Lending interest rate (as a decimal)
  years : ℝ  -- Duration of the transaction in years
  yearlyGain : ℝ  -- Gain per year

/-- Calculates the total gain over the entire period -/
def totalGain (t : Transaction) : ℝ :=
  (t.lendRate - t.borrowRate) * t.amount * t.years

/-- The main theorem that proves the borrowed amount given the conditions -/
theorem borrowed_amount (t : Transaction) 
    (h1 : t.years = 2)
    (h2 : t.borrowRate = 0.04)
    (h3 : t.lendRate = 0.06)
    (h4 : t.yearlyGain = 80) :
    t.amount = 2000 := by
  sorry

#check borrowed_amount

end borrowed_amount_l270_27096


namespace cloth_cost_calculation_l270_27081

/-- The total cost of cloth given its length and price per meter -/
def totalCost (length : ℝ) (pricePerMeter : ℝ) : ℝ :=
  length * pricePerMeter

/-- Theorem: The total cost of 9.25 meters of cloth at $47 per meter is $434.75 -/
theorem cloth_cost_calculation :
  totalCost 9.25 47 = 434.75 := by
  sorry

end cloth_cost_calculation_l270_27081


namespace dividend_calculation_l270_27086

/-- Calculates the dividend received from an investment in shares -/
theorem dividend_calculation (investment : ℝ) (share_value : ℝ) (premium_rate : ℝ) (dividend_rate : ℝ)
  (h1 : investment = 14400)
  (h2 : share_value = 100)
  (h3 : premium_rate = 0.2)
  (h4 : dividend_rate = 0.05) :
  let actual_share_price := share_value * (1 + premium_rate)
  let num_shares := investment / actual_share_price
  let dividend_per_share := share_value * dividend_rate
  dividend_per_share * num_shares = 600 := by
  sorry

end dividend_calculation_l270_27086


namespace fraction_to_zero_power_is_one_l270_27045

theorem fraction_to_zero_power_is_one (a b : ℤ) (hb : b ≠ 0) : (a / b : ℚ) ^ (0 : ℕ) = 1 := by
  sorry

end fraction_to_zero_power_is_one_l270_27045


namespace geometric_sequence_fourth_term_l270_27092

/-- Given a geometric sequence {a_n} with common ratio 2 and a_1 * a_3 = 6 * a_2, prove that a_4 = 24 -/
theorem geometric_sequence_fourth_term (a : ℕ → ℝ) :
  (∀ n, a (n + 1) = 2 * a n) →  -- geometric sequence with common ratio 2
  a 1 * a 3 = 6 * a 2 →         -- given condition
  a 4 = 24 := by
sorry

end geometric_sequence_fourth_term_l270_27092


namespace percentage_between_55_and_65_l270_27075

/-- Represents the percentage of students who scored at least 55% on the test -/
def scored_at_least_55 : ℝ := 55

/-- Represents the percentage of students who scored at most 65% on the test -/
def scored_at_most_65 : ℝ := 65

/-- Represents the percentage of students who scored between 55% and 65% (inclusive) on the test -/
def scored_between_55_and_65 : ℝ := scored_at_most_65 - (100 - scored_at_least_55)

theorem percentage_between_55_and_65 : scored_between_55_and_65 = 20 := by
  sorry

end percentage_between_55_and_65_l270_27075


namespace remaining_cooking_time_l270_27057

def total_potatoes : ℕ := 15
def cooked_potatoes : ℕ := 8
def cooking_time_per_potato : ℕ := 9

theorem remaining_cooking_time : 
  (total_potatoes - cooked_potatoes) * cooking_time_per_potato = 63 := by
  sorry

end remaining_cooking_time_l270_27057


namespace m_range_proof_l270_27047

theorem m_range_proof (m : ℝ) : 
  (∀ x : ℝ, (4 * x - m < 0 → -1 ≤ x ∧ x ≤ 2) ∧ 
  (∃ y : ℝ, -1 ≤ y ∧ y ≤ 2 ∧ ¬(4 * y - m < 0))) → 
  m > 8 := by
sorry

end m_range_proof_l270_27047
