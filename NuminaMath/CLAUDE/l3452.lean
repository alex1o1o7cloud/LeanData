import Mathlib

namespace NUMINAMATH_CALUDE_car_distance_proof_l3452_345275

/-- Proves that the initial distance covered by a car is 180 km, given the conditions of the problem. -/
theorem car_distance_proof (initial_time : ℝ) (new_speed : ℝ) :
  initial_time = 6 →
  new_speed = 20 →
  ∃ (D : ℝ),
    D = new_speed * (3/2 * initial_time) ∧
    D = 180 :=
by
  sorry

#check car_distance_proof

end NUMINAMATH_CALUDE_car_distance_proof_l3452_345275


namespace NUMINAMATH_CALUDE_students_correct_both_experiments_l3452_345298

/-- Given a group of students performing physics and chemistry experiments, 
    calculate the number of students who conducted both experiments correctly. -/
theorem students_correct_both_experiments 
  (total : ℕ) 
  (physics_correct : ℕ) 
  (chemistry_correct : ℕ) 
  (both_incorrect : ℕ) 
  (h1 : total = 50)
  (h2 : physics_correct = 40)
  (h3 : chemistry_correct = 31)
  (h4 : both_incorrect = 5) :
  physics_correct + chemistry_correct + both_incorrect - total = 26 := by
  sorry

#eval 40 + 31 + 5 - 50  -- Should output 26

end NUMINAMATH_CALUDE_students_correct_both_experiments_l3452_345298


namespace NUMINAMATH_CALUDE_multiple_properties_l3452_345216

-- Define x and y as integers
variable (x y : ℤ)

-- Define the conditions
variable (h1 : ∃ k : ℤ, x = 8 * k)
variable (h2 : ∃ m : ℤ, y = 12 * m)

-- Theorem to prove
theorem multiple_properties :
  (∃ n : ℤ, y = 4 * n) ∧ (∃ p : ℤ, x - y = 4 * p) :=
by sorry

end NUMINAMATH_CALUDE_multiple_properties_l3452_345216


namespace NUMINAMATH_CALUDE_tree_height_problem_l3452_345294

theorem tree_height_problem (h₁ h₂ : ℝ) : 
  h₁ = h₂ + 24 →  -- One tree is 24 feet taller than the other
  h₂ / h₁ = 2 / 3 →  -- The heights are in the ratio 2:3
  h₁ = 72 :=  -- The height of the taller tree is 72 feet
by
  sorry

end NUMINAMATH_CALUDE_tree_height_problem_l3452_345294


namespace NUMINAMATH_CALUDE_small_poster_price_is_six_l3452_345288

/-- Represents Laran's poster business --/
structure PosterBusiness where
  total_posters_per_day : ℕ
  large_posters_per_day : ℕ
  large_poster_price : ℕ
  large_poster_cost : ℕ
  small_poster_cost : ℕ
  weekly_profit : ℕ
  days_per_week : ℕ

/-- Calculates the selling price of small posters --/
def small_poster_price (business : PosterBusiness) : ℕ :=
  let small_posters_per_day := business.total_posters_per_day - business.large_posters_per_day
  let daily_profit := business.weekly_profit / business.days_per_week
  let large_poster_profit := business.large_poster_price - business.large_poster_cost
  let daily_large_poster_profit := large_poster_profit * business.large_posters_per_day
  let daily_small_poster_profit := daily_profit - daily_large_poster_profit
  let small_poster_profit := daily_small_poster_profit / small_posters_per_day
  small_poster_profit + business.small_poster_cost

/-- Theorem stating that the small poster price is $6 --/
theorem small_poster_price_is_six (business : PosterBusiness) 
    (h1 : business.total_posters_per_day = 5)
    (h2 : business.large_posters_per_day = 2)
    (h3 : business.large_poster_price = 10)
    (h4 : business.large_poster_cost = 5)
    (h5 : business.small_poster_cost = 3)
    (h6 : business.weekly_profit = 95)
    (h7 : business.days_per_week = 5) :
  small_poster_price business = 6 := by
  sorry

#eval small_poster_price {
  total_posters_per_day := 5,
  large_posters_per_day := 2,
  large_poster_price := 10,
  large_poster_cost := 5,
  small_poster_cost := 3,
  weekly_profit := 95,
  days_per_week := 5
}

end NUMINAMATH_CALUDE_small_poster_price_is_six_l3452_345288


namespace NUMINAMATH_CALUDE_ladybugs_without_spots_l3452_345232

theorem ladybugs_without_spots (total : Nat) (with_spots : Nat) (without_spots : Nat) : 
  total = 67082 → with_spots = 12170 → without_spots = total - with_spots → without_spots = 54912 := by
  sorry

end NUMINAMATH_CALUDE_ladybugs_without_spots_l3452_345232


namespace NUMINAMATH_CALUDE_number_of_diagonals_sum_of_interior_angles_l3452_345267

-- Define the number of sides
def n : ℕ := 150

-- Theorem for the number of diagonals
theorem number_of_diagonals : 
  n * (n - 3) / 2 = 11025 :=
sorry

-- Theorem for the sum of interior angles
theorem sum_of_interior_angles : 
  180 * (n - 2) = 26640 :=
sorry

end NUMINAMATH_CALUDE_number_of_diagonals_sum_of_interior_angles_l3452_345267


namespace NUMINAMATH_CALUDE_robotics_club_theorem_l3452_345229

theorem robotics_club_theorem (total : ℕ) (cs : ℕ) (elec : ℕ) (both : ℕ)
  (h_total : total = 75)
  (h_cs : cs = 44)
  (h_elec : elec = 40)
  (h_both : both = 25) :
  total - (cs + elec - both) = 16 := by
  sorry

end NUMINAMATH_CALUDE_robotics_club_theorem_l3452_345229


namespace NUMINAMATH_CALUDE_family_change_is_71_l3452_345276

/-- Represents a family member with their age and ticket price. -/
structure FamilyMember where
  age : ℕ
  ticketPrice : ℕ

/-- Calculates the change received after a family visit to an amusement park. -/
def amusementParkChange (family : List FamilyMember) (regularPrice discountAmount paidAmount : ℕ) : ℕ :=
  let totalCost := family.foldl (fun acc member => acc + member.ticketPrice) 0
  paidAmount - totalCost

/-- Theorem: The family receives $71 in change. -/
theorem family_change_is_71 :
  let family : List FamilyMember := [
    { age := 6, ticketPrice := 114 },
    { age := 10, ticketPrice := 114 },
    { age := 13, ticketPrice := 129 },
    { age := 8, ticketPrice := 114 },
    { age := 30, ticketPrice := 129 },  -- Assuming parent age
    { age := 30, ticketPrice := 129 }   -- Assuming parent age
  ]
  let regularPrice := 129
  let discountAmount := 15
  let paidAmount := 800
  amusementParkChange family regularPrice discountAmount paidAmount = 71 := by
sorry

end NUMINAMATH_CALUDE_family_change_is_71_l3452_345276


namespace NUMINAMATH_CALUDE_right_triangle_area_l3452_345201

/-- The area of a right triangle with legs of 60 feet and 80 feet is 345600 square inches -/
theorem right_triangle_area : 
  let leg1_feet : ℝ := 60
  let leg2_feet : ℝ := 80
  let inches_per_foot : ℝ := 12
  let leg1_inches : ℝ := leg1_feet * inches_per_foot
  let leg2_inches : ℝ := leg2_feet * inches_per_foot
  let area : ℝ := (1/2) * leg1_inches * leg2_inches
  area = 345600 := by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l3452_345201


namespace NUMINAMATH_CALUDE_parabola_points_theorem_l3452_345266

/-- Parabola structure -/
structure Parabola where
  f : ℝ → ℝ
  eq : ∀ x, f x ^ 2 = 8 * x

/-- Point on a parabola -/
structure PointOnParabola (p : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : y ^ 2 = 8 * x

/-- Theorem about two points on a parabola -/
theorem parabola_points_theorem (p : Parabola) 
    (A B : PointOnParabola p) (F : ℝ × ℝ) :
  A.y + B.y = 8 →
  F = (2, 0) →
  (B.y - A.y) / (B.x - A.x) = 1 ∧
  ((A.x - F.1) ^ 2 + (A.y - F.2) ^ 2) ^ (1/2 : ℝ) +
  ((B.x - F.1) ^ 2 + (B.y - F.2) ^ 2) ^ (1/2 : ℝ) = 16 :=
by sorry

end NUMINAMATH_CALUDE_parabola_points_theorem_l3452_345266


namespace NUMINAMATH_CALUDE_second_car_speed_l3452_345209

/-- Given two cars starting from opposite ends of a 333-mile highway at the same time,
    with one car traveling at 54 mph and both cars meeting after 3 hours,
    prove that the speed of the second car is 57 mph. -/
theorem second_car_speed (highway_length : ℝ) (time : ℝ) (speed1 : ℝ) (speed2 : ℝ) :
  highway_length = 333 →
  time = 3 →
  speed1 = 54 →
  speed1 * time + speed2 * time = highway_length →
  speed2 = 57 := by
  sorry

end NUMINAMATH_CALUDE_second_car_speed_l3452_345209


namespace NUMINAMATH_CALUDE_or_implies_at_least_one_true_l3452_345242

theorem or_implies_at_least_one_true (p q : Prop) : 
  (p ∨ q) → (p ∨ q) := by sorry

end NUMINAMATH_CALUDE_or_implies_at_least_one_true_l3452_345242


namespace NUMINAMATH_CALUDE_northwest_molded_break_even_l3452_345217

/-- Break-even point calculation for Northwest Molded -/
theorem northwest_molded_break_even :
  let fixed_cost : ℝ := 7640
  let variable_cost : ℝ := 0.60
  let selling_price : ℝ := 4.60
  let break_even_point := fixed_cost / (selling_price - variable_cost)
  break_even_point = 1910 := by
  sorry

end NUMINAMATH_CALUDE_northwest_molded_break_even_l3452_345217


namespace NUMINAMATH_CALUDE_y_intercept_of_f_l3452_345277

/-- A linear function f(x) = x + 1 -/
def f (x : ℝ) : ℝ := x + 1

/-- The y-intercept of f is the point (0, 1) -/
theorem y_intercept_of_f :
  f 0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_of_f_l3452_345277


namespace NUMINAMATH_CALUDE_remainder_of_x_divided_by_82_l3452_345260

theorem remainder_of_x_divided_by_82 (x : ℤ) (k m R : ℤ) 
  (h1 : x = 82 * k + R)
  (h2 : 0 ≤ R ∧ R < 82)
  (h3 : x + 7 = 41 * m + 12) :
  R = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_x_divided_by_82_l3452_345260


namespace NUMINAMATH_CALUDE_minimum_blue_beads_l3452_345231

/-- Represents the color of a bead -/
inductive BeadColor
  | Red
  | Blue
  | Green

/-- Represents a necklace as a cyclic list of bead colors -/
def Necklace := List BeadColor

/-- Returns true if the necklace satisfies the condition that each red bead has neighbors of different colors -/
def redBeadCondition (n : Necklace) : Prop := sorry

/-- Returns true if the necklace satisfies the condition that any segment between two green beads contains at least one blue bead -/
def greenSegmentCondition (n : Necklace) : Prop := sorry

/-- Counts the number of blue beads in the necklace -/
def countBlueBeads (n : Necklace) : Nat := sorry

theorem minimum_blue_beads (n : Necklace) :
  n.length = 175 →
  redBeadCondition n →
  greenSegmentCondition n →
  countBlueBeads n ≥ 30 ∧ ∃ (m : Necklace), m.length = 175 ∧ redBeadCondition m ∧ greenSegmentCondition m ∧ countBlueBeads m = 30 :=
sorry

end NUMINAMATH_CALUDE_minimum_blue_beads_l3452_345231


namespace NUMINAMATH_CALUDE_remainder_theorem_l3452_345248

theorem remainder_theorem (N : ℤ) (h : ∃ k : ℤ, N = 39 * k + 18) : 
  ∃ m : ℤ, N = 13 * m + 5 := by
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3452_345248


namespace NUMINAMATH_CALUDE_sqrt_neg_three_squared_l3452_345250

theorem sqrt_neg_three_squared : Real.sqrt ((-3)^2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_neg_three_squared_l3452_345250


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l3452_345269

theorem interest_rate_calculation (total amount : ℕ) (first_part : ℕ) (first_rate : ℚ) (yearly_income : ℕ) : 
  total = 2500 →
  first_part = 2000 →
  first_rate = 5/100 →
  yearly_income = 130 →
  ∃ second_rate : ℚ,
    (first_part * first_rate + (total - first_part) * second_rate = yearly_income) ∧
    second_rate = 6/100 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l3452_345269


namespace NUMINAMATH_CALUDE_no_integer_solutions_l3452_345211

theorem no_integer_solutions : ¬∃ (x y z : ℤ),
  (x^2 - 4*x*y + 3*y^2 - z^2 = 24) ∧
  (-x^2 + 3*y*z + 5*z^2 = 60) ∧
  (x^2 + 2*x*y + 5*z^2 = 85) :=
by sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l3452_345211


namespace NUMINAMATH_CALUDE_problem_4_l3452_345203

theorem problem_4 (x y : ℝ) (hx : x = 1) (hy : y = 2^100) :
  (x + 2*y)^2 + (x + 2*y)*(x - 2*y) - 4*x*y = 2 := by
sorry

end NUMINAMATH_CALUDE_problem_4_l3452_345203


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3452_345207

theorem inequality_equivalence :
  (∀ x : ℝ, |x + 1| + |x - 1| ≥ a) ↔ (a ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3452_345207


namespace NUMINAMATH_CALUDE_no_same_color_in_large_rectangle_l3452_345270

/-- A coloring of the plane is a function from pairs of integers to colors. -/
def Coloring (Color : Type) := ℤ × ℤ → Color

/-- A rectangle in the plane is defined by its top-left and bottom-right corners. -/
structure Rectangle :=
  (top_left : ℤ × ℤ)
  (bottom_right : ℤ × ℤ)

/-- The perimeter of a rectangle. -/
def Rectangle.perimeter (r : Rectangle) : ℤ :=
  2 * (r.bottom_right.1 - r.top_left.1 + r.top_left.2 - r.bottom_right.2)

/-- A predicate that checks if a coloring satisfies the condition that
    no rectangle with perimeter 100 contains two squares of the same color. -/
def valid_coloring (c : Coloring (Fin 1201)) : Prop :=
  ∀ r : Rectangle, r.perimeter = 100 →
    ∀ x y : ℤ × ℤ, x ≠ y →
      x.1 ∈ Set.Icc r.top_left.1 r.bottom_right.1 →
      x.2 ∈ Set.Icc r.bottom_right.2 r.top_left.2 →
      y.1 ∈ Set.Icc r.top_left.1 r.bottom_right.1 →
      y.2 ∈ Set.Icc r.bottom_right.2 r.top_left.2 →
      c x ≠ c y

/-- The main theorem: if a coloring is valid, then no 1×1201 or 1201×1 rectangle
    contains two squares of the same color. -/
theorem no_same_color_in_large_rectangle
  (c : Coloring (Fin 1201)) (h : valid_coloring c) :
  (∀ r : Rectangle,
    (r.bottom_right.1 - r.top_left.1 = 1200 ∧ r.top_left.2 - r.bottom_right.2 = 0) ∨
    (r.bottom_right.1 - r.top_left.1 = 0 ∧ r.top_left.2 - r.bottom_right.2 = 1200) →
    ∀ x y : ℤ × ℤ, x ≠ y →
      x.1 ∈ Set.Icc r.top_left.1 r.bottom_right.1 →
      x.2 ∈ Set.Icc r.bottom_right.2 r.top_left.2 →
      y.1 ∈ Set.Icc r.top_left.1 r.bottom_right.1 →
      y.2 ∈ Set.Icc r.bottom_right.2 r.top_left.2 →
      c x ≠ c y) :=
by sorry

end NUMINAMATH_CALUDE_no_same_color_in_large_rectangle_l3452_345270


namespace NUMINAMATH_CALUDE_omega_range_l3452_345286

/-- Given a function f(x) = sin(ωx + π/4) where ω > 0, 
    if f(x) is monotonically decreasing in the interval (π/2, π),
    then 1/2 ≤ ω ≤ 5/4 -/
theorem omega_range (ω : ℝ) (h_pos : ω > 0) :
  let f : ℝ → ℝ := λ x ↦ Real.sin (ω * x + π / 4)
  (∀ x ∈ Set.Ioo (π / 2) π, ∀ y ∈ Set.Ioo (π / 2) π, x < y → f x > f y) →
  1 / 2 ≤ ω ∧ ω ≤ 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_omega_range_l3452_345286


namespace NUMINAMATH_CALUDE_okeydokey_earthworms_calculation_l3452_345230

/-- The number of apples Okeydokey invested -/
def okeydokey_apples : ℕ := 5

/-- The number of apples Artichokey invested -/
def artichokey_apples : ℕ := 7

/-- The total number of earthworms in the box -/
def total_earthworms : ℕ := 60

/-- The number of earthworms Okeydokey should receive -/
def okeydokey_earthworms : ℕ := 25

/-- Theorem stating that Okeydokey should receive 25 earthworms -/
theorem okeydokey_earthworms_calculation :
  (okeydokey_apples : ℚ) / (okeydokey_apples + artichokey_apples : ℚ) * total_earthworms = okeydokey_earthworms := by
  sorry

end NUMINAMATH_CALUDE_okeydokey_earthworms_calculation_l3452_345230


namespace NUMINAMATH_CALUDE_tangent_line_y_intercept_l3452_345243

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a 2D plane, represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Predicate to check if a line is tangent to a circle -/
def isTangent (l : Line) (c : Circle) : Prop :=
  ∃ (p : ℝ × ℝ), (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2 ∧
                 p.2 = l.slope * p.1 + l.yIntercept

theorem tangent_line_y_intercept : 
  ∀ (l : Line) (c1 c2 : Circle),
    c1.center = (3, 0) →
    c1.radius = 3 →
    c2.center = (8, 0) →
    c2.radius = 2 →
    isTangent l c1 →
    isTangent l c2 →
    (∃ (p1 p2 : ℝ × ℝ), 
      isTangent l c1 ∧ 
      isTangent l c2 ∧ 
      p1.2 > 0 ∧ 
      p2.2 > 0) →
    l.yIntercept = Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_y_intercept_l3452_345243


namespace NUMINAMATH_CALUDE_first_earthquake_collapse_l3452_345228

/-- Represents the number of buildings collapsed in the first earthquake -/
def first_collapse : ℕ := sorry

/-- Represents the total number of collapsed buildings after four earthquakes -/
def total_collapse : ℕ := 60

/-- Theorem stating that the number of buildings collapsed in the first earthquake is 4 -/
theorem first_earthquake_collapse : 
  (first_collapse + 2 * first_collapse + 4 * first_collapse + 8 * first_collapse = total_collapse) → 
  first_collapse = 4 := by
  sorry

end NUMINAMATH_CALUDE_first_earthquake_collapse_l3452_345228


namespace NUMINAMATH_CALUDE_exactly_two_correct_propositions_l3452_345285

-- Define the basic geometric concepts
def Line : Type := sorry
def intersect (l1 l2 : Line) : Prop := sorry
def perpendicular (l1 l2 : Line) : Prop := sorry
def angle (l1 l2 : Line) : ℝ := sorry
def supplementary_angle (α : ℝ) : ℝ := sorry
def adjacent_angle (l1 l2 : Line) (α : ℝ) : ℝ := sorry
def alternate_interior_angles (l1 l2 : Line) (α β : ℝ) : Prop := sorry
def same_side_interior_angles (l1 l2 : Line) (α β : ℝ) : Prop := sorry
def angle_bisector (l : Line) (α : ℝ) : Line := sorry
def complementary (α β : ℝ) : Prop := sorry

-- Define the four propositions
def proposition1 : Prop :=
  ∀ l1 l2 : Line, intersect l1 l2 →
    ∀ α : ℝ, adjacent_angle l1 l2 α = adjacent_angle l1 l2 (supplementary_angle α) →
      perpendicular l1 l2

def proposition2 : Prop :=
  ∀ l1 l2 : Line, intersect l1 l2 →
    ∀ α : ℝ, α = supplementary_angle α →
      perpendicular l1 l2

def proposition3 : Prop :=
  ∀ l1 l2 : Line, ∀ α β : ℝ,
    alternate_interior_angles l1 l2 α β → α = β →
      perpendicular (angle_bisector l1 α) (angle_bisector l2 β)

def proposition4 : Prop :=
  ∀ l1 l2 : Line, ∀ α β : ℝ,
    same_side_interior_angles l1 l2 α β → complementary α β →
      perpendicular (angle_bisector l1 α) (angle_bisector l2 β)

-- The main theorem
theorem exactly_two_correct_propositions :
  (proposition1 = False ∧
   proposition2 = True ∧
   proposition3 = False ∧
   proposition4 = True) :=
sorry

end NUMINAMATH_CALUDE_exactly_two_correct_propositions_l3452_345285


namespace NUMINAMATH_CALUDE_cube_equation_solution_l3452_345279

theorem cube_equation_solution (a e : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 25 * 45 * e) : e = 49 := by
  sorry

end NUMINAMATH_CALUDE_cube_equation_solution_l3452_345279


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l3452_345246

theorem cubic_equation_solution (a b c : ℝ) : 
  (a^3 - 7*a^2 + 12*a = 18) ∧ 
  (b^3 - 7*b^2 + 12*b = 18) ∧ 
  (c^3 - 7*c^2 + 12*c = 18) →
  a*b/c + b*c/a + c*a/b = -6 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l3452_345246


namespace NUMINAMATH_CALUDE_stamp_collection_value_l3452_345220

theorem stamp_collection_value
  (total_stamps : ℕ)
  (sample_stamps : ℕ)
  (sample_value : ℚ)
  (h1 : total_stamps = 18)
  (h2 : sample_stamps = 6)
  (h3 : sample_value = 15)
  : ℚ :=
by
  -- The total value of the stamp collection is 45 dollars
  sorry

end NUMINAMATH_CALUDE_stamp_collection_value_l3452_345220


namespace NUMINAMATH_CALUDE_franks_filled_boxes_l3452_345287

/-- Given that Frank had 13 boxes initially and 5 boxes are left unfilled,
    prove that the number of boxes he filled with toys is 8. -/
theorem franks_filled_boxes (total : ℕ) (unfilled : ℕ) (filled : ℕ) : 
  total = 13 → unfilled = 5 → filled = total - unfilled → filled = 8 := by sorry

end NUMINAMATH_CALUDE_franks_filled_boxes_l3452_345287


namespace NUMINAMATH_CALUDE_final_coin_count_l3452_345205

/-- Represents the number of coins in the jar at each hour -/
def coin_count : Fin 11 → ℕ
| 0 => 0  -- Initial state
| 1 => 20
| 2 => coin_count 1 + 30
| 3 => coin_count 2 + 30
| 4 => coin_count 3 + 40
| 5 => coin_count 4 - (coin_count 4 * 20 / 100)
| 6 => coin_count 5 + 50
| 7 => coin_count 6 + 60
| 8 => coin_count 7 - (coin_count 7 / 5)
| 9 => coin_count 8 + 70
| 10 => coin_count 9 - (coin_count 9 * 15 / 100)

theorem final_coin_count : coin_count 10 = 200 := by
  sorry

end NUMINAMATH_CALUDE_final_coin_count_l3452_345205


namespace NUMINAMATH_CALUDE_least_three_digit_8_heavy_l3452_345251

def is_8_heavy (n : ℕ) : Prop := n % 8 > 6

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem least_three_digit_8_heavy : 
  (∀ n : ℕ, is_three_digit n → is_8_heavy n → 103 ≤ n) ∧ 
  is_three_digit 103 ∧ 
  is_8_heavy 103 :=
sorry

end NUMINAMATH_CALUDE_least_three_digit_8_heavy_l3452_345251


namespace NUMINAMATH_CALUDE_subsets_with_even_l3452_345289

def S : Finset Nat := {1, 2, 3, 4}

theorem subsets_with_even (A : Finset (Finset Nat)) : 
  A = {s : Finset Nat | s ⊆ S ∧ ∃ n ∈ s, Even n} → Finset.card A = 12 := by
  sorry

end NUMINAMATH_CALUDE_subsets_with_even_l3452_345289


namespace NUMINAMATH_CALUDE_correct_years_passed_l3452_345233

def initial_ages : List Nat := [19, 34, 37, 42, 48]

def new_stem_leaf_plot : List (Nat × List Nat) := 
  [(1, []), (2, [5, 5]), (3, []), (4, [0, 3, 8]), (5, [4])]

def years_passed (initial : List Nat) (new_plot : List (Nat × List Nat)) : Nat :=
  sorry

theorem correct_years_passed :
  years_passed initial_ages new_stem_leaf_plot = 6 := by sorry

end NUMINAMATH_CALUDE_correct_years_passed_l3452_345233


namespace NUMINAMATH_CALUDE_polynomial_alternating_sum_l3452_345223

theorem polynomial_alternating_sum (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (2*x + 1)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  a₀ - a₁ + a₂ - a₃ + a₄ = 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_alternating_sum_l3452_345223


namespace NUMINAMATH_CALUDE_f_odd_f_2a_f_3a_f_monotone_decreasing_l3452_345208

/-- Function with specific properties -/
noncomputable def f (x : ℝ) : ℝ := sorry

/-- Positive constant a -/
noncomputable def a : ℝ := sorry

/-- Domain of f -/
def domain (x : ℝ) : Prop := ∀ k : ℤ, x ≠ k * Real.pi

axiom f_domain : ∀ x : ℝ, domain x → f x ≠ 0

axiom f_equation : ∀ x y : ℝ, domain x → domain y → 
  f (x - y) = (f x * f y + 1) / (f y - f x)

axiom f_a : f a = 1

axiom a_pos : a > 0

axiom f_pos_interval : ∀ x : ℝ, 0 < x → x < 2 * a → f x > 0

/-- f is an odd function -/
theorem f_odd : ∀ x : ℝ, domain x → f (-x) = -f x := by sorry

/-- f(2a) = 0 -/
theorem f_2a : f (2 * a) = 0 := by sorry

/-- f(3a) = -1 -/
theorem f_3a : f (3 * a) = -1 := by sorry

/-- f is monotonically decreasing on [2a, 3a] -/
theorem f_monotone_decreasing : 
  ∀ x y : ℝ, 2 * a ≤ x → x < y → y ≤ 3 * a → f x > f y := by sorry

end NUMINAMATH_CALUDE_f_odd_f_2a_f_3a_f_monotone_decreasing_l3452_345208


namespace NUMINAMATH_CALUDE_expression_value_l3452_345227

theorem expression_value (x y : ℝ) (h : x - 2*y = 1) : 3 - 4*y + 2*x = 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3452_345227


namespace NUMINAMATH_CALUDE_complex_fraction_bounds_l3452_345244

theorem complex_fraction_bounds (z w : ℂ) (hz : z ≠ 0) (hw : w ≠ 0) :
  ∃ (min max : ℝ),
    (∀ z w : ℂ, z ≠ 0 → w ≠ 0 → min ≤ Complex.abs (z + w) / (Complex.abs z + Complex.abs w) ∧
                        Complex.abs (z + w) / (Complex.abs z + Complex.abs w) ≤ max) ∧
    min = 0 ∧ max = 1 ∧ max - min = 1 :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_bounds_l3452_345244


namespace NUMINAMATH_CALUDE_no_positive_integer_solution_l3452_345225

theorem no_positive_integer_solution : 
  ¬∃ (n m : ℕ+), n^4 - m^4 = 42 := by
  sorry

end NUMINAMATH_CALUDE_no_positive_integer_solution_l3452_345225


namespace NUMINAMATH_CALUDE_circle_and_trajectory_l3452_345262

-- Define the circle C
def circle_C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 4)^2 + p.2^2 = 5}

-- Define points M and N
def M : ℝ × ℝ := (5, 2)
def N : ℝ × ℝ := (3, 2)

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Theorem statement
theorem circle_and_trajectory :
  (M ∈ circle_C) ∧ 
  (N ∈ circle_C) ∧ 
  (∀ p ∈ circle_C, p.1 = 4 → p.2 = 0) →
  (∀ A ∈ circle_C, 
    ∃ P : ℝ × ℝ, 
      (P.1 - O.1 = 2 * (A.1 - O.1)) ∧ 
      (P.2 - O.2 = 2 * (A.2 - O.2)) ∧
      (P.1 - 8)^2 + P.2^2 = 20) := by
  sorry

end NUMINAMATH_CALUDE_circle_and_trajectory_l3452_345262


namespace NUMINAMATH_CALUDE_total_trees_is_86_l3452_345271

/-- Calculates the number of trees that can be planted on a street --/
def treesOnStreet (length : ℕ) (spacing : ℕ) : ℕ :=
  (length / spacing) + 1

/-- The total number of trees that can be planted on all five streets --/
def totalTrees : ℕ :=
  treesOnStreet 151 14 + treesOnStreet 210 18 + treesOnStreet 275 12 +
  treesOnStreet 345 20 + treesOnStreet 475 22

theorem total_trees_is_86 : totalTrees = 86 := by
  sorry

end NUMINAMATH_CALUDE_total_trees_is_86_l3452_345271


namespace NUMINAMATH_CALUDE_power_multiplication_l3452_345237

theorem power_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l3452_345237


namespace NUMINAMATH_CALUDE_envelope_stuffing_l3452_345265

/-- The total number of envelopes Rachel needs to stuff -/
def total_envelopes : ℕ := 1500

/-- The total time Rachel has to complete the task -/
def total_time : ℕ := 8

/-- The number of envelopes Rachel stuffs in the first hour -/
def first_hour : ℕ := 135

/-- The number of envelopes Rachel stuffs in the second hour -/
def second_hour : ℕ := 141

/-- The number of envelopes Rachel needs to stuff per hour to finish the job -/
def required_rate : ℕ := 204

theorem envelope_stuffing :
  total_envelopes = first_hour + second_hour + required_rate * (total_time - 2) := by
  sorry

end NUMINAMATH_CALUDE_envelope_stuffing_l3452_345265


namespace NUMINAMATH_CALUDE_contradiction_elements_correct_l3452_345274

/-- Elements used in the method of contradiction -/
inductive ContradictionElement
  | assumption
  | originalCondition
  | axiomTheoremDefinition

/-- The set of elements used in the method of contradiction -/
def contradictionElements : Set ContradictionElement :=
  {ContradictionElement.assumption, ContradictionElement.originalCondition, ContradictionElement.axiomTheoremDefinition}

/-- Theorem stating that the set of elements used in the method of contradiction
    is exactly the set containing assumptions, original conditions, and axioms/theorems/definitions -/
theorem contradiction_elements_correct :
  contradictionElements = {ContradictionElement.assumption, ContradictionElement.originalCondition, ContradictionElement.axiomTheoremDefinition} := by
  sorry


end NUMINAMATH_CALUDE_contradiction_elements_correct_l3452_345274


namespace NUMINAMATH_CALUDE_f_properties_l3452_345254

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sin (2 * x) + a * (Real.cos x) ^ 2

theorem f_properties (a : ℝ) (h : f a (π / 4) = 0) :
  -- The smallest positive period of f(x) is π
  (∃ (T : ℝ), T > 0 ∧ T = π ∧ ∀ (x : ℝ), f a (x + T) = f a x) ∧
  -- The maximum value of f(x) on [π/24, 11π/24] is √2 - 1
  (∀ (x : ℝ), x ∈ Set.Icc (π / 24) (11 * π / 24) → f a x ≤ Real.sqrt 2 - 1) ∧
  (∃ (x : ℝ), x ∈ Set.Icc (π / 24) (11 * π / 24) ∧ f a x = Real.sqrt 2 - 1) ∧
  -- The minimum value of f(x) on [π/24, 11π/24] is -√2/2 - 1
  (∀ (x : ℝ), x ∈ Set.Icc (π / 24) (11 * π / 24) → f a x ≥ -Real.sqrt 2 / 2 - 1) ∧
  (∃ (x : ℝ), x ∈ Set.Icc (π / 24) (11 * π / 24) ∧ f a x = -Real.sqrt 2 / 2 - 1) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l3452_345254


namespace NUMINAMATH_CALUDE_constant_term_of_expansion_l3452_345200

theorem constant_term_of_expansion (x : ℝ) (x_pos : x > 0) :
  ∃ (c : ℝ), (∀ (ε : ℝ), ε > 0 → 
    ∃ (δ : ℝ), δ > 0 ∧ 
    ∀ (y : ℝ), abs (y - x) < δ → 
    abs ((y.sqrt + 3 / y)^10 - c) < ε) ∧
  c = 59049 := by
sorry

end NUMINAMATH_CALUDE_constant_term_of_expansion_l3452_345200


namespace NUMINAMATH_CALUDE_remainder_sum_powers_mod_seven_l3452_345255

theorem remainder_sum_powers_mod_seven :
  (9^6 + 8^7 + 7^8) % 7 = 2 := by
sorry

end NUMINAMATH_CALUDE_remainder_sum_powers_mod_seven_l3452_345255


namespace NUMINAMATH_CALUDE_money_duration_l3452_345292

def mowing_earnings : ℕ := 9
def weed_eating_earnings : ℕ := 18
def weekly_spending : ℕ := 3

theorem money_duration : 
  (mowing_earnings + weed_eating_earnings) / weekly_spending = 9 := by
  sorry

end NUMINAMATH_CALUDE_money_duration_l3452_345292


namespace NUMINAMATH_CALUDE_greatest_integer_problem_l3452_345235

theorem greatest_integer_problem : 
  ⌊100 * (Real.cos (18.5 * π / 180) / Real.sin (17.5 * π / 180))⌋ = 273 := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_problem_l3452_345235


namespace NUMINAMATH_CALUDE_division_problem_l3452_345213

theorem division_problem (x : ℝ) (h : 10 / x = 2) : 20 / x = 4 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l3452_345213


namespace NUMINAMATH_CALUDE_max_production_years_l3452_345290

/-- The cumulative production function after n years -/
def f (n : ℕ) : ℚ := (1/2) * n * (n + 1) * (2 * n + 1)

/-- The annual production function -/
def annual_production (n : ℕ) : ℚ := 
  if n = 1 then f 1 else f n - f (n - 1)

/-- The maximum allowed annual production -/
def max_allowed_production : ℚ := 150

/-- The maximum number of years the production line can operate -/
def max_years : ℕ := 7

theorem max_production_years : 
  (∀ n : ℕ, n ≤ max_years → annual_production n ≤ max_allowed_production) ∧
  (annual_production (max_years + 1) > max_allowed_production) :=
sorry

end NUMINAMATH_CALUDE_max_production_years_l3452_345290


namespace NUMINAMATH_CALUDE_material_mix_ratio_l3452_345245

theorem material_mix_ratio (x y : ℝ) 
  (h1 : 50 * x + 40 * y = 50 * (1 + 0.1) * x + 40 * (1 - 0.15) * y) : 
  x / y = 6 / 5 := by
  sorry

end NUMINAMATH_CALUDE_material_mix_ratio_l3452_345245


namespace NUMINAMATH_CALUDE_shaded_triangle_probability_l3452_345299

/-- Given a set of triangles, some of which are shaded, this theorem proves
    the probability of selecting a shaded triangle. -/
theorem shaded_triangle_probability
  (total_triangles : ℕ)
  (shaded_triangles : ℕ)
  (h1 : total_triangles = 6)
  (h2 : shaded_triangles = 3)
  (h3 : shaded_triangles ≤ total_triangles)
  (h4 : total_triangles > 0) :
  (shaded_triangles : ℚ) / total_triangles = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_shaded_triangle_probability_l3452_345299


namespace NUMINAMATH_CALUDE_special_quadrilateral_not_necessarily_square_l3452_345238

/-- A convex quadrilateral with special diagonal properties -/
structure SpecialQuadrilateral where
  /-- The quadrilateral is convex -/
  convex : Bool
  /-- Any diagonal divides the quadrilateral into two isosceles triangles -/
  diagonal_isosceles : Bool
  /-- Both diagonals divide the quadrilateral into four isosceles triangles -/
  both_diagonals_isosceles : Bool

/-- An isosceles trapezoid -/
structure IsoscelesTrapezoid where
  /-- The trapezoid has parallel bases -/
  parallel_bases : Bool
  /-- The non-parallel sides are equal -/
  equal_legs : Bool
  /-- The smaller base is equal to the legs -/
  base_equals_legs : Bool

/-- Theorem: There exists a quadrilateral satisfying the special properties that is not a square -/
theorem special_quadrilateral_not_necessarily_square :
  ∃ (q : SpecialQuadrilateral) (t : IsoscelesTrapezoid),
    q.convex ∧
    q.diagonal_isosceles ∧
    q.both_diagonals_isosceles ∧
    t.parallel_bases ∧
    t.equal_legs ∧
    t.base_equals_legs ∧
    (q ≠ square) := by
  sorry

end NUMINAMATH_CALUDE_special_quadrilateral_not_necessarily_square_l3452_345238


namespace NUMINAMATH_CALUDE_intersection_at_midpoint_l3452_345263

/-- Given a line segment from (3,6) to (5,10) and a line x + y = b that
    intersects this segment at its midpoint, prove that b = 12. -/
theorem intersection_at_midpoint (b : ℝ) : 
  (∃ (x y : ℝ), x + y = b ∧ 
    x = (3 + 5) / 2 ∧ 
    y = (6 + 10) / 2) → 
  b = 12 := by
sorry

end NUMINAMATH_CALUDE_intersection_at_midpoint_l3452_345263


namespace NUMINAMATH_CALUDE_typist_salary_problem_l3452_345214

/-- Proves that if a salary S is increased by 10% and then decreased by 5%,
    resulting in Rs. 6270, then the original salary S was Rs. 6000. -/
theorem typist_salary_problem (S : ℝ) : 
  (S * 1.1 * 0.95 = 6270) → S = 6000 := by
  sorry

end NUMINAMATH_CALUDE_typist_salary_problem_l3452_345214


namespace NUMINAMATH_CALUDE_triangle_prime_angles_l3452_345215

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem triangle_prime_angles 
  (a b c : ℕ) 
  (sum_180 : a + b + c = 180) 
  (all_prime : is_prime a ∧ is_prime b ∧ is_prime c) 
  (all_less_120 : a < 120 ∧ b < 120 ∧ c < 120) : 
  ((a = 2 ∧ b = 71 ∧ c = 107) ∨ (a = 2 ∧ b = 89 ∧ c = 89)) ∨
  ((a = 71 ∧ b = 2 ∧ c = 107) ∨ (a = 89 ∧ b = 2 ∧ c = 89)) ∨
  ((a = 71 ∧ b = 107 ∧ c = 2) ∨ (a = 89 ∧ b = 89 ∧ c = 2)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_prime_angles_l3452_345215


namespace NUMINAMATH_CALUDE_closest_ratio_l3452_345261

/-- The annual interest rate -/
def interest_rate : ℝ := 0.05

/-- The number of years -/
def years : ℕ := 10

/-- The ratio of final amount to initial amount after compound interest -/
def ratio : ℝ := (1 + interest_rate) ^ years

/-- The given options for the ratio -/
def options : List ℝ := [1.5, 1.6, 1.7, 1.8]

/-- Theorem stating that 1.6 is the closest option to the actual ratio -/
theorem closest_ratio : 
  ∃ (x : ℝ), x ∈ options ∧ ∀ (y : ℝ), y ∈ options → |ratio - x| ≤ |ratio - y| ∧ x = 1.6 :=
sorry

end NUMINAMATH_CALUDE_closest_ratio_l3452_345261


namespace NUMINAMATH_CALUDE_inequality_statements_l3452_345239

theorem inequality_statements :
  (∀ a b c : ℝ, c ≠ 0 → (a * c^2 < b * c^2 → a < b)) ∧
  (∃ a x y : ℝ, x > y ∧ ¬(-a^2 * x < -a^2 * y)) ∧
  (∀ a b c : ℝ, c ≠ 0 → (a / c^2 < b / c^2 → a < b)) ∧
  (∀ a b : ℝ, a > b → 2 - a < 2 - b) :=
by sorry

end NUMINAMATH_CALUDE_inequality_statements_l3452_345239


namespace NUMINAMATH_CALUDE_existence_of_h₁_h₂_l3452_345253

theorem existence_of_h₁_h₂ :
  ∃ (h₁ h₂ : ℝ → ℝ),
    ∀ (g₁ g₂ : ℝ → ℝ) (x : ℝ),
      (∀ s, 1 ≤ g₁ s) →
      (∀ s, 1 ≤ g₂ s) →
      (∃ M, ∀ s, g₁ s ≤ M) →
      (∃ N, ∀ s, g₂ s ≤ N) →
      (⨆ s, (g₁ s) ^ x * g₂ s) = ⨆ t, x * h₁ t + h₂ t :=
by
  sorry

end NUMINAMATH_CALUDE_existence_of_h₁_h₂_l3452_345253


namespace NUMINAMATH_CALUDE_min_players_team_l3452_345219

theorem min_players_team (n : ℕ) : 
  (n % 8 = 0 ∧ n % 9 = 0 ∧ n % 10 = 0 ∧ n % 11 = 0) → n ≥ 7920 :=
by sorry

end NUMINAMATH_CALUDE_min_players_team_l3452_345219


namespace NUMINAMATH_CALUDE_original_number_of_classes_l3452_345278

theorem original_number_of_classes : 
  ∃! x : ℕ+, 
    (280 % x.val = 0) ∧ 
    (585 % (x.val + 6) = 0) ∧ 
    x.val = 7 := by
  sorry

end NUMINAMATH_CALUDE_original_number_of_classes_l3452_345278


namespace NUMINAMATH_CALUDE_intersection_of_symmetric_lines_l3452_345280

/-- Two lines that are symmetric about the x-axis -/
structure SymmetricLines where
  k : ℝ
  b : ℝ
  l₁ : ℝ → ℝ := fun x ↦ k * x + 2
  l₂ : ℝ → ℝ := fun x ↦ -x + b
  symmetric : l₁ 0 = -l₂ 0

/-- The intersection point of two symmetric lines is (-2, 0) -/
theorem intersection_of_symmetric_lines (lines : SymmetricLines) :
  ∃ x y, lines.l₁ x = lines.l₂ x ∧ x = -2 ∧ y = 0 := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_symmetric_lines_l3452_345280


namespace NUMINAMATH_CALUDE_total_outfits_is_168_l3452_345206

/-- The number of shirts available -/
def num_shirts : ℕ := 8

/-- The number of ties available -/
def num_ties : ℕ := 7

/-- The number of hats available -/
def num_hats : ℕ := 2

/-- The number of hat options (including not wearing a hat) -/
def hat_options : ℕ := num_hats + 1

/-- The total number of possible outfits -/
def total_outfits : ℕ := num_shirts * num_ties * hat_options

/-- Theorem stating that the total number of outfits is 168 -/
theorem total_outfits_is_168 : total_outfits = 168 := by
  sorry

end NUMINAMATH_CALUDE_total_outfits_is_168_l3452_345206


namespace NUMINAMATH_CALUDE_log_23_between_consecutive_integers_l3452_345224

theorem log_23_between_consecutive_integers :
  ∃ (a b : ℤ), (a + 1 = b) ∧ (a < Real.log 23 / Real.log 10) ∧ (Real.log 23 / Real.log 10 < b) ∧ (a + b = 3) := by
  sorry

end NUMINAMATH_CALUDE_log_23_between_consecutive_integers_l3452_345224


namespace NUMINAMATH_CALUDE_solution_set_a_range_l3452_345218

-- Define the functions f and g
def f (x a : ℝ) : ℝ := |x - 1| + |x + a|
def g (a : ℝ) : ℝ := a^2 - a - 2

-- Part 1
theorem solution_set (x : ℝ) :
  (f x 3 > g 3 + 2) ↔ (x < -4 ∨ x > 2) :=
sorry

-- Part 2
theorem a_range (a : ℝ) :
  (∀ x ∈ Set.Icc (-a) 1, f x a ≤ g a) → a ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_solution_set_a_range_l3452_345218


namespace NUMINAMATH_CALUDE_symmetry_about_origin_l3452_345204

/-- Given a point (x, y) in R^2, its symmetrical point about the origin is (-x, -y) -/
def symmetrical_point (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)

/-- The original point -/
def original_point : ℝ × ℝ := (2, -3)

/-- The proposed symmetrical point -/
def proposed_symmetrical_point : ℝ × ℝ := (-2, 3)

theorem symmetry_about_origin :
  symmetrical_point original_point = proposed_symmetrical_point :=
by sorry

end NUMINAMATH_CALUDE_symmetry_about_origin_l3452_345204


namespace NUMINAMATH_CALUDE_jeff_bought_seven_one_yuan_socks_l3452_345293

/-- Represents the number of sock pairs at each price point -/
structure SockPurchase where
  one_yuan : ℕ
  three_yuan : ℕ
  four_yuan : ℕ

/-- Checks if a SockPurchase satisfies the given conditions -/
def is_valid_purchase (p : SockPurchase) : Prop :=
  p.one_yuan + p.three_yuan + p.four_yuan = 12 ∧
  p.one_yuan * 1 + p.three_yuan * 3 + p.four_yuan * 4 = 24 ∧
  p.one_yuan ≥ 1 ∧ p.three_yuan ≥ 1 ∧ p.four_yuan ≥ 1

/-- The main theorem stating that the only valid purchase has 7 pairs of 1-yuan socks -/
theorem jeff_bought_seven_one_yuan_socks :
  ∀ p : SockPurchase, is_valid_purchase p → p.one_yuan = 7 := by
  sorry

end NUMINAMATH_CALUDE_jeff_bought_seven_one_yuan_socks_l3452_345293


namespace NUMINAMATH_CALUDE_lcm_factor_problem_l3452_345272

theorem lcm_factor_problem (A B : ℕ+) (X : ℕ) (hcf : Nat.gcd A B = 42) 
  (lcm : Nat.lcm A B = 42 * X * 14) (a_val : A = 588) (a_greater : A > B) : X = 1 := by
  sorry

end NUMINAMATH_CALUDE_lcm_factor_problem_l3452_345272


namespace NUMINAMATH_CALUDE_teacher_selection_arrangements_l3452_345241

theorem teacher_selection_arrangements (n_male : ℕ) (n_female : ℕ) (n_select : ℕ) : 
  n_male = 5 → n_female = 4 → n_select = 3 →
  (Nat.choose (n_male + n_female) n_select - Nat.choose n_male n_select - Nat.choose n_female n_select) = 70 := by
  sorry

end NUMINAMATH_CALUDE_teacher_selection_arrangements_l3452_345241


namespace NUMINAMATH_CALUDE_factors_of_N_l3452_345240

/-- The number of natural-number factors of N, where N = 2^5 * 3^4 * 5^3 * 7^2 * 11^1 -/
def number_of_factors (N : ℕ) : ℕ :=
  (5 + 1) * (4 + 1) * (3 + 1) * (2 + 1) * (1 + 1)

/-- Theorem stating that the number of natural-number factors of N is 720 -/
theorem factors_of_N :
  let N : ℕ := 2^5 * 3^4 * 5^3 * 7^2 * 11^1
  number_of_factors N = 720 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_N_l3452_345240


namespace NUMINAMATH_CALUDE_jerry_total_miles_l3452_345264

/-- The total miles Jerry walked over three days -/
def total_miles (monday tuesday wednesday : ℕ) : ℕ :=
  monday + tuesday + wednesday

/-- Theorem stating that Jerry walked 45 miles in total -/
theorem jerry_total_miles :
  total_miles 15 18 12 = 45 := by
  sorry

end NUMINAMATH_CALUDE_jerry_total_miles_l3452_345264


namespace NUMINAMATH_CALUDE_perfect_squares_theorem_l3452_345212

theorem perfect_squares_theorem (x y z : ℕ+) 
  (h_coprime : ∀ d : ℕ, d > 1 → ¬(d ∣ x ∧ d ∣ y ∧ d ∣ z))
  (h_eq : (x : ℚ)⁻¹ + (y : ℚ)⁻¹ = (z : ℚ)⁻¹) :
  ∃ (a b : ℕ), 
    (x : ℤ) - (z : ℤ) = a^2 ∧ 
    (y : ℤ) - (z : ℤ) = b^2 ∧ 
    (x : ℤ) + (y : ℤ) = (a + b)^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_squares_theorem_l3452_345212


namespace NUMINAMATH_CALUDE_ages_solution_l3452_345258

def mother_daughter_ages (daughter_age : ℕ) (mother_age : ℕ) : Prop :=
  (mother_age = daughter_age + 45) ∧
  (mother_age - 5 = 6 * (daughter_age - 5))

theorem ages_solution : ∃ (daughter_age : ℕ) (mother_age : ℕ),
  mother_daughter_ages daughter_age mother_age ∧
  daughter_age = 14 ∧ mother_age = 59 := by
  sorry

end NUMINAMATH_CALUDE_ages_solution_l3452_345258


namespace NUMINAMATH_CALUDE_bus_average_speed_with_stoppages_l3452_345282

/-- Calculates the average speed of a bus including stoppages -/
theorem bus_average_speed_with_stoppages 
  (speed_without_stoppages : ℝ) 
  (stoppage_time : ℝ) 
  (total_time : ℝ) :
  speed_without_stoppages = 50 →
  stoppage_time = 12 →
  total_time = 60 →
  (speed_without_stoppages * (total_time - stoppage_time) / total_time) = 40 :=
by sorry

end NUMINAMATH_CALUDE_bus_average_speed_with_stoppages_l3452_345282


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3452_345249

theorem complex_equation_solution (x : ℝ) (i : ℂ) 
  (h1 : i * i = -1) 
  (h2 : (1 - 2*i) * (x + i) = 4 - 3*i) : 
  x = 2 := by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3452_345249


namespace NUMINAMATH_CALUDE_only_event3_mutually_exclusive_l3452_345295

-- Define the set of numbers
def NumberSet : Set Nat := {n | 1 ≤ n ∧ n ≤ 9}

-- Define the sample space
def SampleSpace : Set (Nat × Nat) :=
  {pair | pair.1 ∈ NumberSet ∧ pair.2 ∈ NumberSet ∧ pair.1 ≠ pair.2}

-- Define event ①
def Event1 (pair : Nat × Nat) : Prop :=
  (pair.1 % 2 = 0 ∧ pair.2 % 2 = 1) ∨ (pair.1 % 2 = 1 ∧ pair.2 % 2 = 0)

-- Define event ②
def Event2 (pair : Nat × Nat) : Prop :=
  (pair.1 % 2 = 1 ∨ pair.2 % 2 = 1) ∧ (pair.1 % 2 = 1 ∧ pair.2 % 2 = 1)

-- Define event ③
def Event3 (pair : Nat × Nat) : Prop :=
  (pair.1 % 2 = 1 ∨ pair.2 % 2 = 1) ∧ (pair.1 % 2 = 0 ∧ pair.2 % 2 = 0)

-- Define event ④
def Event4 (pair : Nat × Nat) : Prop :=
  (pair.1 % 2 = 1 ∨ pair.2 % 2 = 1) ∧ (pair.1 % 2 = 0 ∨ pair.2 % 2 = 0)

-- Theorem stating that only Event3 is mutually exclusive with other events
theorem only_event3_mutually_exclusive :
  ∀ (pair : Nat × Nat), pair ∈ SampleSpace →
    (¬(Event1 pair ∧ Event3 pair) ∧
     ¬(Event2 pair ∧ Event3 pair) ∧
     ¬(Event4 pair ∧ Event3 pair)) ∧
    ((Event1 pair ∧ Event2 pair) ∨
     (Event1 pair ∧ Event4 pair) ∨
     (Event2 pair ∧ Event4 pair)) :=
by sorry

end NUMINAMATH_CALUDE_only_event3_mutually_exclusive_l3452_345295


namespace NUMINAMATH_CALUDE_unique_function_exists_l3452_345202

/-- A function satisfying the given inequality for all real x, y, z and fixed positive integer k -/
def SatisfiesInequality (f : ℝ → ℝ) (k : ℕ+) : Prop :=
  ∀ x y z : ℝ, f (x * y) + f (x + z) + k * f x * f (y * z) ≥ k^2

/-- There exists only one function satisfying the inequality -/
theorem unique_function_exists (k : ℕ+) : ∃! f : ℝ → ℝ, SatisfiesInequality f k := by
  sorry

end NUMINAMATH_CALUDE_unique_function_exists_l3452_345202


namespace NUMINAMATH_CALUDE_square_area_12m_l3452_345281

theorem square_area_12m (side_length : ℝ) (area : ℝ) : 
  side_length = 12 → area = side_length^2 → area = 144 := by sorry

end NUMINAMATH_CALUDE_square_area_12m_l3452_345281


namespace NUMINAMATH_CALUDE_planet_colonization_combinations_l3452_345252

/-- Represents the number of planets of each type -/
structure PlanetCounts where
  venusLike : Nat
  jupiterLike : Nat

/-- Represents the colonization units required for each planet type -/
structure ColonizationUnits where
  venusLike : Nat
  jupiterLike : Nat

/-- Calculates the number of ways to choose planets given the constraints -/
def countPlanetCombinations (totalPlanets : PlanetCounts) (units : ColonizationUnits) (totalUnits : Nat) : Nat :=
  sorry

/-- The main theorem stating the number of combinations for the given problem -/
theorem planet_colonization_combinations :
  let totalPlanets := PlanetCounts.mk 7 5
  let units := ColonizationUnits.mk 3 1
  let totalUnits := 15
  countPlanetCombinations totalPlanets units totalUnits = 435 := by
  sorry

end NUMINAMATH_CALUDE_planet_colonization_combinations_l3452_345252


namespace NUMINAMATH_CALUDE_semicircle_perimeter_approx_l3452_345283

/-- The perimeter of a semicircle with radius 11 is approximately 56.56 -/
theorem semicircle_perimeter_approx :
  let r : ℝ := 11
  let π_approx : ℝ := 3.14159
  let semicircle_perimeter := π_approx * r + 2 * r
  ∃ ε > 0, abs (semicircle_perimeter - 56.56) < ε :=
by sorry

end NUMINAMATH_CALUDE_semicircle_perimeter_approx_l3452_345283


namespace NUMINAMATH_CALUDE_smallest_n_for_sqrt_difference_l3452_345247

theorem smallest_n_for_sqrt_difference : 
  ∀ n : ℕ, n > 0 → (Real.sqrt n - Real.sqrt (n - 1) < 0.02 → n ≥ 626) ∧ 
  (Real.sqrt 626 - Real.sqrt 625 < 0.02) := by
sorry

end NUMINAMATH_CALUDE_smallest_n_for_sqrt_difference_l3452_345247


namespace NUMINAMATH_CALUDE_smallest_norm_u_l3452_345234

theorem smallest_norm_u (u : ℝ × ℝ) (h : ‖u + (5, 2)‖ = 10) :
  ∃ (v : ℝ × ℝ), ‖v‖ = 10 - Real.sqrt 29 ∧ ∀ w : ℝ × ℝ, ‖w + (5, 2)‖ = 10 → ‖v‖ ≤ ‖w‖ := by
  sorry

end NUMINAMATH_CALUDE_smallest_norm_u_l3452_345234


namespace NUMINAMATH_CALUDE_room_length_calculation_l3452_345236

/-- Given a room with specified width, total paving cost, and paving rate per square meter,
    calculate the length of the room. -/
theorem room_length_calculation (width : ℝ) (total_cost : ℝ) (rate_per_sqm : ℝ) :
  width = 3.75 →
  total_cost = 6187.5 →
  rate_per_sqm = 300 →
  (total_cost / rate_per_sqm) / width = 5.5 :=
by
  sorry

end NUMINAMATH_CALUDE_room_length_calculation_l3452_345236


namespace NUMINAMATH_CALUDE_triangle_theorem_l3452_345226

open Real

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)
  (h1 : 0 < a ∧ 0 < b ∧ 0 < c)
  (h2 : 0 < A ∧ 0 < B ∧ 0 < C)
  (h3 : A + B + C = π)
  (h4 : a * sin A = b * sin B)
  (h5 : b * sin B = c * sin C)
  (h6 : a * sin A - c * sin C = (a - b) * sin B)

/-- The theorem stating the angle C and maximum area of the triangle -/
theorem triangle_theorem (t : Triangle) (h : t.c = sqrt 6) :
  t.C = π / 3 ∧
  ∃ (S : ℝ), S = (3 * sqrt 3) / 2 ∧ ∀ (S' : ℝ), S' ≤ S := by
  sorry


end NUMINAMATH_CALUDE_triangle_theorem_l3452_345226


namespace NUMINAMATH_CALUDE_problem_solution_l3452_345210

theorem problem_solution (x y : ℝ) (h1 : 3*x + y = 5) (h2 : x + 3*y = 8) :
  5*x^2 + 11*x*y + 5*y^2 = 89 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3452_345210


namespace NUMINAMATH_CALUDE_problem_solution_l3452_345284

def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | m - 4 ≤ x ∧ x ≤ 3 * m + 2}

theorem problem_solution :
  (∀ m : ℝ, A ∪ B m = B m → m ∈ Set.Icc 1 2) ∧
  (∀ m : ℝ, A ∩ B m = B m → m < -3) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3452_345284


namespace NUMINAMATH_CALUDE_equation_solution_l3452_345222

theorem equation_solution :
  ∃ (x : ℝ), x ≠ -3 ∧ (2 / (x + 3) + 3 * x / (x + 3) - 5 / (x + 3) = 5) ∧ x = -9 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3452_345222


namespace NUMINAMATH_CALUDE_min_value_quadratic_l3452_345291

theorem min_value_quadratic (x : ℝ) :
  let f : ℝ → ℝ := λ x => 3 * x^2 + 18 * x + 7
  ∃ (min_val : ℝ), (∀ x, f x ≥ min_val) ∧ (min_val = -20) := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l3452_345291


namespace NUMINAMATH_CALUDE_mary_walking_distance_approx_l3452_345296

/-- Represents the journey Mary took to her sister's house -/
structure Journey where
  total_distance : ℝ
  bike_speed : ℝ
  walk_speed : ℝ
  bike_portion : ℝ
  total_time : ℝ

/-- Calculates the walking distance for a given journey -/
def walking_distance (j : Journey) : ℝ :=
  (1 - j.bike_portion) * j.total_distance

/-- The theorem stating that Mary's walking distance is approximately 0.3 km -/
theorem mary_walking_distance_approx (j : Journey) 
  (h1 : j.bike_speed = 15)
  (h2 : j.walk_speed = 4)
  (h3 : j.bike_portion = 0.4)
  (h4 : j.total_time = 0.6) : 
  ∃ (ε : ℝ), ε > 0 ∧ abs (walking_distance j - 0.3) < ε := by
  sorry

#check mary_walking_distance_approx

end NUMINAMATH_CALUDE_mary_walking_distance_approx_l3452_345296


namespace NUMINAMATH_CALUDE_wall_tiling_impossible_l3452_345268

/-- Represents the dimensions of a rectangular cuboid -/
structure Dimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Checks if a smaller cuboid can tile a larger cuboid -/
def can_tile (wall : Dimensions) (brick : Dimensions) : Prop :=
  ∃ (a b c : ℕ), 
    (a * brick.length = wall.length ∧ 
     b * brick.width = wall.width ∧ 
     c * brick.height = wall.height) ∨
    (a * brick.length = wall.length ∧ 
     b * brick.width = wall.height ∧ 
     c * brick.height = wall.width) ∨
    (a * brick.length = wall.width ∧ 
     b * brick.width = wall.length ∧ 
     c * brick.height = wall.height) ∨
    (a * brick.length = wall.width ∧ 
     b * brick.width = wall.height ∧ 
     c * brick.height = wall.length) ∨
    (a * brick.length = wall.height ∧ 
     b * brick.width = wall.length ∧ 
     c * brick.height = wall.width) ∨
    (a * brick.length = wall.height ∧ 
     b * brick.width = wall.width ∧ 
     c * brick.height = wall.length)

theorem wall_tiling_impossible (wall : Dimensions) 
  (brick1 : Dimensions) (brick2 : Dimensions) : 
  wall.length = 27 ∧ wall.width = 16 ∧ wall.height = 15 →
  brick1.length = 3 ∧ brick1.width = 5 ∧ brick1.height = 7 →
  brick2.length = 2 ∧ brick2.width = 5 ∧ brick2.height = 6 →
  ¬(can_tile wall brick1 ∨ can_tile wall brick2) :=
sorry

end NUMINAMATH_CALUDE_wall_tiling_impossible_l3452_345268


namespace NUMINAMATH_CALUDE_larger_cuboid_height_l3452_345257

/-- Prove that the height of a larger cuboid is 2 meters given specific conditions -/
theorem larger_cuboid_height (small_length small_width small_height : ℝ)
  (large_length large_width : ℝ) (num_small_cuboids : ℝ) :
  small_length = 6 →
  small_width = 4 →
  small_height = 3 →
  large_length = 18 →
  large_width = 15 →
  num_small_cuboids = 7.5 →
  ∃ (large_height : ℝ),
    num_small_cuboids * (small_length * small_width * small_height) =
      large_length * large_width * large_height ∧
    large_height = 2 := by
  sorry

end NUMINAMATH_CALUDE_larger_cuboid_height_l3452_345257


namespace NUMINAMATH_CALUDE_prime_square_minus_one_divisible_by_thirty_l3452_345256

theorem prime_square_minus_one_divisible_by_thirty {p : ℕ} (hp : Prime p) (hp_ge_7 : p ≥ 7) :
  30 ∣ p^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_prime_square_minus_one_divisible_by_thirty_l3452_345256


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3452_345259

/-- A quadratic function y = px^2 + qx + r with specific properties -/
structure QuadraticFunction where
  p : ℝ
  q : ℝ
  r : ℝ
  is_parabola : True
  vertex_x : p ≠ 0 → -q / (2 * p) = -3
  vertex_y : p ≠ 0 → p * (-3)^2 + q * (-3) + r = 4
  passes_through_origin : p * 0^2 + q * 0 + r = -2
  vertical_symmetry : True

/-- The sum of coefficients p, q, and r equals -20/3 -/
theorem sum_of_coefficients (f : QuadraticFunction) : f.p + f.q + f.r = -20/3 := by
  sorry


end NUMINAMATH_CALUDE_sum_of_coefficients_l3452_345259


namespace NUMINAMATH_CALUDE_coffee_per_day_l3452_345297

/-- The number of times Maria goes to the coffee shop per day. -/
def visits_per_day : ℕ := 2

/-- The number of cups of coffee Maria orders each visit. -/
def cups_per_visit : ℕ := 3

/-- Theorem: Maria orders 6 cups of coffee per day. -/
theorem coffee_per_day : visits_per_day * cups_per_visit = 6 := by
  sorry

end NUMINAMATH_CALUDE_coffee_per_day_l3452_345297


namespace NUMINAMATH_CALUDE_fraction_value_l3452_345221

theorem fraction_value (a b : ℚ) (h : 2 * a = 3 * b) : a / b = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l3452_345221


namespace NUMINAMATH_CALUDE_factorial_divisibility_l3452_345273

theorem factorial_divisibility (n : ℕ) : 
  (∃ (p q : ℕ), p ≤ n ∧ q ≤ n ∧ n + 2 = p * q) ∨ 
  (∃ (p : ℕ), p ≥ 3 ∧ Prime p ∧ n + 2 = p^2) ↔ 
  (n + 2) ∣ n! :=
sorry

end NUMINAMATH_CALUDE_factorial_divisibility_l3452_345273
