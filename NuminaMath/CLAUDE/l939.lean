import Mathlib

namespace NUMINAMATH_CALUDE_canal_meeting_participants_l939_93966

theorem canal_meeting_participants (total : Nat) (greetings : Nat) : total = 12 ∧ greetings = 31 →
  ∃ (egyptians panamanians : Nat),
    egyptians + panamanians = total ∧
    egyptians > panamanians ∧
    egyptians * (egyptians - 1) / 2 + panamanians * (panamanians - 1) / 2 = greetings ∧
    egyptians = 7 ∧
    panamanians = 5 := by
  sorry

end NUMINAMATH_CALUDE_canal_meeting_participants_l939_93966


namespace NUMINAMATH_CALUDE_dispatch_riders_travel_time_l939_93960

/-- Represents the travel scenario of two dispatch riders -/
structure DispatchRiders where
  a : ℝ  -- Speed increase of the first rider in km/h
  x : ℝ  -- Initial speed of the first rider in km/h
  y : ℝ  -- Speed of the second rider in km/h
  z : ℝ  -- Actual travel time of the first rider in hours

/-- The conditions of the dispatch riders' travel -/
def travel_conditions (d : DispatchRiders) : Prop :=
  d.a > 0 ∧ d.a < 30 ∧
  d.x > 0 ∧ d.y > 0 ∧ d.z > 0 ∧
  180 / d.x - 180 / d.y = 6 ∧
  d.z * (d.x + d.a) = 180 ∧
  (d.z - 3) * d.y = 180

/-- The theorem stating the travel times of both riders -/
theorem dispatch_riders_travel_time (d : DispatchRiders) 
  (h : travel_conditions d) : 
  d.z = (-3 * d.a + 3 * Real.sqrt (d.a^2 + 240 * d.a)) / (2 * d.a) ∧
  d.z - 3 = (-9 * d.a + 3 * Real.sqrt (d.a^2 + 240 * d.a)) / (2 * d.a) := by
  sorry

end NUMINAMATH_CALUDE_dispatch_riders_travel_time_l939_93960


namespace NUMINAMATH_CALUDE_sum_of_valid_a_l939_93958

theorem sum_of_valid_a : ∃ (S : Finset ℤ), 
  (∀ a ∈ S, (∃! (x₁ x₂ : ℤ), x₁ ≠ x₂ ∧ 
    5 * x₁ ≥ 3 * (x₁ + 2) ∧ x₁ - (x₁ + 3) / 2 ≤ a / 16 ∧
    5 * x₂ ≥ 3 * (x₂ + 2) ∧ x₂ - (x₂ + 3) / 2 ≤ a / 16) ∧
   (∃ y : ℤ, y < 0 ∧ 5 + a * y = 2 * y - 7)) ∧
  (S.sum id = 22) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_valid_a_l939_93958


namespace NUMINAMATH_CALUDE_circle_area_through_points_l939_93943

/-- The area of a circle with center P(2, -5) passing through Q(-7, 6) is 202π. -/
theorem circle_area_through_points :
  let P : ℝ × ℝ := (2, -5)
  let Q : ℝ × ℝ := (-7, 6)
  let r : ℝ := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
  (π * r^2) = 202 * π := by sorry

end NUMINAMATH_CALUDE_circle_area_through_points_l939_93943


namespace NUMINAMATH_CALUDE_root_value_theorem_l939_93979

theorem root_value_theorem (m : ℝ) : 2 * m^2 - 3 * m - 3 = 0 → 4 * m^2 - 6 * m + 2017 = 2023 := by
  sorry

end NUMINAMATH_CALUDE_root_value_theorem_l939_93979


namespace NUMINAMATH_CALUDE_betty_total_items_betty_total_cost_l939_93930

/-- The number of slippers Betty ordered -/
def slippers : ℕ := 6

/-- The number of lipsticks Betty ordered -/
def lipsticks : ℕ := 4

/-- The number of hair colors Betty ordered -/
def hair_colors : ℕ := 8

/-- The cost of each slipper -/
def slipper_cost : ℚ := 5/2

/-- The cost of each lipstick -/
def lipstick_cost : ℚ := 5/4

/-- The cost of each hair color -/
def hair_color_cost : ℚ := 3

/-- The total amount Betty paid -/
def total_paid : ℚ := 44

/-- Theorem stating that Betty ordered 18 items in total -/
theorem betty_total_items : slippers + lipsticks + hair_colors = 18 := by
  sorry

/-- Theorem verifying the total cost matches the amount Betty paid -/
theorem betty_total_cost : 
  slippers * slipper_cost + lipsticks * lipstick_cost + hair_colors * hair_color_cost = total_paid := by
  sorry

end NUMINAMATH_CALUDE_betty_total_items_betty_total_cost_l939_93930


namespace NUMINAMATH_CALUDE_correlation_relationships_l939_93913

/-- Represents a relationship between two variables -/
structure Relationship where
  variable1 : String
  variable2 : String

/-- Determines if a relationship represents a correlation -/
def is_correlation (r : Relationship) : Prop :=
  match r with
  | ⟨"snowfall", "traffic accidents"⟩ => True
  | ⟨"brain capacity", "intelligence"⟩ => True
  | ⟨"age", "weight"⟩ => False
  | ⟨"rainfall", "crop yield"⟩ => True
  | _ => False

/-- The main theorem stating which relationships represent correlations -/
theorem correlation_relationships :
  let r1 : Relationship := ⟨"snowfall", "traffic accidents"⟩
  let r2 : Relationship := ⟨"brain capacity", "intelligence"⟩
  let r3 : Relationship := ⟨"age", "weight"⟩
  let r4 : Relationship := ⟨"rainfall", "crop yield"⟩
  is_correlation r1 ∧ is_correlation r2 ∧ ¬is_correlation r3 ∧ is_correlation r4 :=
by sorry


end NUMINAMATH_CALUDE_correlation_relationships_l939_93913


namespace NUMINAMATH_CALUDE_train_crossing_time_l939_93972

/-- The time it takes for a train to cross a man walking in the opposite direction -/
theorem train_crossing_time (train_length : Real) (train_speed : Real) (man_speed : Real) :
  train_length = 50 ∧ 
  train_speed = 24.997600191984645 ∧ 
  man_speed = 5 →
  (train_length / ((train_speed + man_speed) * (1000 / 3600))) = 6 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l939_93972


namespace NUMINAMATH_CALUDE_light_glow_duration_l939_93950

/-- The number of times the light glowed between 1:57:58 and 3:20:47 am -/
def glow_count : ℝ := 292.29411764705884

/-- The total time in seconds between 1:57:58 am and 3:20:47 am -/
def total_time : ℕ := 4969

/-- The duration of each light glow in seconds -/
def glow_duration : ℕ := 17

theorem light_glow_duration :
  Int.floor (total_time / glow_count) = glow_duration := by sorry

end NUMINAMATH_CALUDE_light_glow_duration_l939_93950


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l939_93941

theorem fraction_equation_solution :
  ∃ (x : ℚ), x ≠ 3 ∧ x ≠ -2 ∧ (x + 4) / (x - 3) = (x - 2) / (x + 2) ∧ x = -2 / 11 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l939_93941


namespace NUMINAMATH_CALUDE_smallest_sum_of_squared_ratios_l939_93988

theorem smallest_sum_of_squared_ratios (c d : ℕ) (hc : c > 0) (hd : d > 0) :
  ∃ (min : ℚ), min = 2 ∧
  (((c + d : ℚ) / (c - d : ℚ))^2 + ((c - d : ℚ) / (c + d : ℚ))^2 ≥ min) ∧
  ∃ (c' d' : ℕ), c' > 0 ∧ d' > 0 ∧
  ((c' + d' : ℚ) / (c' - d' : ℚ))^2 + ((c' - d' : ℚ) / (c' + d' : ℚ))^2 = min :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_squared_ratios_l939_93988


namespace NUMINAMATH_CALUDE_gcd_bound_from_lcm_l939_93901

theorem gcd_bound_from_lcm (a b : ℕ) : 
  (10^6 ≤ a ∧ a < 10^7) →
  (10^6 ≤ b ∧ b < 10^7) →
  (10^11 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10^12) →
  Nat.gcd a b < 1000 := by sorry

end NUMINAMATH_CALUDE_gcd_bound_from_lcm_l939_93901


namespace NUMINAMATH_CALUDE_heartsuit_three_eight_l939_93912

-- Define the ♥ operation
def heartsuit (x y : ℝ) : ℝ := 4 * x + 6 * y

-- Theorem statement
theorem heartsuit_three_eight : heartsuit 3 8 = 60 := by
  sorry

end NUMINAMATH_CALUDE_heartsuit_three_eight_l939_93912


namespace NUMINAMATH_CALUDE_initial_marbles_equation_l939_93987

/-- The number of marbles Connie had initially -/
def initial_marbles : ℕ := sorry

/-- The number of marbles Connie gave to Juan -/
def marbles_given : ℕ := 73

/-- The number of marbles Connie has left -/
def marbles_left : ℕ := 70

/-- Theorem stating that the initial number of marbles is equal to
    the sum of marbles given away and marbles left -/
theorem initial_marbles_equation : initial_marbles = marbles_given + marbles_left := by
  sorry

end NUMINAMATH_CALUDE_initial_marbles_equation_l939_93987


namespace NUMINAMATH_CALUDE_third_set_total_l939_93982

/-- Represents a set of candies -/
structure CandySet where
  hard : ℕ
  chocolate : ℕ
  gummy : ℕ

/-- The problem setup -/
def candy_problem (set1 set2 set3 : CandySet) : Prop :=
  -- Total number of each type of candy is equal across all sets
  set1.hard + set2.hard + set3.hard = set1.chocolate + set2.chocolate + set3.chocolate ∧
  set1.hard + set2.hard + set3.hard = set1.gummy + set2.gummy + set3.gummy ∧
  -- First set conditions
  set1.chocolate = set1.gummy ∧
  set1.hard = set1.chocolate + 7 ∧
  -- Second set conditions
  set2.hard = set2.chocolate ∧
  set2.gummy = set2.hard - 15 ∧
  -- Third set condition
  set3.hard = 0

/-- The theorem to prove -/
theorem third_set_total (set1 set2 set3 : CandySet) 
  (h : candy_problem set1 set2 set3) : 
  set3.chocolate + set3.gummy = 29 := by
  sorry

end NUMINAMATH_CALUDE_third_set_total_l939_93982


namespace NUMINAMATH_CALUDE_rectangular_table_capacity_l939_93956

/-- The number of rectangular tables in the library -/
def num_rectangular_tables : ℕ := 7

/-- The number of pupils a square table can seat -/
def pupils_per_square_table : ℕ := 4

/-- The number of square tables in the library -/
def num_square_tables : ℕ := 5

/-- The total number of pupils that can be seated -/
def total_pupils : ℕ := 90

/-- The number of pupils a rectangular table can seat -/
def pupils_per_rectangular_table : ℕ := 10

theorem rectangular_table_capacity :
  pupils_per_rectangular_table * num_rectangular_tables +
  pupils_per_square_table * num_square_tables = total_pupils :=
by sorry

end NUMINAMATH_CALUDE_rectangular_table_capacity_l939_93956


namespace NUMINAMATH_CALUDE_casey_pumping_rate_l939_93940

def corn_rows : ℕ := 4
def corn_plants_per_row : ℕ := 15
def water_per_corn_plant : ℚ := 1/2
def num_pigs : ℕ := 10
def water_per_pig : ℚ := 4
def num_ducks : ℕ := 20
def water_per_duck : ℚ := 1/4
def pumping_time : ℕ := 25

theorem casey_pumping_rate :
  let total_corn_plants := corn_rows * corn_plants_per_row
  let water_for_corn := (total_corn_plants : ℚ) * water_per_corn_plant
  let water_for_pigs := (num_pigs : ℚ) * water_per_pig
  let water_for_ducks := (num_ducks : ℚ) * water_per_duck
  let total_water := water_for_corn + water_for_pigs + water_for_ducks
  total_water / (pumping_time : ℚ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_casey_pumping_rate_l939_93940


namespace NUMINAMATH_CALUDE_circle_tangent_triangle_area_l939_93992

/-- Represents a circle with a center point and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a point in 2D space -/
def Point := ℝ × ℝ

/-- The distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Checks if a point lies on a circle -/
def onCircle (p : Point) (c : Circle) : Prop := sorry

/-- Checks if two circles are externally tangent -/
def externallyTangent (c1 c2 : Circle) : Prop := sorry

/-- Checks if a line segment is tangent to a circle -/
def isTangent (p1 p2 : Point) (c : Circle) : Prop := sorry

/-- Calculates the area of a triangle given its three vertices -/
def triangleArea (p1 p2 p3 : Point) : ℝ := sorry

theorem circle_tangent_triangle_area 
  (ω₁ ω₂ ω₃ : Circle)
  (P₁ P₂ P₃ : Point)
  (h_radius : ω₁.radius = 24 ∧ ω₂.radius = 24 ∧ ω₃.radius = 24)
  (h_tangent : externallyTangent ω₁ ω₂ ∧ externallyTangent ω₂ ω₃ ∧ externallyTangent ω₃ ω₁)
  (h_on_circle : onCircle P₁ ω₁ ∧ onCircle P₂ ω₂ ∧ onCircle P₃ ω₃)
  (h_equidistant : distance P₁ P₂ = distance P₂ P₃ ∧ distance P₂ P₃ = distance P₃ P₁)
  (h_tangent_sides : isTangent P₁ P₂ ω₂ ∧ isTangent P₂ P₃ ω₃ ∧ isTangent P₃ P₁ ω₁)
  : ∃ (a b : ℕ), triangleArea P₁ P₂ P₃ = Real.sqrt a + Real.sqrt b ∧ a + b = 288 := by
  sorry

end NUMINAMATH_CALUDE_circle_tangent_triangle_area_l939_93992


namespace NUMINAMATH_CALUDE_wage_multiple_l939_93934

/-- Given Kem's hourly wage and Shem's daily wage for 8 hours, 
    calculate the multiple of Shem's hourly wage compared to Kem's. -/
theorem wage_multiple (kem_hourly_wage shem_daily_wage : ℚ) 
  (h1 : kem_hourly_wage = 4)
  (h2 : shem_daily_wage = 80)
  (h3 : shem_daily_wage = 8 * (shem_daily_wage / 8)) : 
  (shem_daily_wage / 8) / kem_hourly_wage = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_wage_multiple_l939_93934


namespace NUMINAMATH_CALUDE_complex_number_real_condition_l939_93989

theorem complex_number_real_condition (a b : ℝ) :
  let z : ℂ := Complex.mk (a^2 + b^2) (a + |a|)
  (z.im = 0) ↔ (a ≤ 0) := by sorry

end NUMINAMATH_CALUDE_complex_number_real_condition_l939_93989


namespace NUMINAMATH_CALUDE_cos_alpha_value_l939_93986

theorem cos_alpha_value (α : Real) (h1 : 0 < α ∧ α < π / 2) 
  (h2 : Real.sin (α - π / 6) = 1 / 3) : 
  Real.cos α = (2 * Real.sqrt 6 - 1) / 6 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l939_93986


namespace NUMINAMATH_CALUDE_bobby_chocolate_pieces_l939_93915

/-- The number of chocolate pieces Bobby ate -/
def chocolate_pieces (initial_candy pieces_more_candy total_pieces : ℕ) : ℕ :=
  total_pieces - (initial_candy + pieces_more_candy)

theorem bobby_chocolate_pieces :
  chocolate_pieces 33 4 51 = 14 := by
  sorry

end NUMINAMATH_CALUDE_bobby_chocolate_pieces_l939_93915


namespace NUMINAMATH_CALUDE_right_of_symmetry_decreasing_l939_93916

-- Define the quadratic function
def f (x : ℝ) : ℝ := -2 * (x - 1)^2

-- Define the axis of symmetry
def axis_of_symmetry : ℝ := 1

-- Theorem statement
theorem right_of_symmetry_decreasing :
  ∀ x₁ x₂ : ℝ, x₁ > axis_of_symmetry → x₂ > x₁ → f x₂ < f x₁ := by
  sorry

end NUMINAMATH_CALUDE_right_of_symmetry_decreasing_l939_93916


namespace NUMINAMATH_CALUDE_frog_climbs_out_l939_93924

def well_depth : ℕ := 19
def day_climb : ℕ := 3
def night_slide : ℕ := 2

def days_to_climb (depth : ℕ) (day_climb : ℕ) (night_slide : ℕ) : ℕ :=
  (depth - day_climb) / (day_climb - night_slide) + 1

theorem frog_climbs_out : days_to_climb well_depth day_climb night_slide = 17 := by
  sorry

end NUMINAMATH_CALUDE_frog_climbs_out_l939_93924


namespace NUMINAMATH_CALUDE_buffy_whiskers_l939_93932

/-- Represents the number of whiskers for each cat -/
structure CatWhiskers where
  juniper : ℕ
  puffy : ℕ
  scruffy : ℕ
  buffy : ℕ

/-- The conditions for the cat whiskers problem -/
def whiskerConditions (c : CatWhiskers) : Prop :=
  c.juniper = 12 ∧
  c.puffy = 3 * c.juniper ∧
  c.scruffy = 2 * c.puffy ∧
  c.buffy = (c.juniper + c.puffy + c.scruffy) / 3

/-- Theorem stating that under the given conditions, Buffy has 40 whiskers -/
theorem buffy_whiskers (c : CatWhiskers) :
  whiskerConditions c → c.buffy = 40 := by
  sorry

end NUMINAMATH_CALUDE_buffy_whiskers_l939_93932


namespace NUMINAMATH_CALUDE_strawberry_yogurt_probability_l939_93900

def prob_strawberry_yogurt (n : ℕ) (k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1 - p)^(n - k)

theorem strawberry_yogurt_probability :
  let n₁ := 3
  let n₂ := 3
  let p₁ := (1 : ℚ) / 2
  let p₂ := (3 : ℚ) / 4
  let total_days := n₁ + n₂
  let success_days := 4
  (total_days.choose success_days : ℚ) *
    (prob_strawberry_yogurt n₁ 2 p₁ * prob_strawberry_yogurt n₂ 2 p₂ +
     prob_strawberry_yogurt n₁ 3 p₁ * prob_strawberry_yogurt n₂ 1 p₂) =
  1485 / 64 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_yogurt_probability_l939_93900


namespace NUMINAMATH_CALUDE_total_spent_is_30_40_l939_93931

/-- Represents the store's inventory and pricing --/
structure Store where
  barrette_price : ℝ
  comb_price : ℝ
  hairband_price : ℝ
  hair_ties_price : ℝ

/-- Represents a customer's purchase --/
structure Purchase where
  barrettes : ℕ
  combs : ℕ
  hairbands : ℕ
  hair_ties : ℕ

/-- Calculates the total cost of a purchase before discount and tax --/
def purchase_cost (s : Store) (p : Purchase) : ℝ :=
  s.barrette_price * p.barrettes +
  s.comb_price * p.combs +
  s.hairband_price * p.hairbands +
  s.hair_ties_price * p.hair_ties

/-- Applies discount if applicable --/
def apply_discount (cost : ℝ) (item_count : ℕ) : ℝ :=
  if item_count > 5 then cost * 0.85 else cost

/-- Applies sales tax --/
def apply_tax (cost : ℝ) : ℝ :=
  cost * 1.08

/-- Calculates the final cost of a purchase after discount and tax --/
def final_cost (s : Store) (p : Purchase) : ℝ :=
  let initial_cost := purchase_cost s p
  let item_count := p.barrettes + p.combs + p.hairbands + p.hair_ties
  let discounted_cost := apply_discount initial_cost item_count
  apply_tax discounted_cost

/-- The main theorem --/
theorem total_spent_is_30_40 (s : Store) (k_purchase c_purchase : Purchase) :
  s.barrette_price = 4 ∧
  s.comb_price = 2 ∧
  s.hairband_price = 3 ∧
  s.hair_ties_price = 2.5 ∧
  k_purchase = { barrettes := 1, combs := 1, hairbands := 2, hair_ties := 0 } ∧
  c_purchase = { barrettes := 3, combs := 1, hairbands := 0, hair_ties := 2 } →
  final_cost s k_purchase + final_cost s c_purchase = 30.40 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_is_30_40_l939_93931


namespace NUMINAMATH_CALUDE_range_of_m_l939_93946

-- Define propositions p and q
def p (x : ℝ) : Prop := x < -2 ∨ x > 10
def q (x m : ℝ) : Prop := 1 - m ≤ x ∧ x ≤ 1 + m^2

-- Define the sufficient but not necessary condition
def sufficient_not_necessary (m : ℝ) : Prop :=
  (∀ x : ℝ, ¬(p x) → q x m) ∧ ¬(∀ x : ℝ, q x m → ¬(p x))

-- Theorem statement
theorem range_of_m :
  ∀ m : ℝ, sufficient_not_necessary m ↔ m > 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l939_93946


namespace NUMINAMATH_CALUDE_specific_tetrahedron_volume_l939_93962

/-- Represents a tetrahedron ABCD with given edge lengths -/
structure Tetrahedron where
  AB : ℝ
  AC : ℝ
  BC : ℝ
  BD : ℝ
  AD : ℝ
  CD : ℝ

/-- Calculate the volume of a tetrahedron given its edge lengths -/
def tetrahedronVolume (t : Tetrahedron) : ℝ :=
  sorry

/-- The theorem stating that the volume of the specific tetrahedron is 24/5 -/
theorem specific_tetrahedron_volume :
  let t : Tetrahedron := {
    AB := 5,
    AC := 3,
    BC := 4,
    BD := 4,
    AD := 3,
    CD := 12/5 * Real.sqrt 2
  }
  tetrahedronVolume t = 24/5 := by
  sorry

end NUMINAMATH_CALUDE_specific_tetrahedron_volume_l939_93962


namespace NUMINAMATH_CALUDE_dave_outer_space_books_l939_93909

/-- The number of books about outer space Dave bought -/
def outer_space_books : ℕ := 6

/-- The number of books about animals Dave bought -/
def animal_books : ℕ := 8

/-- The number of books about trains Dave bought -/
def train_books : ℕ := 3

/-- The cost of each book in dollars -/
def book_cost : ℕ := 6

/-- The total amount Dave spent on books in dollars -/
def total_spent : ℕ := 102

theorem dave_outer_space_books :
  outer_space_books = (total_spent - book_cost * (animal_books + train_books)) / book_cost :=
by sorry

end NUMINAMATH_CALUDE_dave_outer_space_books_l939_93909


namespace NUMINAMATH_CALUDE_pasta_preference_ratio_l939_93905

theorem pasta_preference_ratio : 
  ∀ (total spaghetti manicotti : ℕ),
    total = 800 →
    spaghetti = 320 →
    manicotti = 160 →
    (spaghetti : ℚ) / (manicotti : ℚ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_pasta_preference_ratio_l939_93905


namespace NUMINAMATH_CALUDE_smallest_x_multiple_of_53_l939_93967

theorem smallest_x_multiple_of_53 : 
  ∃ (x : ℕ), x > 0 ∧ 
  (∀ (y : ℕ), y > 0 → (3 * y + 28)^2 % 53 = 0 → x ≤ y) ∧ 
  (3 * x + 28)^2 % 53 = 0 ∧ 
  x = 26 := by
sorry

end NUMINAMATH_CALUDE_smallest_x_multiple_of_53_l939_93967


namespace NUMINAMATH_CALUDE_given_number_scientific_notation_l939_93995

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coefficient_range : 1 ≤ coefficient ∧ coefficient < 10

/-- The given number in meters -/
def given_number : ℝ := 0.000000014

/-- The scientific notation representation of the given number -/
def scientific_representation : ScientificNotation :=
  { coefficient := 1.4
    exponent := -8
    coefficient_range := by sorry }

theorem given_number_scientific_notation :
  given_number = scientific_representation.coefficient * (10 : ℝ) ^ scientific_representation.exponent :=
by sorry

end NUMINAMATH_CALUDE_given_number_scientific_notation_l939_93995


namespace NUMINAMATH_CALUDE_michael_pet_sitting_cost_l939_93965

-- Define the number of cats and dogs
def num_cats : ℕ := 2
def num_dogs : ℕ := 3

-- Define the cost per animal per night
def cost_per_animal : ℕ := 13

-- Define the total number of animals
def total_animals : ℕ := num_cats + num_dogs

-- State the theorem
theorem michael_pet_sitting_cost :
  total_animals * cost_per_animal = 65 := by
  sorry

end NUMINAMATH_CALUDE_michael_pet_sitting_cost_l939_93965


namespace NUMINAMATH_CALUDE_inequality_solution_set_l939_93976

theorem inequality_solution_set (x : ℝ) (h : x ≠ 3) :
  (2 * x - 1) / (x - 3) ≥ 1 ↔ x > 3 ∨ x ≤ -2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l939_93976


namespace NUMINAMATH_CALUDE_find_n_l939_93999

theorem find_n (m n : ℕ) (h1 : Nat.lcm m n = 690) (h2 : ¬3 ∣ n) (h3 : ¬2 ∣ m) : n = 230 := by
  sorry

end NUMINAMATH_CALUDE_find_n_l939_93999


namespace NUMINAMATH_CALUDE_complex_magnitude_l939_93957

theorem complex_magnitude (z : ℂ) (h : (1 - Complex.I) * z = 2) : Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l939_93957


namespace NUMINAMATH_CALUDE_uniform_price_calculation_l939_93914

/-- Calculates the price of a uniform given the conditions of a servant's employment --/
def uniform_price (full_year_salary : ℚ) (actual_salary : ℚ) (months_worked : ℕ) : ℚ :=
  full_year_salary * (months_worked / 12) - actual_salary

theorem uniform_price_calculation :
  uniform_price 900 650 9 = 25 := by
  sorry

end NUMINAMATH_CALUDE_uniform_price_calculation_l939_93914


namespace NUMINAMATH_CALUDE_solution_set_not_negative_interval_l939_93984

theorem solution_set_not_negative_interval (a b : ℝ) :
  {x : ℝ | a * x > b} ≠ Set.Iio (-b/a) :=
sorry

end NUMINAMATH_CALUDE_solution_set_not_negative_interval_l939_93984


namespace NUMINAMATH_CALUDE_volume_of_sphere_wedge_l939_93902

/-- The volume of a wedge of a sphere -/
theorem volume_of_sphere_wedge (c : ℝ) (h : c = 12 * Real.pi) :
  let r := c / (2 * Real.pi)
  let sphere_volume := (4 / 3) * Real.pi * r^3
  let wedge_volume := sphere_volume / 4
  wedge_volume = 72 * Real.pi := by
sorry


end NUMINAMATH_CALUDE_volume_of_sphere_wedge_l939_93902


namespace NUMINAMATH_CALUDE_no_positive_rational_root_l939_93975

theorem no_positive_rational_root : ¬∃ (q : ℚ), q > 0 ∧ q^3 - 10*q^2 + q - 2021 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_positive_rational_root_l939_93975


namespace NUMINAMATH_CALUDE_inequality_proof_l939_93964

theorem inequality_proof (n : ℕ) : (n - 1 : ℝ)^(n + 1) * (n + 1 : ℝ)^(n - 1) < n^(2 * n) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l939_93964


namespace NUMINAMATH_CALUDE_money_sum_is_fifty_l939_93908

def jack_money : ℕ := 26

def ben_money (jack : ℕ) : ℕ := jack - 9

def eric_money (ben : ℕ) : ℕ := ben - 10

def total_money (jack ben eric : ℕ) : ℕ := jack + ben + eric

theorem money_sum_is_fifty :
  total_money jack_money (ben_money jack_money) (eric_money (ben_money jack_money)) = 50 := by
  sorry

end NUMINAMATH_CALUDE_money_sum_is_fifty_l939_93908


namespace NUMINAMATH_CALUDE_exactly_one_integer_satisfies_inequality_l939_93996

theorem exactly_one_integer_satisfies_inequality :
  ∃! (n : ℕ), n > 0 ∧ 30 - 6 * n > 18 :=
by sorry

end NUMINAMATH_CALUDE_exactly_one_integer_satisfies_inequality_l939_93996


namespace NUMINAMATH_CALUDE_grasshopper_return_to_origin_l939_93981

def jump_length (n : ℕ) : ℕ := n

def is_horizontal (n : ℕ) : Bool :=
  n % 2 = 1

theorem grasshopper_return_to_origin :
  let horizontal_jumps := List.range 31 |>.filter is_horizontal |>.map jump_length
  let vertical_jumps := List.range 31 |>.filter (fun n => ¬ is_horizontal n) |>.map jump_length
  (List.sum horizontal_jumps = 0) ∧ (List.sum vertical_jumps = 0) := by
  sorry

end NUMINAMATH_CALUDE_grasshopper_return_to_origin_l939_93981


namespace NUMINAMATH_CALUDE_peculiar_animal_farm_l939_93935

theorem peculiar_animal_farm (cats dogs : ℕ) : 
  dogs = cats + 180 →
  (cats + (dogs / 5 : ℚ)) / (cats + dogs : ℚ) = 32 / 100 →
  cats + dogs = 240 := by
sorry

end NUMINAMATH_CALUDE_peculiar_animal_farm_l939_93935


namespace NUMINAMATH_CALUDE_symmetry_proof_l939_93954

/-- Given two lines in the 2D plane represented by their equations,
    this function returns true if they are symmetric with respect to the line y = -x. -/
def are_symmetric (line1 line2 : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, line1 x y ↔ line2 (-y) (-x)

/-- The equation of the original line: 3x - 4y + 5 = 0 -/
def original_line (x y : ℝ) : Prop := 3 * x - 4 * y + 5 = 0

/-- The equation of the symmetric line: 4x - 3y + 5 = 0 -/
def symmetric_line (x y : ℝ) : Prop := 4 * x - 3 * y + 5 = 0

/-- Theorem stating that the symmetric_line is indeed symmetric to the original_line
    with respect to the line x + y = 0 -/
theorem symmetry_proof : are_symmetric original_line symmetric_line :=
sorry

end NUMINAMATH_CALUDE_symmetry_proof_l939_93954


namespace NUMINAMATH_CALUDE_line_conditions_l939_93963

-- Define the types for points and lines
def Point := ℝ × ℝ
def Line := ℝ → ℝ → ℝ

-- Define the lines from the problem
def line1 : Line := λ x y => x + y - 1
def line2 : Line := λ x y => x + y - 2
def line3 : Line := λ x y => x - 3*y + 3
def line4 : Line := λ x y => 3*x + y + 1

-- Define the points from the problem
def point1 : Point := (-1, 2)
def point2 : Point := (0, 1)

-- Define what it means for a line to pass through a point
def passes_through (l : Line) (p : Point) : Prop :=
  l p.1 p.2 = 0

-- Define what it means for two lines to be parallel
def parallel (l1 l2 : Line) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x y, l1 x y = k * l2 x y

-- Define what it means for two lines to be perpendicular
def perpendicular (l1 l2 : Line) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x y, l1 x y = k * l2 y (-x)

-- State the theorem
theorem line_conditions : 
  (passes_through line1 point1 ∧ parallel line1 line2) ∧
  (passes_through line3 point2 ∧ perpendicular line3 line4) := by
  sorry

end NUMINAMATH_CALUDE_line_conditions_l939_93963


namespace NUMINAMATH_CALUDE_vacuum_pump_usage_l939_93970

/-- The fraction of air remaining after each use of the pump -/
def remaining_fraction : ℝ := 0.4

/-- The target fraction of air to reach -/
def target_fraction : ℝ := 0.005

/-- The minimum number of pump uses required -/
def min_pump_uses : ℕ := 6

theorem vacuum_pump_usage (n : ℕ) :
  n ≥ min_pump_uses ↔ remaining_fraction ^ n < target_fraction :=
sorry

end NUMINAMATH_CALUDE_vacuum_pump_usage_l939_93970


namespace NUMINAMATH_CALUDE_max_product_sum_300_l939_93937

theorem max_product_sum_300 :
  ∃ (x : ℤ), x * (300 - x) = 22500 ∧ ∀ (y : ℤ), y * (300 - y) ≤ 22500 :=
sorry

end NUMINAMATH_CALUDE_max_product_sum_300_l939_93937


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l939_93903

theorem imaginary_part_of_z (z : ℂ) (h : z * (2 + Complex.I) = 3 - 6 * Complex.I) : 
  z.im = -3 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l939_93903


namespace NUMINAMATH_CALUDE_inserted_numbers_in_arithmetic_sequence_l939_93919

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : Fin n → ℝ :=
  λ i => a₁ + d * i.val

theorem inserted_numbers_in_arithmetic_sequence :
  let n : ℕ := 8
  let a₁ : ℝ := 8
  let aₙ : ℝ := 36
  let d : ℝ := (aₙ - a₁) / (n - 1)
  let seq := arithmetic_sequence a₁ d n
  (seq 1 = 12) ∧
  (seq 2 = 16) ∧
  (seq 3 = 20) ∧
  (seq 4 = 24) ∧
  (seq 5 = 28) ∧
  (seq 6 = 32) :=
by sorry

end NUMINAMATH_CALUDE_inserted_numbers_in_arithmetic_sequence_l939_93919


namespace NUMINAMATH_CALUDE_line_through_P_and_origin_equation_line_l_equation_l939_93994

-- Define the lines l₁, l₂, and l₃
def l₁ (x y : ℝ) : Prop := 3 * x + 4 * y - 2 = 0
def l₂ (x y : ℝ) : Prop := 2 * x + y + 2 = 0
def l₃ (x y : ℝ) : Prop := 2 * x + y + 2 = 0

-- Define the intersection point P
def P : ℝ × ℝ := (-2, 2)

-- Define the line passing through P and the origin
def line_through_P_and_origin (x y : ℝ) : Prop := x + y = 0

-- Define the line l passing through P and perpendicular to l₃
def line_l (x y : ℝ) : Prop := x - 2 * y + 6 = 0

-- Theorem 1: The line passing through P and the origin has the equation x + y = 0
theorem line_through_P_and_origin_equation :
  ∀ x y : ℝ, l₁ x y ∧ l₂ x y → line_through_P_and_origin x y :=
by sorry

-- Theorem 2: The line l passing through P and perpendicular to l₃ has the equation x - 2y + 6 = 0
theorem line_l_equation :
  ∀ x y : ℝ, l₁ x y ∧ l₂ x y ∧ l₃ x y → line_l x y :=
by sorry

end NUMINAMATH_CALUDE_line_through_P_and_origin_equation_line_l_equation_l939_93994


namespace NUMINAMATH_CALUDE_sin_cos_sum_equals_negative_one_l939_93917

theorem sin_cos_sum_equals_negative_one : 
  Real.sin (315 * π / 180) - Real.cos (135 * π / 180) + 2 * Real.sin (570 * π / 180) = -1 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_equals_negative_one_l939_93917


namespace NUMINAMATH_CALUDE_problem_statement_l939_93990

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = a * b) :
  (a + b ≥ 4) ∧ (a + 4 * b ≥ 9) ∧ (1 / a^2 + 2 / b^2 ≥ 2 / 3) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l939_93990


namespace NUMINAMATH_CALUDE_money_division_theorem_l939_93959

/-- Represents the shares of P, Q, and R in the money division problem -/
structure Shares where
  p : ℝ
  q : ℝ
  r : ℝ

/-- The problem of dividing money between P, Q, and R -/
def MoneyDivisionProblem (s : Shares) : Prop :=
  ∃ (x : ℝ),
    s.p = 5 * x ∧
    s.q = 11 * x ∧
    s.r = 19 * x ∧
    s.q - s.p = 12100

theorem money_division_theorem (s : Shares) 
  (h : MoneyDivisionProblem s) : s.r - s.q = 16133.36 := by
  sorry


end NUMINAMATH_CALUDE_money_division_theorem_l939_93959


namespace NUMINAMATH_CALUDE_probability_one_from_each_name_l939_93998

/-- The probability of selecting one letter from each name when drawing two cards without replacement -/
theorem probability_one_from_each_name (total_cards : ℕ) (amelia_cards : ℕ) (lucas_cards : ℕ) :
  total_cards = amelia_cards + lucas_cards →
  total_cards = 10 →
  amelia_cards = 6 →
  lucas_cards = 4 →
  (amelia_cards * lucas_cards : ℚ) / ((total_cards * (total_cards - 1)) / 2) = 8 / 15 := by
  sorry

end NUMINAMATH_CALUDE_probability_one_from_each_name_l939_93998


namespace NUMINAMATH_CALUDE_incorrect_propositions_l939_93942

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between lines and between a line and a plane
variable (parallel_line : Line → Line → Prop)
variable (parallel_plane : Line → Plane → Prop)
variable (contained_in : Line → Plane → Prop)

-- Define the intersection of two planes
variable (intersection : Plane → Plane → Line)

theorem incorrect_propositions :
  ∃ (l m : Line) (α β : Plane),
    -- Proposition A
    ¬(parallel_line l m ∧ contained_in m α → parallel_plane l α) ∧
    -- Proposition B
    ¬(parallel_plane l α ∧ parallel_plane m α → parallel_line l m) ∧
    -- Proposition C
    ¬(parallel_line l m ∧ parallel_plane m α → parallel_plane l α) :=
by sorry

end NUMINAMATH_CALUDE_incorrect_propositions_l939_93942


namespace NUMINAMATH_CALUDE_replaced_crew_member_weight_l939_93906

/-- Given a crew of 10 oarsmen, if replacing one member with a new member weighing 71 kg
    increases the average weight by 1.8 kg, then the replaced member weighed 53 kg. -/
theorem replaced_crew_member_weight
  (n : ℕ)
  (new_weight : ℝ)
  (avg_increase : ℝ)
  (h_crew_size : n = 10)
  (h_new_weight : new_weight = 71)
  (h_avg_increase : avg_increase = 1.8) :
  let old_total := n * (avg_increase + (new_weight - 53) / n)
  let new_total := n * (avg_increase + new_weight / n)
  new_total - old_total = n * avg_increase :=
by sorry

end NUMINAMATH_CALUDE_replaced_crew_member_weight_l939_93906


namespace NUMINAMATH_CALUDE_lucille_house_height_difference_l939_93944

/-- Proves that Lucille's house is 9.32 feet shorter than the average height of all houses. -/
theorem lucille_house_height_difference :
  let lucille_height : ℝ := 80
  let neighbor1_height : ℝ := 70.5
  let neighbor2_height : ℝ := 99.3
  let neighbor3_height : ℝ := 84.2
  let neighbor4_height : ℝ := 112.6
  let total_height : ℝ := lucille_height + neighbor1_height + neighbor2_height + neighbor3_height + neighbor4_height
  let average_height : ℝ := total_height / 5
  average_height - lucille_height = 9.32 := by
  sorry

#eval (80 + 70.5 + 99.3 + 84.2 + 112.6) / 5 - 80

end NUMINAMATH_CALUDE_lucille_house_height_difference_l939_93944


namespace NUMINAMATH_CALUDE_symmetric_function_minimum_value_l939_93926

-- Define the function f
def f (a b x : ℝ) : ℝ := (x - 1) * (x + 2) * (x^2 + a*x + b)

-- State the theorem
theorem symmetric_function_minimum_value (a b : ℝ) :
  (∀ x, f a b x = f a b (-x)) →  -- Symmetry condition
  (∃ x, ∀ y, f a b y ≥ f a b x ∧ f a b x = -9/4) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_function_minimum_value_l939_93926


namespace NUMINAMATH_CALUDE_parabola_point_ordering_l939_93952

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = -2 * (x - 1)^2 + 3

/-- Point A lies on the parabola -/
def point_A (y₁ : ℝ) : Prop := parabola (-3) y₁

/-- Point B lies on the parabola -/
def point_B (y₂ : ℝ) : Prop := parabola 2 y₂

/-- Theorem stating the ordering of y₁, y₂, and 3 -/
theorem parabola_point_ordering (y₁ y₂ : ℝ) 
  (hA : point_A y₁) (hB : point_B y₂) : y₁ < y₂ ∧ y₂ < 3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_ordering_l939_93952


namespace NUMINAMATH_CALUDE_transformation_has_integer_root_intermediate_l939_93922

/-- Represents a quadratic equation ax^2 + bx + c = 0 -/
structure QuadraticEquation where
  a : ℤ
  b : ℤ
  c : ℤ

/-- Checks if a quadratic equation has integer roots -/
def has_integer_root (eq : QuadraticEquation) : Prop :=
  ∃ x : ℤ, eq.a * x^2 + eq.b * x + eq.c = 0

/-- Represents a single step in the transformation process -/
inductive TransformationStep
  | IncreaseP
  | DecreaseP
  | IncreaseQ
  | DecreaseQ

/-- Applies a transformation step to a quadratic equation -/
def apply_step (eq : QuadraticEquation) (step : TransformationStep) : QuadraticEquation :=
  match step with
  | TransformationStep.IncreaseP => ⟨eq.a, eq.b + 1, eq.c⟩
  | TransformationStep.DecreaseP => ⟨eq.a, eq.b - 1, eq.c⟩
  | TransformationStep.IncreaseQ => ⟨eq.a, eq.b, eq.c + 1⟩
  | TransformationStep.DecreaseQ => ⟨eq.a, eq.b, eq.c - 1⟩

theorem transformation_has_integer_root_intermediate 
  (initial : QuadraticEquation) 
  (final : QuadraticEquation) 
  (h_initial : initial = ⟨1, -2013, -13⟩) 
  (h_final : final = ⟨1, 13, 2013⟩) :
  ∀ steps : List TransformationStep, 
    (List.foldl apply_step initial steps = final) → 
    (∃ intermediate : QuadraticEquation, 
      intermediate ∈ List.scanl apply_step initial steps ∧ 
      has_integer_root intermediate) :=
sorry

end NUMINAMATH_CALUDE_transformation_has_integer_root_intermediate_l939_93922


namespace NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l939_93945

/-- Given a geometric sequence with first term a and common ratio r,
    the nth term is given by a * r^(n-1) -/
def geometric_sequence (a r : ℚ) (n : ℕ) : ℚ := a * r^(n-1)

/-- The common ratio of a geometric sequence can be found by dividing
    the second term by the first term -/
def common_ratio (a₁ a₂ : ℚ) : ℚ := a₂ / a₁

theorem seventh_term_of_geometric_sequence (a₁ a₂ : ℚ) 
  (h₁ : a₁ = 3)
  (h₂ : a₂ = -3/2) :
  geometric_sequence a₁ (common_ratio a₁ a₂) 7 = 3/64 := by
  sorry


end NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l939_93945


namespace NUMINAMATH_CALUDE_total_value_after_depreciation_l939_93927

def calculate_depreciated_value (initial_value : ℝ) (depreciation_rate : ℝ) (years : ℕ) : ℝ :=
  initial_value * (1 - depreciation_rate) ^ years

theorem total_value_after_depreciation 
  (machine1_value : ℝ) (machine2_value : ℝ) (machine3_value : ℝ)
  (machine1_rate : ℝ) (machine2_rate : ℝ) (machine3_rate : ℝ)
  (years : ℕ) :
  machine1_value = 2500 →
  machine2_value = 3500 →
  machine3_value = 4500 →
  machine1_rate = 0.05 →
  machine2_rate = 0.07 →
  machine3_rate = 0.04 →
  years = 3 →
  (calculate_depreciated_value machine1_value machine1_rate years +
   calculate_depreciated_value machine2_value machine2_rate years +
   calculate_depreciated_value machine3_value machine3_rate years) = 8940 := by
  sorry

end NUMINAMATH_CALUDE_total_value_after_depreciation_l939_93927


namespace NUMINAMATH_CALUDE_euler_polynomial_consecutive_composites_l939_93951

theorem euler_polynomial_consecutive_composites :
  ∃ k : ℤ, ∀ j ∈ Finset.range 40,
    ∃ d : ℤ, d ∣ ((k + j)^2 + (k + j) + 41) ∧ d ≠ 1 ∧ d ≠ ((k + j)^2 + (k + j) + 41) := by
  sorry

end NUMINAMATH_CALUDE_euler_polynomial_consecutive_composites_l939_93951


namespace NUMINAMATH_CALUDE_diamond_equation_solution_l939_93953

-- Define the binary operation
noncomputable def diamond (a b : ℝ) : ℝ := sorry

-- Define the properties of the operation
axiom diamond_assoc (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  diamond a (diamond b c) = (diamond a b) ^ c

axiom diamond_self (a : ℝ) (ha : a ≠ 0) : diamond a a = 1

-- State the theorem
theorem diamond_equation_solution :
  ∃! x : ℝ, x ≠ 0 ∧ diamond 2048 (diamond 4 x) = 16 := by sorry

end NUMINAMATH_CALUDE_diamond_equation_solution_l939_93953


namespace NUMINAMATH_CALUDE_marble_arrangement_theorem_l939_93980

/-- The number of green marbles -/
def green_marbles : ℕ := 6

/-- The maximum number of red marbles that satisfies the arrangement condition -/
def max_red_marbles : ℕ := 18

/-- The total number of marbles in the arrangement -/
def total_marbles : ℕ := green_marbles + max_red_marbles

/-- The number of ways to arrange the marbles -/
def arrangement_count : ℕ := Nat.choose total_marbles green_marbles

theorem marble_arrangement_theorem :
  arrangement_count % 1000 = 564 := by sorry

end NUMINAMATH_CALUDE_marble_arrangement_theorem_l939_93980


namespace NUMINAMATH_CALUDE_triangle_area_l939_93907

theorem triangle_area (a b : ℝ) (θ : ℝ) (h1 : a = 3) (h2 : b = 2) (h3 : θ = π / 3) :
  (1 / 2) * a * b * Real.sin θ = (3 / 2) * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l939_93907


namespace NUMINAMATH_CALUDE_inequality_not_always_satisfied_l939_93978

theorem inequality_not_always_satisfied :
  ∃ (p q r : ℝ), p < 1 ∧ q < 2 ∧ r < 3 ∧ p^2 + 2*q*r ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_not_always_satisfied_l939_93978


namespace NUMINAMATH_CALUDE_carnival_tickets_total_l939_93939

/-- Represents the number of tickets used for a carnival ride -/
structure RideTickets where
  ferrisWheel : ℕ
  bumperCars : ℕ
  rollerCoaster : ℕ

/-- Calculates the total number of tickets used for a set of rides -/
def totalTickets (rides : RideTickets) (ferrisWheelCost bumperCarsCost rollerCoasterCost : ℕ) : ℕ :=
  rides.ferrisWheel * ferrisWheelCost + rides.bumperCars * bumperCarsCost + rides.rollerCoaster * rollerCoasterCost

/-- Theorem stating the total number of tickets used by Oliver, Emma, and Sophia -/
theorem carnival_tickets_total : 
  let ferrisWheelCost := 7
  let bumperCarsCost := 5
  let rollerCoasterCost := 9
  let oliver := RideTickets.mk 5 4 0
  let emma := RideTickets.mk 0 6 3
  let sophia := RideTickets.mk 3 2 2
  totalTickets oliver ferrisWheelCost bumperCarsCost rollerCoasterCost +
  totalTickets emma ferrisWheelCost bumperCarsCost rollerCoasterCost +
  totalTickets sophia ferrisWheelCost bumperCarsCost rollerCoasterCost = 161 := by
  sorry

end NUMINAMATH_CALUDE_carnival_tickets_total_l939_93939


namespace NUMINAMATH_CALUDE_sum_of_roots_absolute_value_equation_l939_93983

theorem sum_of_roots_absolute_value_equation : 
  ∃ (r₁ r₂ r₃ : ℝ), 
    (∀ x : ℝ, (|x + 3| - |x - 1| = x + 1) ↔ (x = r₁ ∨ x = r₂ ∨ x = r₃)) ∧ 
    r₁ + r₂ + r₃ = -3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_absolute_value_equation_l939_93983


namespace NUMINAMATH_CALUDE_seventh_row_cans_l939_93936

/-- Represents a triangular display of cans -/
structure CanDisplay where
  rows : Nat
  first_row_cans : Nat
  increment : Nat

/-- Calculate the number of cans in a specific row -/
def cans_in_row (d : CanDisplay) (row : Nat) : Nat :=
  d.first_row_cans + d.increment * (row - 1)

/-- Calculate the total number of cans in the display -/
def total_cans (d : CanDisplay) : Nat :=
  (d.rows * (2 * d.first_row_cans + (d.rows - 1) * d.increment)) / 2

/-- The main theorem -/
theorem seventh_row_cans (d : CanDisplay) :
  d.rows = 10 ∧ d.increment = 3 ∧ total_cans d < 150 →
  cans_in_row d 7 = 19 := by
  sorry

#eval cans_in_row { rows := 10, first_row_cans := 1, increment := 3 } 7

end NUMINAMATH_CALUDE_seventh_row_cans_l939_93936


namespace NUMINAMATH_CALUDE_janes_change_is_correct_l939_93973

/-- The change Jane receives when buying an apple -/
def janes_change (apple_price : ℚ) (paid_amount : ℚ) : ℚ :=
  paid_amount - apple_price

/-- Theorem: Jane receives $4.25 in change -/
theorem janes_change_is_correct : 
  janes_change 0.75 5.00 = 4.25 := by
  sorry

end NUMINAMATH_CALUDE_janes_change_is_correct_l939_93973


namespace NUMINAMATH_CALUDE_inequality_solution_length_l939_93938

theorem inequality_solution_length (c d : ℝ) : 
  (∀ x : ℝ, d ≤ x^2 + 4*x + 3 ∧ x^2 + 4*x + 3 ≤ c) →
  (∃ a b : ℝ, ∀ x : ℝ, (d ≤ x^2 + 4*x + 3 ∧ x^2 + 4*x + 3 ≤ c) ↔ (a ≤ x ∧ x ≤ b)) →
  (∃ a b : ℝ, b - a = 8 ∧ ∀ x : ℝ, (d ≤ x^2 + 4*x + 3 ∧ x^2 + 4*x + 3 ≤ c) ↔ (a ≤ x ∧ x ≤ b)) →
  c - d = 0 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_length_l939_93938


namespace NUMINAMATH_CALUDE_three_layer_runner_area_l939_93925

/-- Given three table runners with a combined area of 204 square inches covering 80% of a table
    with an area of 175 square inches, and an area of 24 square inches covered by exactly two
    layers of runner, prove that the area covered by three layers of runner is 20 square inches. -/
theorem three_layer_runner_area
  (total_runner_area : ℝ)
  (table_area : ℝ)
  (coverage_percent : ℝ)
  (two_layer_area : ℝ)
  (h1 : total_runner_area = 204)
  (h2 : table_area = 175)
  (h3 : coverage_percent = 0.8)
  (h4 : two_layer_area = 24)
  : ∃ (three_layer_area : ℝ),
    three_layer_area = 20 ∧
    coverage_percent * table_area = (total_runner_area - two_layer_area - three_layer_area) + 2 * two_layer_area + 3 * three_layer_area :=
by sorry

end NUMINAMATH_CALUDE_three_layer_runner_area_l939_93925


namespace NUMINAMATH_CALUDE_kim_total_sweaters_l939_93933

/-- The number of sweaters Kim knit on each day of the week --/
structure SweaterCount where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ

/-- The conditions of Kim's sweater knitting for the week --/
def kim_sweater_conditions (sc : SweaterCount) : Prop :=
  sc.monday = 8 ∧
  sc.tuesday = sc.monday + 2 ∧
  sc.wednesday = sc.tuesday - 4 ∧
  sc.thursday = sc.wednesday ∧
  sc.friday = sc.monday / 2

/-- The theorem stating the total number of sweaters Kim knit that week --/
theorem kim_total_sweaters (sc : SweaterCount) 
  (h : kim_sweater_conditions sc) : 
  sc.monday + sc.tuesday + sc.wednesday + sc.thursday + sc.friday = 34 := by
  sorry


end NUMINAMATH_CALUDE_kim_total_sweaters_l939_93933


namespace NUMINAMATH_CALUDE_polygon_area_is_7_5_l939_93921

/-- Calculates the area of a polygon using the Shoelace formula -/
def polygonArea (vertices : List (ℝ × ℝ)) : ℝ :=
  let n := vertices.length
  let pairs := List.zip vertices (vertices.rotate 1)
  0.5 * (pairs.foldl (fun sum (v1, v2) => sum + v1.1 * v2.2 - v1.2 * v2.1) 0)

theorem polygon_area_is_7_5 :
  let vertices := [(2, 1), (4, 3), (7, 1), (4, 6)]
  polygonArea vertices = 7.5 := by
  sorry

#eval polygonArea [(2, 1), (4, 3), (7, 1), (4, 6)]

end NUMINAMATH_CALUDE_polygon_area_is_7_5_l939_93921


namespace NUMINAMATH_CALUDE_hotel_profit_theorem_l939_93977

/-- Calculates the hotel's weekly profit given the operations expenses and service percentages --/
def hotel_profit (operations_expenses : ℚ) 
  (meetings_percent : ℚ) (events_percent : ℚ) (rooms_percent : ℚ)
  (meetings_tax : ℚ) (meetings_commission : ℚ)
  (events_tax : ℚ) (events_commission : ℚ)
  (rooms_tax : ℚ) (rooms_commission : ℚ) : ℚ :=
  let meetings_income := meetings_percent * operations_expenses
  let events_income := events_percent * operations_expenses
  let rooms_income := rooms_percent * operations_expenses
  let total_income := meetings_income + events_income + rooms_income
  let meetings_additional := meetings_income * (meetings_tax + meetings_commission)
  let events_additional := events_income * (events_tax + events_commission)
  let rooms_additional := rooms_income * (rooms_tax + rooms_commission)
  let total_additional := meetings_additional + events_additional + rooms_additional
  total_income - operations_expenses - total_additional

/-- The hotel's weekly profit is $1,283.75 given the specified conditions --/
theorem hotel_profit_theorem : 
  hotel_profit 5000 (5/8) (3/10) (11/20) (1/10) (1/20) (2/25) (3/50) (3/25) (3/100) = 1283.75 := by
  sorry


end NUMINAMATH_CALUDE_hotel_profit_theorem_l939_93977


namespace NUMINAMATH_CALUDE_book_arrangement_combinations_l939_93971

-- Define the number of each type of book
def geometry_books : ℕ := 4
def number_theory_books : ℕ := 5

-- Define the total number of books
def total_books : ℕ := geometry_books + number_theory_books

-- Define the number of remaining spots after placing the first geometry book
def remaining_spots : ℕ := total_books - 1

-- Define the number of remaining geometry books to place
def remaining_geometry_books : ℕ := geometry_books - 1

-- Theorem statement
theorem book_arrangement_combinations :
  (remaining_spots.choose remaining_geometry_books) = 56 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_combinations_l939_93971


namespace NUMINAMATH_CALUDE_fraction_sum_minus_five_equals_negative_four_l939_93928

theorem fraction_sum_minus_five_equals_negative_four (a b : ℝ) (h : a ≠ b) :
  a / (a - b) + b / (b - a) - 5 = -4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_minus_five_equals_negative_four_l939_93928


namespace NUMINAMATH_CALUDE_p_squared_plus_26_composite_l939_93904

theorem p_squared_plus_26_composite (p : Nat) (hp : Prime p) : 
  ∃ (a b : Nat), a > 1 ∧ b > 1 ∧ p^2 + 26 = a * b :=
sorry

end NUMINAMATH_CALUDE_p_squared_plus_26_composite_l939_93904


namespace NUMINAMATH_CALUDE_A_minus_2B_y_value_when_independent_l939_93949

-- Define the expressions A and B
def A (x y : ℝ) : ℝ := 4 * x^2 - x * y + 2 * y
def B (x y : ℝ) : ℝ := 2 * x^2 - x * y + x

-- Theorem 1: A - 2B = xy - 2x + 2y
theorem A_minus_2B (x y : ℝ) : A x y - 2 * B x y = x * y - 2 * x + 2 * y := by sorry

-- Theorem 2: If A - 2B is independent of x, then y = 2
theorem y_value_when_independent (y : ℝ) : 
  (∀ x : ℝ, A x y - 2 * B x y = A 0 y - 2 * B 0 y) → y = 2 := by sorry

end NUMINAMATH_CALUDE_A_minus_2B_y_value_when_independent_l939_93949


namespace NUMINAMATH_CALUDE_square_roots_sum_zero_l939_93997

theorem square_roots_sum_zero (x a : ℝ) : 
  x > 0 → (a + 1) ^ 2 = x → (a - 3) ^ 2 = x → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_roots_sum_zero_l939_93997


namespace NUMINAMATH_CALUDE_annual_increase_fraction_l939_93974

theorem annual_increase_fraction (initial_amount final_amount : ℝ) 
  (h1 : initial_amount > 0)
  (h2 : final_amount > initial_amount)
  (h3 : initial_amount * (1 + f)^2 = final_amount)
  (h4 : initial_amount = 57600)
  (h5 : final_amount = 72900) : 
  f = 0.125 := by
sorry

end NUMINAMATH_CALUDE_annual_increase_fraction_l939_93974


namespace NUMINAMATH_CALUDE_factorization_equality_l939_93910

theorem factorization_equality (x y : ℝ) : x^2 + x*y + x = x*(x + y + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l939_93910


namespace NUMINAMATH_CALUDE_inequality_solution_implies_m_range_l939_93993

theorem inequality_solution_implies_m_range 
  (m : ℝ) 
  (h : ∀ x : ℝ, (m * x - 1) * (x - 2) > 0 ↔ (1 / m < x ∧ x < 2)) : 
  m < 0 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_implies_m_range_l939_93993


namespace NUMINAMATH_CALUDE_unique_solution_congruences_l939_93948

theorem unique_solution_congruences :
  ∃! x : ℕ, x < 120 ∧
    (4 + x) % 8 = 3^2 % 8 ∧
    (6 + x) % 27 = 4^2 % 27 ∧
    (8 + x) % 125 = 6^2 % 125 ∧
    x = 37 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_congruences_l939_93948


namespace NUMINAMATH_CALUDE_x_value_proof_l939_93955

theorem x_value_proof (x y : ℚ) : x / y = 12 / 5 → y = 20 → x = 48 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l939_93955


namespace NUMINAMATH_CALUDE_system_solution_l939_93911

theorem system_solution : ∃ (x y : ℝ), 2*x - 3*y = -7 ∧ 5*x + 4*y = -6 ∧ (x, y) = (-2, 1) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l939_93911


namespace NUMINAMATH_CALUDE_stratified_sampling_medium_supermarkets_l939_93929

theorem stratified_sampling_medium_supermarkets 
  (large : ℕ) 
  (medium : ℕ) 
  (small : ℕ) 
  (sample_size : ℕ) 
  (h1 : large = 200) 
  (h2 : medium = 400) 
  (h3 : small = 1400) 
  (h4 : sample_size = 100) :
  (medium : ℚ) * sample_size / (large + medium + small) = 20 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_medium_supermarkets_l939_93929


namespace NUMINAMATH_CALUDE_unique_parallel_line_in_plane_l939_93968

/-- A plane in 3D space -/
structure Plane3D where
  -- (Placeholder for plane definition)

/-- A line in 3D space -/
structure Line3D where
  -- (Placeholder for line definition)

/-- A point in 3D space -/
structure Point3D where
  -- (Placeholder for point definition)

/-- Predicate for a line being parallel to a plane -/
def parallel_line_plane (l : Line3D) (α : Plane3D) : Prop :=
  sorry

/-- Predicate for a point being on a plane -/
def point_on_plane (P : Point3D) (α : Plane3D) : Prop :=
  sorry

/-- Predicate for a line passing through a point -/
def line_through_point (l : Line3D) (P : Point3D) : Prop :=
  sorry

/-- Predicate for two lines being parallel -/
def parallel_lines (l1 l2 : Line3D) : Prop :=
  sorry

/-- Predicate for a line lying in a plane -/
def line_in_plane (l : Line3D) (α : Plane3D) : Prop :=
  sorry

theorem unique_parallel_line_in_plane 
  (l : Line3D) (α : Plane3D) (P : Point3D)
  (h1 : parallel_line_plane l α)
  (h2 : point_on_plane P α) :
  ∃! m : Line3D, line_through_point m P ∧ parallel_lines m l ∧ line_in_plane m α :=
sorry

end NUMINAMATH_CALUDE_unique_parallel_line_in_plane_l939_93968


namespace NUMINAMATH_CALUDE_museum_trip_total_people_l939_93920

theorem museum_trip_total_people : 
  let first_bus : ℕ := 12
  let second_bus : ℕ := 2 * first_bus
  let third_bus : ℕ := second_bus - 6
  let fourth_bus : ℕ := first_bus + 9
  first_bus + second_bus + third_bus + fourth_bus = 75
  := by sorry

end NUMINAMATH_CALUDE_museum_trip_total_people_l939_93920


namespace NUMINAMATH_CALUDE_range_of_a_when_p_is_false_l939_93969

theorem range_of_a_when_p_is_false :
  (∀ m : ℝ, -1 ≤ m ∧ m ≤ 1 → a^2 - 5*a + 3 ≥ m + 2) →
  a ∈ Set.Iic 0 ∪ Set.Ici 5 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_when_p_is_false_l939_93969


namespace NUMINAMATH_CALUDE_quadratic_factorization_l939_93985

theorem quadratic_factorization (x : ℝ) : 4 * x^2 - 8 * x + 4 = 4 * (x - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l939_93985


namespace NUMINAMATH_CALUDE_periodic_function_value_l939_93991

def is_periodic (f : ℝ → ℝ) (period : ℝ) : Prop :=
  ∀ x, f (x + period) = f x

theorem periodic_function_value (f : ℝ → ℝ) :
  is_periodic f 3 →
  (∀ x ∈ Set.Icc (-1) 2, f x = x + 1) →
  f 2017 = 2 := by
sorry

end NUMINAMATH_CALUDE_periodic_function_value_l939_93991


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l939_93947

/-- Represents the number of employees in each age group -/
structure EmployeeCount where
  total : ℕ
  young : ℕ
  middleAged : ℕ
  elderly : ℕ

/-- Calculates the number of elderly employees to sample given the number of young employees sampled -/
def elderlyToSample (ec : EmployeeCount) (youngSampled : ℕ) : ℕ :=
  (youngSampled * ec.elderly) / ec.young

/-- Theorem stating that given the specific employee counts and 7 young employees sampled, 
    3 elderly employees should be sampled -/
theorem stratified_sampling_theorem (ec : EmployeeCount) 
  (h1 : ec.total = 750)
  (h2 : ec.young = 350)
  (h3 : ec.middleAged = 250)
  (h4 : ec.elderly = 150)
  (h5 : ec.total = ec.young + ec.middleAged + ec.elderly) :
  elderlyToSample ec 7 = 3 := by
  sorry

#eval elderlyToSample { total := 750, young := 350, middleAged := 250, elderly := 150 } 7

end NUMINAMATH_CALUDE_stratified_sampling_theorem_l939_93947


namespace NUMINAMATH_CALUDE_trig_power_sum_l939_93923

theorem trig_power_sum (x : Real) 
  (h : Real.sin x ^ 10 + Real.cos x ^ 10 = 11/36) : 
  Real.sin x ^ 14 + Real.cos x ^ 14 = 41/216 := by
  sorry

end NUMINAMATH_CALUDE_trig_power_sum_l939_93923


namespace NUMINAMATH_CALUDE_elvins_internet_charge_l939_93961

/-- Proves that the fixed monthly charge for internet service is $6 given the conditions of Elvin's telephone bills. -/
theorem elvins_internet_charge (january_bill february_bill : ℕ) 
  (h1 : january_bill = 48)
  (h2 : february_bill = 90)
  (fixed_charge : ℕ) (january_calls february_calls : ℕ)
  (h3 : february_calls = 2 * january_calls)
  (h4 : january_bill = fixed_charge + january_calls)
  (h5 : february_bill = fixed_charge + february_calls) :
  fixed_charge = 6 := by
  sorry

end NUMINAMATH_CALUDE_elvins_internet_charge_l939_93961


namespace NUMINAMATH_CALUDE_urn_theorem_l939_93918

/-- Represents the state of the urn -/
structure UrnState where
  black : ℕ
  white : ℕ

/-- Represents the four possible operations -/
inductive Operation
  | RemoveBlack
  | RemoveBlackWhite
  | RemoveBlackAddWhite
  | RemoveWhiteAddBlack

/-- Applies an operation to the urn state -/
def applyOperation (state : UrnState) (op : Operation) : UrnState :=
  match op with
  | Operation.RemoveBlack => ⟨state.black - 1, state.white⟩
  | Operation.RemoveBlackWhite => ⟨state.black, state.white - 1⟩
  | Operation.RemoveBlackAddWhite => ⟨state.black - 1, state.white⟩
  | Operation.RemoveWhiteAddBlack => ⟨state.black + 1, state.white - 1⟩

/-- Checks if the given state is reachable from the initial state -/
def isReachable (initialState : UrnState) (targetState : UrnState) : Prop :=
  ∃ (n : ℕ) (ops : Fin n → Operation),
    (List.foldl applyOperation initialState (List.ofFn ops)) = targetState

/-- The theorem to be proven -/
theorem urn_theorem :
  let initialState : UrnState := ⟨150, 150⟩
  let targetState : UrnState := ⟨50, 50⟩
  isReachable initialState targetState :=
sorry

end NUMINAMATH_CALUDE_urn_theorem_l939_93918
